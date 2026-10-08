#!/usr/bin/env python3
"""
QC a batch of CMG STARS SLURM logs.

Scans a ``sim_logs/<batch>_log`` folder of ``slurm-<jobid>_<task>.out`` files and
reports, per case:

  * termination status and whether the run reached the target end time
  * producer / injector rate imbalance (final, cumulative, and worst sustained)
  * water material-balance error and numerical-stability counters

Two failure modes motivated this script, both from batch 260831:

  * **Early termination** -- ``FATAL ERROR (from subroutine: PRTOUT): Too many
    consecutive timestep cuts``. The run stops short of the target end time and
    any array extracted from its .sr3 is silently short along the time axis.
  * **Imbalanced rate** -- the injector holds its target while the producer is
    constraint-limited, so the reservoir accumulates water for decades. The run
    terminates *normally*, so a status-only check misses it entirely.

Note that STARS reports water production at the start of a timestep and water
injection at its end, so the two columns cannot be differenced within a row; see
:func:`_imbalance_metrics` for the correction.

Usage
-----
    python src/qc_sim_logs.py results/Vienna_geothermal/sim_logs/260831_log
    python src/qc_sim_logs.py <log_dir> --npy-dir <sim_py_dir> --csv qc.csv
    python src/qc_sim_logs.py <log_dir> --all        # every case, not just flagged

In a notebook::

    from src.qc_sim_logs import scan_batch, to_dataframe
    df = to_dataframe(scan_batch("results/Vienna_geothermal/sim_logs/260831_log"))
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# CMG writes compact floats such as "3799e4" (= 3.799e7) and "22e-3" (= 0.022),
# i.e. a mantissa with no decimal point followed by an exponent.
_COMPACT = re.compile(r"^(-?\d*\.?\d+)e(-?\d+)$")
_LOG_NAME = re.compile(r"slurm-(\d+)_(\d+)\.out$")
_DASHES = re.compile(r"^[\s-]*-{4,}[\s-]*$")
_DATE = re.compile(r"\d{4}/\d{2}/\d{2}")

# Column indices in the STARS "TIME STEP SUMMARY" table, counted from the
# dashed rule under the header. 17 columns:
#   0 No.  1 Size  2 IT  3 Cuts  4 days  5 yy/mm/dd  6 Oil  7 Gas  8 Water
#   9 GOR  10 WatCut  11 GasInj  12 WaterInj  13 MatBalErr  14 Pres  15 Sat  16 Temp
_COL_CUTS, _COL_DAYS = 3, 4
_COL_WPROD, _COL_WINJ = 8, 12
_COL_MATBAL, _COL_TEMP = 13, 16


def _parse_float(token: str) -> Optional[float]:
    """
    Parse one numeric token from a CMG log, including its compact exponent form.

    Args:
        token: Raw text of a single table or summary field.

    Returns:
        The value as a float, or None if the token is blank or non-numeric
        (CMG uses "NA" and empty columns for phases that are not present).
    """
    token = token.strip()
    if not token or token == "NA":
        return None
    match = _COMPACT.match(token)
    if match:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    try:
        return float(token)
    except ValueError:
        return None


def _column_spans(rule: str) -> List[Tuple[int, int]]:
    """
    Derive fixed-width column boundaries from a table's dashed rule.

    Fixed-width slicing is required because STARS leaves oil, gas, GOR and water-cut
    columns blank in a single-phase water run; whitespace tokenising would then
    shift the injection rate into the production slot.

    Args:
        rule: The line of dashes that sits directly under the table header.

    Returns:
        List of (start, end) character offsets, one per column.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"-+", rule)]


def _summary_water(text: str, label: str) -> Optional[float]:
    """
    Read the water column from a "Run Summary Information" line.

    The layout is ``<label>  <oil>  <gas>  <water>  NA  NA``, so the water
    value is the third field from the end.

    Args:
        text: Full log text.
        label: Line label, e.g. "Cumulative Production" or "Production Rates".

    Returns:
        The water value, or None if the label is absent.
    """
    match = re.search(rf"^\s*{re.escape(label)}\s+(.*)$", text, re.MULTILINE)
    if not match:
        return None
    fields = match.group(1).split()
    return _parse_float(fields[-3]) if len(fields) >= 3 else None


def _timestep_table(text: str) -> List[Dict[str, Optional[float]]]:
    """
    Extract the per-timestep history from all repeats of the TIME STEP SUMMARY table.

    Args:
        text: Full log text.

    Returns:
        List of dicts with keys ``day``, ``wprod``, ``winj``, ``matbal``, ``temp``,
        ``cuts``, in file order. Empty if no table rule is found.
    """
    lines = text.splitlines()
    spans: List[Tuple[int, int]] = []
    rows: List[Dict[str, Optional[float]]] = []

    for line in lines:
        # The widest dashed rule in the file is the timestep table's; capture it once.
        if _DASHES.match(line) and line.count("-") > 60:
            spans = _column_spans(line)
            continue
        if not spans or not line[:8].strip():
            continue
        if not re.match(r"^\s*\d+w?\s", line):
            continue

        cells = [line[a:b] if a < len(line) else "" for a, b in spans]
        if len(cells) <= _COL_TEMP or not _DATE.search(line):
            continue
        rows.append(
            {
                "day": _parse_float(cells[_COL_DAYS]),
                "wprod": _parse_float(cells[_COL_WPROD]),
                "winj": _parse_float(cells[_COL_WINJ]),
                "matbal": _parse_float(cells[_COL_MATBAL]),
                "temp": _parse_float(cells[_COL_TEMP]),
                "cuts": _parse_float(cells[_COL_CUTS]),
            }
        )
    return rows


def _imbalance_metrics(
    rows: Sequence[Dict[str, Optional[float]]],
    thresholds: Sequence[float] = (5.0, 20.0),
    closure_pct: float = 10.0,
    lag_tol_pct: float = 0.5,
) -> Dict[str, Optional[float]]:
    """
    Summarise producer/injector rate imbalance over the timestep history.

    STARS prints water production at the *start* of a timestep and water injection
    at its *end*, so in a balanced doublet ``wprod(n)`` equals ``winj(n-1)`` to
    every printed digit. Comparing the two columns *within* a row therefore
    measures how fast the rate is declining across that step, not a
    producer/injector mismatch. That manufactured an apparent 5-15% imbalance in
    about a dozen cases of batch 260831-2, including one (case40) that is balanced
    exactly once the lag is undone.

    Each row is scored twice -- lag-corrected (``wprod(n)`` against ``winj(n-1)``)
    and same-row -- and the *smaller* magnitude is kept. A real deficit holds
    injection roughly steady, so both comparisons agree and the signal survives;
    the reporting lag only ever inflates the same-row comparison, so it is always
    discarded. Taking the minimum also absorbs the rows just after a timestep cut,
    where the retried step breaks row-to-row alignment.

    Imbalance is signed: negative means the producer is running below the injector,
    i.e. the reservoir is accumulating water.

    Args:
        rows: Timestep records from :func:`_timestep_table`.
        thresholds: Imbalance percentages at which to accumulate elapsed time.
        closure_pct: Imbalance percentage used to report the last day the
            imbalance exceeds it. This is what separates a long startup transient
            (closes after days) from a deficit that runs for the whole simulation.
        lag_tol_pct: Deviation under which ``wprod(n)`` counts as matching
            ``winj(n-1)``, for the diagnostic match fraction.

    Returns:
        Dict with ``peak_imbal_pct`` (signed), ``peak_imbal_day``,
        ``imbal_closes_day``, ``lag_match_pct``, and one ``frac_time_gt<N>pct``
        entry per threshold.
    """
    out: Dict[str, Optional[float]] = {
        "peak_imbal_pct": None,
        "peak_imbal_day": None,
        "imbal_closes_day": None,
        "lag_match_pct": None,
    }
    for thr in thresholds:
        out[f"frac_time_gt{int(thr)}pct"] = None

    total = next((r["day"] for r in reversed(rows) if r["day"] is not None), None)
    if not total:
        return out

    above = {thr: 0.0 for thr in thresholds}
    peak, peak_day, closes_day, prev_day = 0.0, None, None, 0.0
    lag_hits = lag_rows = 0
    prev_winj: Optional[float] = None

    for row in rows:
        day, wprod, winj = row["day"], row["wprod"], row["winj"]
        # Carry the previous row's injection rate before anything can skip ahead,
        # so the lag reference stays aligned to the row immediately above.
        winj_before, prev_winj = prev_winj, winj
        if day is None:
            continue
        step = max(day - prev_day, 0.0)
        prev_day = day
        if wprod is None:
            continue  # blank production column on a crashed final step

        same_row = (wprod - winj) / abs(winj) * 100.0 if winj else None
        lagged = (
            (wprod - winj_before) / abs(winj_before) * 100.0 if winj_before else None
        )
        if lagged is not None:
            lag_rows += 1
            lag_hits += abs(lagged) < lag_tol_pct

        candidates = [d for d in (same_row, lagged) if d is not None]
        if not candidates:
            continue
        imbal = min(candidates, key=abs)

        if abs(imbal) > abs(peak):
            peak, peak_day = imbal, day
        for thr in thresholds:
            if abs(imbal) > thr:
                above[thr] += step
        if abs(imbal) > closure_pct:
            closes_day = day

    out["peak_imbal_pct"] = peak
    out["peak_imbal_day"] = peak_day
    out["imbal_closes_day"] = closes_day
    out["lag_match_pct"] = lag_hits / lag_rows * 100.0 if lag_rows else None
    for thr in thresholds:
        out[f"frac_time_gt{int(thr)}pct"] = above[thr] / total * 100.0
    return out


def npy_time_steps(npy_path: Path) -> Optional[int]:
    """
    Read the trailing (time) axis length from a .npy header without loading data.

    Only the 128-byte header is read, which matters when the arrays live on a
    network or cloud-synced filesystem where mmap of 80 files is slow.

    Args:
        npy_path: Path to a .npy file.

    Returns:
        Length of the last axis, or None if the file is missing or unparsable.
    """
    try:
        with open(npy_path, "rb") as handle:
            header = handle.read(256)
    except OSError:
        return None
    match = re.search(rb"'shape':\s*\(([^)]*)\)", header)
    if not match:
        return None
    dims = [d.strip() for d in match.group(1).split(b",") if d.strip()]
    return int(dims[-1]) if dims else None


def parse_log(path: Path) -> Dict[str, object]:
    """
    Parse a single SLURM .out file from a STARS run.

    Args:
        path: Path to a ``slurm-<jobid>_<task>.out`` file.

    Returns:
        Dict of QC fields for the case (see :func:`scan_batch`).
    """
    text = path.read_text(errors="ignore")
    name = _LOG_NAME.search(path.name)

    status = re.search(r"End of Simulation:\s*(\w+)\s+Termination", text)
    fatal = re.search(
        r"FATAL ERROR \(from subroutine: (\w+)\)\s*=*\s*\n\s*(.+?)\n", text
    )
    end_time = re.search(r"Time at end of simulation:\s*([\d.]+)", text)
    matbal = re.search(r"Material Balances \(owge\):\s*(.*)", text)
    counters = {
        key: re.search(rf"{label}:\s*(\d+)", text)
        for key, label in (
            ("timesteps", r"Timesteps"),
            ("cuts", r"Cuts"),
            ("solver_failures", r"Total Number of Solver Failures"),
        )
    }

    cum_prod = _summary_water(text, "Cumulative Production")
    cum_inj = _summary_water(text, "Cumulative Injection")
    rate_prod = _summary_water(text, "Production Rates")
    rate_inj = _summary_water(text, "Injection Rates")

    rows = _timestep_table(text)
    record: Dict[str, object] = {
        "case": int(name.group(2)) if name else None,
        "job_id": name.group(1) if name else None,
        "log": path.name,
        "status": status.group(1) if status else "MISSING",
        "fatal_subroutine": fatal.group(1) if fatal else None,
        "fatal_message": fatal.group(2).strip() if fatal else None,
        "end_time_day": _parse_float(end_time.group(1)) if end_time else None,
        "cum_prod_m3": cum_prod,
        "cum_inj_m3": cum_inj,
        "rate_prod_m3d": rate_prod,
        "rate_inj_m3d": rate_inj,
        "matbal_water_pct": (
            _parse_float(matbal.group(1).split()[1])
            if matbal and len(matbal.group(1).split()) > 1
            else None
        ),
        "n_rows_parsed": len(rows),
    }
    for key, match in counters.items():
        record[key] = int(match.group(1)) if match else None

    record["cum_imbal_pct"] = (
        (cum_prod - cum_inj) / cum_inj * 100.0
        if cum_prod is not None and cum_inj
        else None
    )
    # A crashed run reports its failed final iterate here, often with a negative
    # production rate; that is a solver artefact, not a physical result.
    record["final_rate_imbal_pct"] = (
        (rate_prod - rate_inj) / rate_inj * 100.0
        if rate_prod is not None and rate_inj
        else None
    )
    record["max_temp_change_degC"] = (
        min((r["temp"] for r in rows if r["temp"] is not None), default=None)
    )
    record.update(_imbalance_metrics(rows))
    return record


def scan_batch(
    log_dir: str | Path,
    npy_dir: str | Path | None = None,
    npy_property: str = "PRES",
    target_end_day: float | None = None,
    cum_imbal_tol_pct: float = 2.0,
    sustained_time_tol_pct: float = 1.0,
    matbal_tol_pct: float = 0.15,
) -> List[Dict[str, object]]:
    """
    Parse every SLURM log in a batch folder and flag the problem cases.

    Args:
        log_dir: Folder of ``slurm-<jobid>_<task>.out`` files.
        npy_dir: Optional folder of extracted ``case<N>_<property>.npy`` arrays;
            when given, the time-axis length of each is checked for truncation.
        npy_property: Property suffix used to locate the .npy files.
        target_end_day: Intended simulation end time. Defaults to the most common
            end time in the batch, which is the majority of runs that finished.
        cum_imbal_tol_pct: Flag a case whose cumulative produced-vs-injected
            volumes differ by more than this.
        sustained_time_tol_pct: Flag a case that spends more than this share of
            simulated time above 20% lag-corrected rate imbalance.
        matbal_tol_pct: Flag a case whose water material-balance error exceeds this.

    Returns:
        List of per-case dicts sorted by case number, each with ``target_end_day``,
        ``pct_of_target``, ``npy_time_steps``, ``flags`` (list of str) and
        ``ok`` (bool) added.
    """
    log_dir = Path(log_dir)
    logs = sorted(
        (p for p in log_dir.glob("slurm-*_*.out") if _LOG_NAME.search(p.name)),
        key=lambda p: int(_LOG_NAME.search(p.name).group(2)),
    )
    if not logs:
        raise FileNotFoundError(f"No slurm-<jobid>_<task>.out files in {log_dir}")

    records = [parse_log(p) for p in logs]

    if target_end_day is None:
        ends = [r["end_time_day"] for r in records if r["end_time_day"] is not None]
        target_end_day = max(set(ends), key=ends.count) if ends else None

    expected = None
    if npy_dir is None:
        for record in records:
            record["npy_time_steps"] = None
    else:
        npy_dir = Path(npy_dir)
        for record in records:
            record["npy_time_steps"] = npy_time_steps(
                npy_dir / f"case{record['case']}_{npy_property}.npy"
            )
        # The expected length is the batch mode, i.e. what the runs that finished agree on.
        lengths = [
            r["npy_time_steps"] for r in records if r["npy_time_steps"] is not None
        ]
        expected = max(set(lengths), key=lengths.count) if lengths else None

    for record in records:
        flags: List[str] = []
        record["target_end_day"] = target_end_day
        end = record["end_time_day"]
        record["pct_of_target"] = (
            end / target_end_day * 100.0 if end is not None and target_end_day else None
        )

        if record["status"] == "MISSING":
            flags.append("no-termination-line")
        elif record["status"].lower() != "normal":
            flags.append(f"{record['status'].lower()}-termination")
        if end is None:
            flags.append("no-end-time")
        elif target_end_day and end < target_end_day - 1e-6:
            flags.append("early-termination")

        cum = record["cum_imbal_pct"]
        if cum is not None and abs(cum) > cum_imbal_tol_pct:
            flags.append("cum-rate-imbalance")
        sustained = record.get("frac_time_gt20pct")
        if sustained is not None and sustained > sustained_time_tol_pct:
            flags.append("sustained-rate-imbalance")
        mb = record["matbal_water_pct"]
        if mb is not None and mb > matbal_tol_pct:
            flags.append("high-matbal-error")
        if (
            expected is not None
            and record["npy_time_steps"] is not None
            and record["npy_time_steps"] != expected
        ):
            flags.append(f"npy-truncated({record['npy_time_steps']}/{expected})")

        record["flags"] = flags
        record["ok"] = not flags
    return records


def to_dataframe(records: Sequence[Dict[str, object]]):
    """
    Convert scan results to a pandas DataFrame for notebook use.

    Args:
        records: Output of :func:`scan_batch`.

    Returns:
        A pandas DataFrame indexed by case number.
    """
    import pandas as pd

    frame = pd.DataFrame(list(records))
    frame["flags"] = frame["flags"].apply(lambda f: ",".join(f) if f else "")
    return frame.set_index("case").sort_index()


def _fmt(value: object, spec: str = ".3g", width: int = 0) -> str:
    """Format a possibly-None value for the report table."""
    text = "-" if value is None else format(value, spec)
    return text.rjust(width) if width else text


def report(records: Sequence[Dict[str, object]], show_all: bool = False) -> None:
    """
    Print a human-readable QC report.

    Args:
        records: Output of :func:`scan_batch`.
        show_all: Print every case rather than only the flagged ones.
    """
    target = next((r["target_end_day"] for r in records), None)
    flagged = [r for r in records if not r["ok"]]
    print(f"Cases parsed        : {len(records)}")
    print(f"Target end time     : {_fmt(target, '.6g')} day")
    print(f"Clean               : {len(records) - len(flagged)}")
    print(f"Flagged             : {len(flagged)}")

    shown = records if show_all else flagged
    if not shown:
        print("\nNo issues found.")
        return

    header = (
        f"\n{'CASE':>5} {'STATUS':>8} {'END_DAY':>11} {'%TGT':>6} "
        f"{'CUM_IMB%':>9} {'PEAK_IMB%':>10} {'T>20%':>7} {'IMB_END_D':>10} "
        f"{'MATBAL%':>8} {'CUTS':>5} {'SOLVF':>6}  FLAGS"
    )
    print(header)
    print("-" * (len(header) + 24))
    for r in shown:
        print(
            f"{r['case']:>5} {str(r['status'])[:8]:>8} "
            f"{_fmt(r['end_time_day'], '.6g', 11)} {_fmt(r['pct_of_target'], '.1f', 6)} "
            f"{_fmt(r['cum_imbal_pct'], '.2f', 9)} {_fmt(r['peak_imbal_pct'], '.1f', 10)} "
            f"{_fmt(r.get('frac_time_gt20pct'), '.1f', 7)} "
            f"{_fmt(r.get('imbal_closes_day'), '.4g', 10)} "
            f"{_fmt(r['matbal_water_pct'], '.3f', 8)} "
            f"{_fmt(r['cuts'], 'd', 5)} {_fmt(r['solver_failures'], 'd', 6)}  "
            f"{','.join(r['flags'])}"
        )

    for r in flagged:
        if r["fatal_message"]:
            print(
                f"\ncase{r['case']}: FATAL in {r['fatal_subroutine']} -- "
                f"{r['fatal_message']}"
                f"\n  stopped at {_fmt(r['end_time_day'], '.6g')} day "
                f"({_fmt(r['pct_of_target'], '.1f')}% of target); "
                f"largest single-step temperature change "
                f"{_fmt(r['max_temp_change_degC'], '.1f')} degC"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """
    Command-line entry point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 if no case was flagged, 1 otherwise (usable as a CI check).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log_dir", help="sim_logs/<batch>_log folder")
    parser.add_argument("--npy-dir", default=None, help="folder of extracted .npy arrays")
    parser.add_argument("--npy-property", default="PRES")
    parser.add_argument("--target-end-day", type=float, default=None)
    parser.add_argument("--cum-imbal-tol", type=float, default=2.0)
    parser.add_argument("--sustained-time-tol", type=float, default=1.0)
    parser.add_argument("--matbal-tol", type=float, default=0.15)
    parser.add_argument("--all", action="store_true", help="report every case")
    parser.add_argument("--csv", default=None, help="also write results to this CSV")
    args = parser.parse_args(argv)

    try:
        records = scan_batch(
            args.log_dir,
            npy_dir=args.npy_dir,
            npy_property=args.npy_property,
            target_end_day=args.target_end_day,
            cum_imbal_tol_pct=args.cum_imbal_tol,
            sustained_time_tol_pct=args.sustained_time_tol,
            matbal_tol_pct=args.matbal_tol,
        )
    except FileNotFoundError as err:
        parser.error(str(err))
    report(records, show_all=args.all)

    if args.csv:
        import csv

        fields = [k for k in records[0] if k != "flags"] + ["flags"]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in records:
                writer.writerow({**r, "flags": ",".join(r["flags"])})
        print(f"\nWrote {args.csv}")

    return 0 if all(r["ok"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
