"""Batch-extract thermal energy AND water volumes per well group from CMG STARS
SR3 files.

Replaces batch_net_energy.py + extract_annual_water.py: both quantities come
out of the same SR3 read, on the same time grid, in one CSV per case.

    end_date,inj_energy_PJ,pro_energy_PJ,inj_water_m3,pro_water_m3
    2030-01-01,0.0,0.0,0.0,0.0
    2031-01-01,0.112202,1.924103,5331918.3,5560542.7
    ...
    2060-01-01,2.622613,42.266272,124627843.4,125535311.0

Rows are 1 January of each year by default, linearly interpolated between
simulator timesteps (exact where a timestep lands on the date). Energy is
the cumulative enthalpy of each group in PJ; net extraction is
pro_energy_PJ - inj_energy_PJ. Water is cumulative volume in m3, summed
over each group.

    python batch_sr3.py /path/to/runs
    python batch_sr3.py /path/to/runs -o results --jobs 8
    python batch_sr3.py /path/to/runs --yearly        # per-year increments
    python batch_sr3.py /path/to/runs --all-steps     # every timestep
    python batch_sr3.py /path/to/runs --reservoir     # WATVOLRC, not WATVOLSC

A summary file (one row per case, final values only) is also written unless
you pass --no-summary.

SR3 layout notes (undocumented by CMG, verified against STARS 202x):
  - Special histories live under 'TimeSeries/SPECIAL HISTORY' -- with a space.
    Data is (ntime, nvar, norigin) with one dummy origin ('SPVALS'), so the
    well name sits in the variable label, e.g. 'ENERGY Cum PRO1'.
  - Energy internal unit is J with gain 1.0 (General/UnitsTable). Verify that
    table if you change *INUNIT / *OUTUNIT.
  - Well vectors live under 'TimeSeries/WELLS', shape (ntime, nvar, nwell),
    names in 'Origins'. 'WATVOLSC' is cumulative surface water in m3, stored
    as a POSITIVE magnitude for producers and injectors alike; 'WATVOLRC' is
    the reservoir-condition equivalent. Both match the .out group summaries to
    within the .out's 5-significant-figure printing.
  - 'WELLOPMO' is negative for producers, positive for injectors. Prefer it to
    well names. WellTable['Parent'] is NOT usable: CMG mirrors every group into
    '-INJ' and '-PRO' copies and points injectors at the '-PRO' one, so INJ1
    reads 'INJ-GRP-PRO'.
  - MasterTimeTable 'Date' is encoded YYYYMMDD.fraction_of_day.
  - Producer and injector enthalpies share the same reference (*TEMR), so
    producers - injectors is a valid net-extraction figure.
  - The .out file cannot substitute for any of this: it only prints well
    summary tables at the *OUTPRN* report times (2030/2040/2050/2060 here).
"""

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SPECIAL_GROUP = "TimeSeries/SPECIAL HISTORY"
WELLS_GROUP = "TimeSeries/WELLS"
LABEL_RE = re.compile(r"^ENERGY\s+Cum\s+(\S+)$", re.IGNORECASE)
WATER_VAR = {"surface": "WATVOLSC", "reservoir": "WATVOLRC"}

DATE_FMT = "%Y-%m-%d"                  # annual output: one row per day
STEP_FMT = "%Y-%m-%d %H:%M:%S"         # --all-steps: sub-day steps need the time

SCRIPT_DIR = Path(__file__).resolve().parent

# Control characters that appear when a Windows path is written without an r"" prefix:
# "C:\new" -> "C:" + newline + "ew".  Real paths never contain these, so seeing one
# means the backslash sequence was eaten by Python's string parser.
_ESCAPE_REPAIR = {
    "\a": r"\a", "\b": r"\b", "\f": r"\f", "\n": r"\n",
    "\r": r"\r", "\t": r"\t", "\v": r"\v", "\0": r"\0",
}


# ------------------------------------------------------------------- paths --

def clean_path(value, name="path"):
    """Strip stray quotes/whitespace and undo backslash escapes eaten by Python."""
    s = str(value).strip().strip('"').strip("'")
    if any(ch in s for ch in _ESCAPE_REPAIR):
        s = "".join(_ESCAPE_REPAIR.get(ch, ch) for ch in s)
        print(
            f"! {name} contained escape characters and was repaired to:\n"
            f"    {s}\n"
            f"  Use a raw string -- {name} = r\"C:\\path\\to\\folder\" -- "
            f"or forward slashes.",
            file=sys.stderr,
        )
    return Path(s)


def anchor(p):
    """Resolve a relative path against the script's folder, not the CWD."""
    p = clean_path(p)
    return p if p.is_absolute() else SCRIPT_DIR / p


def safe_write(df, path, **kwargs):
    """Write a CSV, stepping aside if the target is locked or read-only."""
    path = Path(path)
    if path.is_dir() or path.suffix.lower() != ".csv":
        path = path / "output.csv"
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    for cand in (path,
                 path.with_name(f"{path.stem}_{stamp}.csv"),   # open in Excel
                 Path.cwd() / path.name):                      # folder locked
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cand, **kwargs)
            return cand
        except OSError as exc:
            print(f"  could not write {cand}: {type(exc).__name__}",
                  file=sys.stderr)
    raise OSError(f"no writable location for {path.name}")


def case_id(path):
    """Pull a case number out of the filename; fall back to the stem."""
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else Path(path).stem


def _natkey(s):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


# -------------------------------------------------------------------- read --

def decode_date(code):
    """SR3 stores dates as the float YYYYMMDD.fraction_of_day."""
    whole = int(code)
    return datetime(whole // 10000, (whole // 100) % 100, whole % 100) + timedelta(
        days=float(code) - whole
    )


def _dates_for(f, steps):
    return [decode_date(c) for c in f["General/MasterTimeTable"]["Date"][steps]]


def _classify(opmode, names, prod_re=None, inj_re=None):
    """PRO/INJ per well. Explicit patterns win, else the WELLOPMO sign."""
    kinds = {}
    for j, w in enumerate(names):
        if prod_re is not None and prod_re.search(w):
            kinds[w] = "PRO"
            continue
        if inj_re is not None and inj_re.search(w):
            kinds[w] = "INJ"
            continue
        vals = opmode[:, j]
        vals = vals[vals != 0]
        if vals.size:
            kinds[w] = "PRO" if np.median(vals) < 0 else "INJ"
        else:
            raise ValueError(f"well {w!r} never opened and matches no pattern")
    return kinds


def read_sr3(path, basis="surface", prod_re=None, inj_re=None):
    """Read one SR3.

    Returns dict with:
        t_water, water (per-well DataFrame, m3), kinds,
        t_energy, inj_PJ, pro_PJ
    Each series keeps its own time axis; they are aligned later.
    """
    var = WATER_VAR[basis]
    with h5py.File(path, "r") as f:
        if WELLS_GROUP not in f:
            raise KeyError("no TimeSeries/WELLS group -- is this a STARS SR3?")
        g = f[WELLS_GROUP]
        names = [o.decode().strip() for o in g["Origins"][:]]
        variables = [v.decode().strip() for v in g["Variables"][:]]
        if var not in variables:
            raise KeyError(f"{var} not among the SR3 well variables")
        if "WELLOPMO" not in variables:
            raise KeyError("WELLOPMO missing -- cannot classify wells")

        data = g["Data"]
        cum = data[:, variables.index(var), :]
        opmode = data[:, variables.index("WELLOPMO"), :]
        t_water = _dates_for(f, g["Timesteps"][:])

        if SPECIAL_GROUP not in f:
            raise KeyError("no SPECIAL HISTORY group -- was *OUTSRF *SPECIAL set?")
        sg = f[SPECIAL_GROUP]
        labels = [s.decode().strip() for s in sg["Variables"][:]]
        steps = sg["Timesteps"][:]
        if len(steps) < 2:
            raise ValueError(f"only {len(steps)} timestep(s) -- run likely died")
        energy = sg["Data"][:, :, 0]           # (ntime, nvar), dummy origin
        t_energy = _dates_for(f, steps)

    kinds = _classify(opmode, names, prod_re, inj_re)
    if not any(k == "PRO" for k in kinds.values()) or \
       not any(k == "INJ" for k in kinds.values()):
        raise ValueError(f"need both producers and injectors, got {kinds}")

    # net energy = producers - injectors, using the same PRO/INJ call as water
    picks = {"INJ": np.zeros(len(labels)), "PRO": np.zeros(len(labels))}
    for i, label in enumerate(labels):
        m = LABEL_RE.match(label)
        if not m:
            continue
        well = m.group(1)
        if well not in kinds:
            raise ValueError(f"special history names unknown well {well!r}")
        picks[kinds[well]][i] = 1.0
    if not picks["INJ"].any() or not picks["PRO"].any():
        raise ValueError("SPECIAL HISTORY lacks 'ENERGY Cum <well>' vectors "
                         "for both producers and injectors")

    order = sorted(names, key=lambda w: (0 if kinds[w] == "INJ" else 1, _natkey(w)))
    return {
        "t_water": pd.DatetimeIndex(t_water),
        "water": pd.DataFrame(cum, columns=names)[order],
        "kinds": kinds,
        "t_energy": pd.DatetimeIndex(t_energy),
        # enthalpy summed over each group, J -> PJ. Both share the *TEMR
        # reference, so pro - inj is the net extraction.
        "inj_PJ": energy.dot(picks["INJ"]) / 1e15,
        "pro_PJ": energy.dot(picks["PRO"]) / 1e15,
    }


# ------------------------------------------------------------------- align --

def _grid(t_start, t_end, annual):
    """1 January of each year inside the run, or the run's own timesteps."""
    if not annual:
        return None
    years = [datetime(y, 1, 1) for y in range(t_start.year, t_end.year + 1)]
    keep = [d for d in years if t_start <= d <= t_end]
    if not keep:
        raise ValueError(
            f"run spans {t_start:%Y-%m-%d} to {t_end:%Y-%m-%d} -- "
            f"no 1 January falls inside it"
        )
    return pd.DatetimeIndex(keep)


def _interp(grid, t, values, t0):
    x = (grid - t0).total_seconds().to_numpy()
    xp = (t - t0).total_seconds().to_numpy()
    return np.interp(x, xp, values)


def case_table(path, basis="surface", annual=True, yearly=False,
               prod_re=None, inj_re=None):
    """Combined per-case table.

    Index = end_date; columns = inj_energy_PJ, pro_energy_PJ, inj_water_m3,
    pro_water_m3. Per-well water stays available via read_sr3().
    """
    r = read_sr3(path, basis, prod_re, inj_re)
    t_w, t_e = r["t_water"], r["t_energy"]

    t_start, t_end = max(t_w[0], t_e[0]), min(t_w[-1], t_e[-1])
    if annual:
        grid = _grid(t_start, t_end, True)
    else:
        grid = t_w.union(t_e)
        grid = grid[(grid >= t_start) & (grid <= t_end)]

    t0 = t_start
    kinds = r["kinds"]
    inj = [w for w in r["water"].columns if kinds[w] == "INJ"]
    pro = [w for w in r["water"].columns if kinds[w] == "PRO"]

    out = pd.DataFrame(index=grid)
    out["inj_energy_PJ"] = _interp(grid, t_e, r["inj_PJ"], t0)
    out["pro_energy_PJ"] = _interp(grid, t_e, r["pro_PJ"], t0)
    out["inj_water_m3"] = _interp(
        grid, t_w, r["water"][inj].sum(axis=1).to_numpy(), t0)
    out["pro_water_m3"] = _interp(
        grid, t_w, r["water"][pro].sum(axis=1).to_numpy(), t0)

    if yearly:                      # per-period increments, energy and water
        out = out.diff().iloc[1:]

    out.index = out.index.strftime(DATE_FMT if annual else STEP_FMT)
    out.index.name = "end_date"

    decimals = {"inj_energy_PJ": 6, "pro_energy_PJ": 6,   # matches the old CSVs
                "inj_water_m3": 1, "pro_water_m3": 1}     # 0.1 m3 is plenty
    return out.round(decimals), r["kinds"], t_w[-1]


# ------------------------------------------------------------------- batch --

def _worker(args):
    path, outdir, basis, annual, yearly, prod_pat, inj_pat = args
    prod_re = re.compile(prod_pat, re.I) if prod_pat else None
    inj_re = re.compile(inj_pat, re.I) if inj_pat else None
    try:
        df, kinds, raw_end = case_table(path, basis, annual, yearly,
                                        prod_re, inj_re)
        suffix = "energy_water_yearly" if yearly else "energy_water"
        if basis == "reservoir":
            suffix += "_rc"          # don't overwrite the surface-basis CSV
        dest = safe_write(df, Path(outdir) / f"{Path(path).stem}_{suffix}.csv")
        final = df.iloc[-1] if not yearly else df.sum()
        row = {"case": case_id(path), "end_date": raw_end,
               "n_rows": len(df), "csv": dest.name}
        row.update({c: float(final[c]) for c in df.columns})
        return path, row, None
    except Exception as exc:
        return path, None, f"{type(exc).__name__}: {exc}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder", nargs="?", default="runs")
    p.add_argument("-o", "--outdir", default="sr3_csv",
                   help="where the per-case CSVs go (created if missing)")
    p.add_argument("--glob", default="*.sr3")
    p.add_argument("--prod-pattern", default=None,
                   help="regex forcing PRO (default: use the WELLOPMO sign)")
    p.add_argument("--inj-pattern", default=None,
                   help="regex forcing INJ (default: use the WELLOPMO sign)")
    p.add_argument("--reservoir", action="store_true",
                   help="reservoir-condition water (WATVOLRC)")
    p.add_argument("--yearly", action="store_true",
                   help="per-year increments instead of running totals")
    p.add_argument("--all-steps", action="store_true",
                   help="every simulator timestep instead of 1 January only")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--no-summary", action="store_true",
                   help="skip the one-row-per-case summary file")
    args = p.parse_args()

    folder, outdir = anchor(args.folder), anchor(args.outdir)
    files = sorted(folder.glob(args.glob))
    if not files:
        sys.exit(f"no files matching {args.glob} in {folder}")
    outdir.mkdir(parents=True, exist_ok=True)

    basis = "reservoir" if args.reservoir else "surface"
    tasks = [(f, outdir, basis, not args.all_steps, args.yearly,
              args.prod_pattern, args.inj_pattern) for f in files]

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            done = [fut.result() for fut in as_completed(futures)]
    else:
        done = [_worker(t) for t in tasks]

    rows, failures = [], []
    for path, payload, err in done:
        if err:
            failures.append((Path(path).name, err))
        else:
            rows.append(payload)

    if not rows:
        sys.exit("nothing parsed")

    summary = pd.DataFrame(rows).sort_values(
        "case", key=lambda s: s.map(lambda v: (isinstance(v, str), v))
    ).reset_index(drop=True)

    # Flag runs that stopped short of the ensemble's usual end date.
    expected = summary["end_date"].mode().iloc[0]
    short = summary[summary["end_date"] != expected]
    for _, r in short.iterrows():
        print(f"  ! case {r['case']}: ends {r['end_date']:%Y-%m-%d}, "
              f"expected {expected:%Y-%m-%d}", file=sys.stderr)

    if not args.no_summary:
        out = summary.drop(columns=["csv"]).copy()
        out["end_date"] = out["end_date"].dt.strftime(DATE_FMT)
        dest = safe_write(out, outdir / "summary.csv", index=False)
        print(f"summary -> {dest}")

    print(f"{len(rows)} case(s) -> {outdir}  "
          f"({'per-year' if args.yearly else 'cumulative'}, {basis} water)")
    for name, err in failures:
        print(f"  x {name}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
