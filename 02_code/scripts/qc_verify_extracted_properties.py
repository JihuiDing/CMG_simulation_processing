"""Independently verify extracted property .dat files against the raw FINIT / F0000 files.

For every realization with property files, rebuild each full property array from the raw
files with code independent of the production pipeline (own parser, own j-reversal, own
N*value decompression) and compare it with the saved .dat file.

Usage:  python qc_verify_extracted_properties.py <realizations_dir> <properties_dir> <out_csv>
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

NI, NJ, NK = 139, 248, 23          # JD_geothermal grid
N_CELLS = NI * NJ * NK
HDR = re.compile(rb"^\s*'([^']{1,8})'\s+(\d+)\s+'(\w{4})'\s*$", re.M)
SOURCES = {"PORO": "FINIT", "PERMX": "FINIT", "TEMP": "F0000"}


def read_arrays(path, keys):
    """Return {key: np.ndarray} for the requested sections of an Eclipse formatted file."""
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"{path.name} contains NUL bytes")
    headers = list(HDR.finditer(data))
    out = {}
    for h, nxt in zip(headers, headers[1:] + [None]):
        name = h.group(1).strip().decode()
        if name in keys:
            body = data[h.end(): nxt.start() if nxt else len(data)]
            arr = np.array(body.split(), dtype=np.float64)
            if arr.size != int(h.group(2)):
                raise ValueError(f"{path.name} {name}: {arr.size} values, header declares {h.group(2)}")
            out[name] = arr
    missing = set(keys) - set(out)
    if missing:
        raise ValueError(f"{path.name}: missing {sorted(missing)}")
    return out


def decompress(path):
    """Expand a CMG 'N*value' file into a flat array."""
    vals = []
    for tok in path.read_text().split():
        if "*" in tok:
            n, v = tok.split("*")
            vals.extend([float(v)] * int(n))
        else:
            vals.append(float(tok))
    return np.array(vals)


def expected_full(active_values, actid):
    full = np.zeros(N_CELLS)
    full[actid.astype(np.int64) - 1] = active_values
    # Petrel order (i fastest, then j, then k) with j reversed for CMG
    return full.reshape(NK, NJ, NI)[:, ::-1, :].ravel()


def main(real_dir, prop_dir, out_csv):
    real_dir, prop_dir = Path(real_dir), Path(prop_dir)
    cases = sorted({p.name.rsplit("_", 1)[0] for p in prop_dir.glob("*.dat")}, key=lambda c: int(c.split("_")[-1]))
    rows = []
    for case in cases:
        folder = real_dir / case
        try:
            finit = read_arrays(folder / f"{case}.FINIT", ["PORO", "PERMX", "ACTID"])
            f0000 = read_arrays(folder / f"{case}.F0000", ["TEMP"])
        except Exception as e:  # report and continue
            rows.append({"case": case, "prop": "ALL", "status": f"RAW ERROR: {e}"})
            print(case, "RAW ERROR", e, flush=True)
            continue
        actid = finit["ACTID"]
        arrays = {"PORO": finit["PORO"], "PERMX": finit["PERMX"], "TEMP": f0000["TEMP"]}
        for prop, active in arrays.items():
            dat = prop_dir / f"{case}_{prop}.dat"
            got = decompress(dat)
            exp = expected_full(active, actid)
            row = {"case": case, "prop": prop, "n_values": got.size, "n_active": actid.size,
                   "n_nonzero_dat": int(np.count_nonzero(got)),
                   "min": got.min(), "max": got.max(), "mean_active": active.mean()}
            if got.size != exp.size:
                row["status"] = f"SIZE {got.size} != {exp.size}"
            else:
                diff = np.abs(got - exp)
                row["max_abs_diff"] = diff.max()
                row["n_exact_mismatch"] = int(np.count_nonzero(diff))
                row["status"] = "EXACT" if diff.max() == 0 else ("OK_TOL" if np.allclose(got, exp, rtol=0, atol=1e-8) else "MISMATCH")
            rows.append(row)
            print(case, prop, row["status"], row.get("max_abs_diff"), flush=True)
    fields = ["case", "prop", "status", "n_values", "n_active", "n_nonzero_dat", "max_abs_diff",
              "n_exact_mismatch", "min", "max", "mean_active"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:4])
