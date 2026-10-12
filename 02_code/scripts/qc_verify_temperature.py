"""Verify TEMP property files generated from the sampled temperature gradient.

For each <case>_TEMP.dat in the properties folder, check that every cell equals
Temp_grad * Z + temp_surface (within the rounding of the written values), with Temp_grad
read from the Petrel sampled-parameter csv and Z from the coordinate file. Also fit TEMP
against Z to recover the gradient and surface temperature independently.

Usage:  python qc_verify_temperature.py <properties_dir> <sampled_params_csv> <coor_npy> <out_csv> [temp_surface] [decimals]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def decompress(path):
    vals = []
    for tok in path.read_text().split():
        if "*" in tok:
            n, v = tok.split("*")
            vals.extend([float(v)] * int(n))
        else:
            vals.append(float(tok))
    return np.array(vals)


def main(prop_dir, params_csv, coor_npy, out_csv, temp_surface=11.0, decimals=4):
    prop_dir = Path(prop_dir)
    df = pd.read_csv(params_csv, skiprows=4)
    df = df[pd.to_numeric(df["Temp_grad"], errors="coerce").notna()]
    grads = {n.upper(): float(g) for n, g in zip(df["Parameter"], df["Temp_grad"])}
    z = np.load(coor_npy)[..., 2].transpose(2, 1, 0).ravel()     # CMG order of the property files
    tol = 0.5 * 10 ** -decimals + 1e-9

    rows = []
    for f in sorted(prop_dir.glob("*_TEMP.dat"), key=lambda p: int(p.name.split("_")[2])):
        case = f.name[: -len("_TEMP.dat")]
        raw = f.read_bytes()
        row = {"case": case, "nul_bytes": raw.count(b"\x00")}
        t = decompress(f)
        g = grads.get(case)
        row.update(n_values=t.size, n_zero=int(np.sum(t == 0)), temp_min=t.min(), temp_max=t.max(), temp_grad_csv=g)
        if t.size != z.size or g is None:
            row["status"] = "SIZE OR GRADIENT MISSING"
        else:
            diff = np.abs(t - (g * z + temp_surface))
            slope, icpt = np.polyfit(z, t, 1)
            row.update(max_abs_diff=diff.max(), fit_grad=slope, fit_surface=icpt)
            ok = row["nul_bytes"] == 0 and row["n_zero"] == 0 and diff.max() <= tol
            row["status"] = "OK" if ok else "FAIL"
        rows.append(row)
        print(case, row["status"], f"max diff {row.get('max_abs_diff', float('nan')):.1e}", flush=True)

    fields = ["case", "status", "n_values", "n_zero", "nul_bytes", "temp_grad_csv", "fit_grad", "fit_surface",
              "max_abs_diff", "temp_min", "temp_max"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args[:4], *(float(a) for a in args[4:5]), *(int(a) for a in args[5:6]))
