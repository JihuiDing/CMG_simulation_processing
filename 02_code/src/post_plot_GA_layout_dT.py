"""
Plan-view map of temperature change T(last) - T(0) in one layer for a GA case,
with fault cells, well locations and the peak fault stress-ratio (tau/sigma_n) cells
at the initial and last time steps (per-cell fault orientation).

Usage (from repo root):
    python src/plot_GA_layout_dT.py --cases 13 172 --gen gen_000 --k 15 --fault 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "src"))
from fault_slip_analysis import FSA_stress_based_per_cell

# diverging map close to cmocean 'balance' (navy - white - dark red)
CMAP = LinearSegmentedColormap.from_list(
    "balance_like",
    ["#0b1640", "#1f4e9c", "#5d9cc9", "#bcd7e6", "#f1ece9", "#e5b9a1", "#c4774f", "#8e3a1b", "#3f0b0b"])
FAULT_COLORS = {0: "#7b3fa0", 1: "#12855f"}


def fault_ratio(sim_dir, case, cases_df, coor, fault_id):
    """tau/sigma_n on the fault cells, shape (n_cells, n_times), plus cell (i,j,k) indices."""
    Sv = np.load(sim_dir / f"case{case}_STRESMXP.npy").astype(np.float64)
    Sh = np.load(sim_dir / f"case{case}_STRESMNP.npy").astype(np.float64)
    SH = np.load(sim_dir / f"case{case}_STRESINT.npy").astype(np.float64)
    az = cases_df.loc[cases_df["case"] == case, "SH_azi_deg"].iloc[0]
    m = coor[..., 3] == fault_id
    sigma, tau, _ = FSA_stress_based_per_cell(
        SH[m], Sh[m], Sv[m], az, (coor[..., 5][m] - 90) % 360, coor[..., 4][m], 0, 0.6)
    return tau / sigma, np.argwhere(m)


def plot_case(case, gen, k, fault_id, vlim, tag, out_dir, sim_dir, cases_df, coor):
    row = cases_df.loc[cases_df["case"] == case].iloc[0]
    kk = k - 1
    T = np.load(sim_dir / f"case{case}_TEMP.npy")
    T0, T1 = T[:, :, kk, 0].astype(float), T[:, :, kk, -1].astype(float)
    inactive = (T0 == 0) | (T1 == 0)
    dT = np.where(inactive, np.nan, T1 - T0)
    X, Y = coor[:, :, kk, 0] / 1000, coor[:, :, kk, 1] / 1000

    ratio, ijk = fault_ratio(sim_dir, case, cases_df, coor, fault_id)
    t_last = ratio.shape[1] - 1
    j0, j1 = np.nanargmax(ratio[:, 0]), np.nanargmax(ratio[:, t_last])
    r0, r1 = ratio[j0, 0], ratio[j1, t_last]
    c0, c1 = ijk[j0], ijk[j1]

    fig, ax = plt.subplots(figsize=(8, 10.8))
    norm = Normalize(-vlim, vlim)
    pc = ax.pcolormesh(X, Y, np.ma.masked_invalid(dT), cmap=CMAP, norm=norm, shading="nearest")

    # fault cells in this layer
    for f, col in FAULT_COLORS.items():
        fm = (coor[:, :, kk, 3] == f) & ~inactive
        ax.pcolormesh(X, Y, np.ma.masked_where(~fm, np.ones_like(dT)),
                      cmap=LinearSegmentedColormap.from_list("f", [col, col]), shading="nearest")

    # temperature contours
    dTc = np.ma.masked_invalid(dT)
    neg = [-70, -60, -50, -40, -30, -20, -10, -5]
    cs = ax.contour(X, Y, dTc, levels=neg, colors="white", linewidths=0.8, linestyles="--")
    ax.clabel(cs, fmt="%d", fontsize=7)
    cs2 = ax.contour(X, Y, dTc, levels=[-1], colors="k", linewidths=0.7, linestyles="--")
    ax.clabel(cs2, fmt="%d", fontsize=7)
    cs3 = ax.contour(X, Y, dTc, levels=[1], colors="k", linewidths=0.7, linestyles="-")
    ax.clabel(cs3, fmt="+%d", fontsize=7)

    # wells (CMG 1-based I,J)
    wells = {}
    for w in ["INJ1", "INJ2", "INJ3", "PRO1", "PRO2", "PRO3"]:
        I, J = int(row[f"{w}_I"]), int(row[f"{w}_J"])
        wells[w] = (I, J)
        xw, yw = X[I - 1, J - 1], Y[I - 1, J - 1]
        mk = "v" if w.startswith("INJ") else "^"
        ax.plot(xw, yw, mk, ms=13, mfc="white", mec="k", mew=1.8, zorder=6)
        name = f"{w[:3]}-{w[3]}"
        ax.annotate(name, (xw, yw), xytext=(-8, 9), textcoords="offset points", ha="right",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=7)

    # peak stress-ratio cells (may lie in another layer: plotted at their i,j)
    def star(cell, r, label, filled):
        i, j, kc = cell
        xs, ys = coor[i, j, kc, 0] / 1000, coor[i, j, kc, 1] / 1000
        ax.plot(xs, ys, "*", ms=20, mfc="gold" if filled else "white", mec="k", mew=1.2, zorder=8)
        ax.annotate(f"{label}: τ/σn = {r:.3f}\nCMG ({i+1},{j+1},{kc+1})", (xs, ys),
                    xytext=(12, -4), textcoords="offset points", fontsize=8.5, va="top",
                    bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="0.6", alpha=0.9), zorder=9)

    star(c0, r0, "max t0", False)
    star(c1, r1, "max last", True)

    ax.text(0.03, 0.985,
            f"ΔT range {np.nanmin(dT):+.1f} to {np.nanmax(dT):+.1f} °C\n(colour scale shared ±{vlim:.1f})",
            transform=ax.transAxes, va="top", fontsize=9, color="0.25")
    ax.text(0.97, 0.985,
            f"max fault{fault_id} τ/σn (all layers)\n time 0: {r0:.3f}\n last:   {r1:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5"))

    inj = " ".join(f"({wells[w][0]},{wells[w][1]})" for w in ["INJ1", "INJ2", "INJ3"])
    pro = " ".join(f"({wells[w][0]},{wells[w][1]})" for w in ["PRO1", "PRO2", "PRO3"])
    handles = [
        Patch(color=FAULT_COLORS[0], label="fault0  (strike 018.1°)"),
        Patch(color=FAULT_COLORS[1], label="fault1  (strike 350.5°)"),
        Line2D([], [], marker="v", ls="", ms=10, mfc="white", mec="k", mew=1.5, label=f"Injectors  CMG {inj}"),
        Line2D([], [], marker="^", ls="", ms=10, mfc="white", mec="k", mew=1.5, label=f"Producers  CMG {pro}"),
        Line2D([], [], marker="*", ls="", ms=13, mfc="white", mec="k", label=f"peak fault{fault_id} τ/σn, time 0"),
        Line2D([], [], marker="*", ls="", ms=13, mfc="gold", mec="k", label=f"peak fault{fault_id} τ/σn, last time"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.95)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.set_title(f"GA {gen}, case{case} (layout {int(row['layout_id'])}, param set {int(row['param_id'])}) — "
                 f"{tag}\ntemperature change T(last) − T(0), layer k = {k}", fontweight="bold", fontsize=12)
    cb = fig.colorbar(pc, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label("ΔT = T(last) − T(0)   (°C)")
    fig.tight_layout()
    out = out_dir / f"dT_{gen}_case{case}_k{k}_{tag.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="gen_000")
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--k", type=int, default=15, help="CMG layer (1-based)")
    ap.add_argument("--fault", type=int, default=1)
    a = ap.parse_args()

    base = repo_root / "results" / "Vienna_geothermal"
    ga = base / "GA"
    sim_dir = ga / f"sim_{a.gen}_sim_py"
    cases_df = pd.read_csv(sorted(ga.glob(f"{a.gen}_cases_*.csv"))[0], encoding="utf-8-sig")
    coor = np.load(base / "coor_fault" / "JD_geothermal_coor&fault&dip.npy")
    out_dir = ga / "figures"
    out_dir.mkdir(exist_ok=True)

    # shared symmetric colour limit across the cases
    vlim = 0
    for c in a.cases:
        T = np.load(sim_dir / f"case{c}_TEMP.npy")[:, :, a.k - 1].astype(float)
        ok = (T[..., 0] != 0) & (T[..., -1] != 0)
        vlim = max(vlim, np.abs(T[..., -1] - T[..., 0])[ok].max())

    tags = a.tags or [f"case{c}" for c in a.cases]
    for c, t in zip(a.cases, tags):
        print("saved", plot_case(c, a.gen, a.k, a.fault, vlim, t, out_dir, sim_dir, cases_df, coor))


if __name__ == "__main__":
    main()
