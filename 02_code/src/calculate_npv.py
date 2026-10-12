"""
CAPEX / OPEX evaluation for a multi-well geothermal scheme on a corner-point grid.

Well geometry (Figure 3)
------------------------
Every well is drilled from ONE shared surface facility and consists of three segments:

  (1) vertical lead      : VERTICAL_LEAD metres straight down from the facility
  (2) deviated section   : straight line from the kick-off point to the well's
                           (i, j) location at the TOP of the reservoir
  (3) reservoir section  : vertical, from reservoir top down to the deepest
                           perforation (base of the deepest reservoir layer)

Index conventions
-----------------
i, j, k supplied by the user are 1-BASED and inclusive (CMG convention).
Reservoir layers 5..20 (1-based) -> python k slices 4..19.

Grid file
---------
`JD_geothermal_coor_fault.npy`, shape (ni, nj, nk, 4):
    [..., 0] = x  (UTM easting, m)
    [..., 1] = y  (UTM northing, m)
    [..., 2] = z  (TVD, m, positive downwards, CELL CENTRE)
    [..., 3] = fault id (NaN where the cell is not faulted)

Because only cell CENTRES are stored, layer faces are reconstructed as the
midpoint between adjacent centres (see `_layer_top` / `_layer_bottom`).

Economic parameters are the defaults in `EconParams` (from the parameter table).
"""

from __future__ import annotations

__version__ = "1.7"  # 1.5 case-table; 1.6 keep_all_cases; 1.7 npv_end_date

import glob
import os
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

try:  # pandas only needed for the tabular helper
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class EconParams:
    """Techno-economic assumptions (values from the project parameter table)."""

    # Discounting
    planning_horizon_yr: int = 30
    discount_rate: float = 0.0686          # real, fraction/yr

    # Revenue (not used by the cost functions, kept for completeness)
    energy_price_usd_per_kwh: float = 0.04

    # CAPEX
    drilling_usd_per_m: float = 2_500.0
    surface_facility_usd: float = 125.0e6  # fixed
    flowline_usd_per_m: float = 5_000.0

    # OPEX
    fixed_cost_per_well_usd_per_yr: float = 200_000.0
    water_handling_usd_per_m3: float = 0.22
    facility_om_fraction_per_yr: float = 0.06   # of facility CAPEX

    # Geometry
    vertical_lead_m: float = 1_000.0       # segment (1), Figure 3

    # Modelling switches (see README notes in the notebook)
    om_includes_flowline: bool = False     # facility CAPEX base for O&M
    perforation_base: str = "layer_bottom"  # or "cell_centre"

    def annuity_factor(self) -> float:
        """PV of 1 $/yr paid at the end of years 1..T at the real discount rate."""
        g, T = self.discount_rate, self.planning_horizon_yr
        if g == 0:
            return float(T)
        return (1.0 - (1.0 + g) ** (-T)) / g


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
def load_grid(path: str) -> np.ndarray:
    """Load the (ni, nj, nk, 4) coordinate/fault array."""
    grid = np.load(path)
    if grid.ndim != 4 or grid.shape[-1] != 4:
        raise ValueError(f"expected (ni, nj, nk, 4) array, got {grid.shape}")
    return grid


def _ij(grid: np.ndarray, i: int, j: int, one_based: bool = True) -> tuple[int, int]:
    """Validate and convert (i, j) to 0-based python indices."""
    ii, jj = (i - 1, j - 1) if one_based else (i, j)
    ni, nj = grid.shape[0], grid.shape[1]
    if not (0 <= ii < ni and 0 <= jj < nj):
        raise IndexError(
            f"well (i={i}, j={j}) outside grid "
            f"(valid {'1..' if one_based else '0..'}{ni if one_based else ni - 1}, "
            f"{'1..' if one_based else '0..'}{nj if one_based else nj - 1})"
        )
    return ii, jj


def _layer_top(grid: np.ndarray, ii: int, jj: int, kk: int) -> float:
    """TVD of the top face of 0-based layer kk, from adjacent cell centres."""
    z = grid[ii, jj, :, 2]
    if kk == 0:
        return float(z[0] - 0.5 * (z[1] - z[0]))
    return float(0.5 * (z[kk - 1] + z[kk]))


def _layer_bottom(grid: np.ndarray, ii: int, jj: int, kk: int) -> float:
    """TVD of the base face of 0-based layer kk, from adjacent cell centres."""
    z = grid[ii, jj, :, 2]
    nk = z.size
    if kk == nk - 1:
        return float(z[-1] + 0.5 * (z[-1] - z[-2]))
    return float(0.5 * (z[kk] + z[kk + 1]))


def reservoir_k_slice(k_top: int = 5, k_bot: int = 20, one_based: bool = True):
    """Return 0-based (kk_top, kk_bot) inclusive for the reservoir interval."""
    return (k_top - 1, k_bot - 1) if one_based else (k_top, k_bot)


# --------------------------------------------------------------------------- #
# Well geometry
# --------------------------------------------------------------------------- #
def well_geometry(
    grid: np.ndarray,
    i: int,
    j: int,
    params: EconParams | None = None,
    facility_xy: Sequence[float] | None = None,
    k_top: int = 5,
    k_bot: int = 20,
    one_based: bool = True,
    facility_ij: Sequence[int] | None = None,
) -> dict:
    """
    Three-segment length breakdown for one well.

    The shared surface pad is given either as `facility_xy` (map coordinates,
    metres) or `facility_ij` (grid index). If neither is given it defaults to
    this well's own (i, j), i.e. a vertical-then-vertical path.
    """
    p = params or EconParams()
    ii, jj = _ij(grid, i, j, one_based)
    kk_top, kk_bot = reservoir_k_slice(k_top, k_bot, one_based)
    nk = grid.shape[2]
    if not (0 <= kk_top <= kk_bot < nk):
        raise IndexError(f"reservoir layers {k_top}..{k_bot} outside k range (nk={nk})")

    xf, yf = _resolve_facility(
        grid, facility_xy, facility_ij, [(i, j)], k_top, one_based
    )

    # Target: (x, y) of the well cell at reservoir top, TVD of the reservoir top face
    x_t = float(grid[ii, jj, kk_top, 0])
    y_t = float(grid[ii, jj, kk_top, 1])
    z_top = _layer_top(grid, ii, jj, kk_top)

    # Deepest perforation
    if p.perforation_base == "layer_bottom":
        z_perf = _layer_bottom(grid, ii, jj, kk_bot)
    elif p.perforation_base == "cell_centre":
        z_perf = float(grid[ii, jj, kk_bot, 2])
    else:
        raise ValueError("perforation_base must be 'layer_bottom' or 'cell_centre'")

    # Segment (1) vertical lead
    L1 = float(p.vertical_lead_m)
    if L1 >= z_top:
        raise ValueError(
            f"well (i={i}, j={j}): vertical lead {L1:.0f} m reaches or passes the "
            f"reservoir top at {z_top:.0f} m TVD - shorten vertical_lead_m"
        )

    # Segment (2) deviated, straight line kick-off -> reservoir-top target
    h_offset = float(np.hypot(x_t - xf, y_t - yf))   # map-view offset, also the flowline length
    L2 = float(np.hypot(h_offset, z_top - L1))

    # Segment (3) reservoir section, vertical to the deepest perforation
    L3 = float(z_perf - z_top)

    faults = grid[ii, jj, kk_top : kk_bot + 1, 3]
    fault_ids = np.unique(faults[~np.isnan(faults)])

    return {
        "i": i,
        "j": j,
        "x_target_m": x_t,
        "y_target_m": y_t,
        "reservoir_top_tvd_m": z_top,
        "perforation_tvd_m": z_perf,
        "horizontal_offset_m": h_offset,
        "L1_vertical_lead_m": L1,
        "L2_deviated_m": L2,
        "L3_reservoir_m": L3,
        "total_length_m": L1 + L2 + L3,
        "inclination_deg": float(np.degrees(np.arctan2(h_offset, z_top - L1))),
        "fault_ids_penetrated": fault_ids.tolist(),
    }


def active_mask(grid: np.ndarray, cv_threshold: float = 0.8) -> np.ndarray:
    """
    (ni, nj) boolean mask of cells that belong to the real model.

    Petrel-style exports pad the i-j array out to a rectangle with cells that
    sit outside the modelled area. Those padding columns carry a simple
    layer-cake extrapolation, so their layer thicknesses are near-uniform,
    while real columns have variable thickness. The coefficient of variation
    of layer thickness down each column separates the two cleanly (bimodal,
    with a wide empty valley around 0.5-1.0 in this grid).
    """
    th = np.diff(grid[..., 2], axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = th.std(axis=2) / th.mean(axis=2)
    return cv > cv_threshold


def _warn_if_padding(grid: np.ndarray, wells, one_based: bool, cv_threshold: float = 0.8):
    """Warn about wells that land in the padding region outside the model."""
    mask = active_mask(grid, cv_threshold)
    bad = []
    for (i, j) in wells:
        ii, jj = _ij(grid, i, j, one_based)
        if not mask[ii, jj]:
            bad.append((i, j))
    if bad:
        warnings.warn(
            f"well(s) {bad} sit in the padding region outside the modelled area; "
            "their depths are extrapolated and the resulting lengths are not "
            "meaningful (pass check_active=False to silence)",
            stacklevel=3,
        )


def facility_location(
    grid: np.ndarray,
    wells: Iterable[tuple[int, int]],
    k_top: int = 5,
    one_based: bool = True,
) -> tuple[float, float]:
    """Default shared-pad position: centroid of the well targets at reservoir top."""
    kk_top, _ = reservoir_k_slice(k_top, k_top, one_based)
    pts = []
    for (i, j) in wells:
        ii, jj = _ij(grid, i, j, one_based)
        pts.append((grid[ii, jj, kk_top, 0], grid[ii, jj, kk_top, 1]))
    pts = np.asarray(pts, dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def ij_to_xy(
    grid: np.ndarray,
    i: int,
    j: int,
    k: int = 5,
    one_based: bool = True,
) -> tuple[float, float]:
    """Map-view (x, y) of the cell centre at (i, j, k). Use this to place the
    surface facility by grid index rather than by UTM coordinate."""
    ii, jj = _ij(grid, i, j, one_based)
    kk, _ = reservoir_k_slice(k, k, one_based)
    return float(grid[ii, jj, kk, 0]), float(grid[ii, jj, kk, 1])


def _resolve_facility(
    grid: np.ndarray,
    facility_xy,
    facility_ij,
    wells,
    k_top: int,
    one_based: bool,
    margin_m: float = 20_000.0,
) -> tuple[float, float]:
    """Resolve the shared-pad position from ij, xy, or the well centroid,
    and reject coordinates that are nowhere near the grid."""
    if facility_xy is not None and facility_ij is not None:
        raise ValueError("give facility_xy or facility_ij, not both")

    if facility_ij is not None:
        return ij_to_xy(grid, facility_ij[0], facility_ij[1], k_top, one_based)

    if facility_xy is None:
        return facility_location(grid, wells, k_top, one_based)

    xf, yf = float(facility_xy[0]), float(facility_xy[1])
    x_lo, x_hi = grid[..., 0].min(), grid[..., 0].max()
    y_lo, y_hi = grid[..., 1].min(), grid[..., 1].max()
    if not (x_lo - margin_m <= xf <= x_hi + margin_m
            and y_lo - margin_m <= yf <= y_hi + margin_m):
        raise ValueError(
            f"facility_xy=({xf:,.1f}, {yf:,.1f}) lies more than {margin_m:,.0f} m "
            f"outside the grid (x {x_lo:,.0f}-{x_hi:,.0f}, y {y_lo:,.0f}-{y_hi:,.0f}). "
            "facility_xy expects map coordinates in metres - if you meant grid "
            "indices, pass facility_ij=(i, j) instead."
        )
    return xf, yf


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
@dataclass
class CostResult:
    wells: list = field(default_factory=list)
    facility_xy: tuple = (np.nan, np.nan)
    total_drilled_length_m: float = 0.0
    total_flowline_length_m: float = 0.0
    capex_drilling_usd: float = 0.0
    capex_surface_usd: float = 0.0
    capex_flowline_usd: float = 0.0
    capex_total_usd: float = 0.0
    opex_fixed_wells_usd_per_yr: float = 0.0
    opex_water_usd_per_yr: float = 0.0
    opex_facility_om_usd_per_yr: float = 0.0
    opex_total_usd_per_yr: float = 0.0
    pv_opex_usd: float = 0.0
    lifecycle_cost_usd: float = 0.0

    def summary(self) -> str:
        m = 1e6
        return (
            f"Wells                      : {len(self.wells)}\n"
            f"Facility (x, y)            : "
            f"{self.facility_xy[0]:,.1f}, {self.facility_xy[1]:,.1f}\n"
            f"Total drilled length       : {self.total_drilled_length_m:,.0f} m\n"
            f"Total flowline length      : {self.total_flowline_length_m:,.0f} m\n"
            f"--- CAPEX ---------------------------------------\n"
            f"  Drilling                 : {self.capex_drilling_usd / m:>10,.2f} MM$\n"
            f"  Surface facility (fixed) : {self.capex_surface_usd / m:>10,.2f} MM$\n"
            f"  Inter-facility flowline  : {self.capex_flowline_usd / m:>10,.2f} MM$\n"
            f"  TOTAL                    : {self.capex_total_usd / m:>10,.2f} MM$\n"
            f"--- OPEX (per year) -----------------------------\n"
            f"  Fixed cost per well      : "
            f"{self.opex_fixed_wells_usd_per_yr / m:>10,.2f} MM$/yr\n"
            f"  Water handling           : "
            f"{self.opex_water_usd_per_yr / m:>10,.2f} MM$/yr\n"
            f"  Facility O&M             : "
            f"{self.opex_facility_om_usd_per_yr / m:>10,.2f} MM$/yr\n"
            f"  TOTAL                    : "
            f"{self.opex_total_usd_per_yr / m:>10,.2f} MM$/yr\n"
            f"--- Discounted ----------------------------------\n"
            f"  PV(OPEX)                 : {self.pv_opex_usd / m:>10,.2f} MM$\n"
            f"  CAPEX + PV(OPEX)         : {self.lifecycle_cost_usd / m:>10,.2f} MM$"
        )

    def to_frame(self):
        if pd is None:
            raise ImportError("pandas is required for to_frame()")
        return pd.DataFrame(self.wells)


def evaluate_scheme(
    grid: np.ndarray,
    injectors: Sequence[tuple[int, int]],
    producers: Sequence[tuple[int, int]],
    params: EconParams | None = None,
    facility_xy: Sequence[float] | None = None,
    facility_ij: Sequence[int] | None = None,
    water_volume_m3_per_yr: float = 0.0,
    k_top: int = 5,
    k_bot: int = 20,
    one_based: bool = True,
    check_active: bool = True,
) -> CostResult:
    """
    CAPEX and annual OPEX for a scheme of injectors + producers.

    Parameters
    ----------
    injectors, producers : sequences of (i, j) well locations (1-based by default)
    facility_xy          : shared surface pad in MAP COORDINATES (metres)
    facility_ij          : shared surface pad by GRID INDEX (i, j) - mutually
                           exclusive with facility_xy. If neither is given, the
                           pad defaults to the centroid of all well targets at
                           reservoir top.
    water_volume_m3_per_yr : total water volume handled per year. This is NOT
                           derivable from the grid - pass the value from the
                           simulation (e.g. cumulative injected or produced
                           volume / horizon). Left at 0 the water term is zero.
    """
    p = params or EconParams()
    all_wells = [(t, ij) for t, group in (("injector", injectors), ("producer", producers))
                 for ij in group]
    if check_active:
        _warn_if_padding(grid, [ij for _, ij in all_wells], one_based)

    facility_xy = _resolve_facility(
        grid, facility_xy, facility_ij,
        [ij for _, ij in all_wells], k_top, one_based,
    )

    rows = []
    for wtype, (i, j) in all_wells:
        g = well_geometry(grid, i, j, p, facility_xy, k_top, k_bot, one_based)
        g["type"] = wtype
        g["drilling_cost_usd"] = g["total_length_m"] * p.drilling_usd_per_m
        # Flowline: facility -> this well, map-view distance (per the agreed convention)
        g["flowline_length_m"] = g["horizontal_offset_m"]
        g["flowline_cost_usd"] = g["flowline_length_m"] * p.flowline_usd_per_m
        rows.append(g)

    total_len = sum(r["total_length_m"] for r in rows)
    total_flow = sum(r["flowline_length_m"] for r in rows)

    capex_drill = total_len * p.drilling_usd_per_m
    capex_surface = p.surface_facility_usd
    capex_flow = total_flow * p.flowline_usd_per_m
    capex_total = capex_drill + capex_surface + capex_flow

    om_base = capex_surface + (capex_flow if p.om_includes_flowline else 0.0)
    opex_wells = p.fixed_cost_per_well_usd_per_yr * len(rows)
    opex_water = p.water_handling_usd_per_m3 * float(water_volume_m3_per_yr)
    opex_om = p.facility_om_fraction_per_yr * om_base
    opex_total = opex_wells + opex_water + opex_om

    pv_opex = opex_total * p.annuity_factor()

    return CostResult(
        wells=rows,
        facility_xy=(float(facility_xy[0]), float(facility_xy[1])),
        total_drilled_length_m=total_len,
        total_flowline_length_m=total_flow,
        capex_drilling_usd=capex_drill,
        capex_surface_usd=capex_surface,
        capex_flowline_usd=capex_flow,
        capex_total_usd=capex_total,
        opex_fixed_wells_usd_per_yr=opex_wells,
        opex_water_usd_per_yr=opex_water,
        opex_facility_om_usd_per_yr=opex_om,
        opex_total_usd_per_yr=opex_total,
        pv_opex_usd=pv_opex,
        lifecycle_cost_usd=capex_total + pv_opex,
    )


# --------------------------------------------------------------------------- #
# Production profiles and NPV
# --------------------------------------------------------------------------- #
PJ_TO_KWH = 1.0e15 / 3.6e6      # 1 PJ = 2.777...e8 kWh


def load_profile(
    path: str,
    energy_basis: str = "net",
    water_basis: str = "produced",
    date_col: str = "end_date",
) -> "pd.DataFrame":
    """
    Turn a cumulative energy/water CSV into annual increments.

    Expects cumulative columns `inj_energy_PJ`, `pro_energy_PJ`, `inj_water_m3`,
    `pro_water_m3` against a date column. The first row is the t = 0 baseline.

    energy_basis : "net" (produced - injected, i.e. extracted heat) or "produced"
    water_basis  : "produced", "injected", "both" (sum) or "max"

    Returns a frame indexed by period number with columns `end_date`, `t_yr`
    (years since the start, used for discounting), `energy_kwh` and `water_m3`.
    """
    if pd is None:
        raise ImportError("pandas is required for load_profile()")

    df = pd.read_csv(path, parse_dates=[date_col]).sort_values(date_col)
    d = df.diff().iloc[1:]

    if energy_basis == "net":
        energy_pj = d["pro_energy_PJ"] - d["inj_energy_PJ"]
    elif energy_basis == "produced":
        energy_pj = d["pro_energy_PJ"]
    else:
        raise ValueError("energy_basis must be 'net' or 'produced'")

    if water_basis == "produced":
        water = d["pro_water_m3"]
    elif water_basis == "injected":
        water = d["inj_water_m3"]
    elif water_basis == "both":
        water = d["pro_water_m3"] + d["inj_water_m3"]
    elif water_basis == "max":
        water = d[["pro_water_m3", "inj_water_m3"]].max(axis=1)
    else:
        raise ValueError("water_basis must be 'produced', 'injected', 'both' or 'max'")

    t_yr = (df[date_col] - df[date_col].iloc[0]).dt.days.iloc[1:] / 365.25

    out = pd.DataFrame({
        "end_date": df[date_col].iloc[1:].to_numpy(),
        "t_yr": t_yr.to_numpy(),
        "energy_pj": energy_pj.to_numpy(),
        "energy_kwh": energy_pj.to_numpy() * PJ_TO_KWH,
        "water_m3": water.to_numpy(),
    })
    out.index = np.arange(1, len(out) + 1)
    out.index.name = "period"
    return out


def evaluate_case(
    grid: np.ndarray,
    injectors: Sequence[tuple[int, int]],
    producers: Sequence[tuple[int, int]],
    profile: "pd.DataFrame",
    params: EconParams | None = None,
    facility_xy: Sequence[float] | None = None,
    facility_ij: Sequence[int] | None = None,
    k_top: int = 5,
    k_bot: int = 20,
    one_based: bool = True,
    name: str = "case",
    discount_on: str = "period",
    npv_end_date: str | None = None,
) -> dict:
    """
    NPV of one case, following

        NPV = -CAPEX_wells - CAPEX_facilities
              + sum_t (1 + gamma)^-t * (REVENUE(t) - OPEX(t))

    CAPEX is taken at t = 0 and undiscounted. CAPEX_wells is the drilling cost
    (proportional to total well length); CAPEX_facilities is the fixed surface
    cost plus the flowline. Periods beyond `planning_horizon_yr` are dropped.

    discount_on : "period" uses the integer period number t = 1, 2, ... exactly
    as written in the equation (assumes one period = one year); "elapsed" uses
    the actual elapsed years from the dates, which differ by a day or two.

    npv_end_date : cut the profile at this date (anything parseable by pandas,
    e.g. "2050-01-01"). A case whose profile stops short of it raises
    ValueError, so in a batch it is reported as having no NPV rather than
    being evaluated over a shorter period than its peers.
    """
    if pd is None:
        raise ImportError("pandas is required for evaluate_case()")
    p = params or EconParams()

    # geometry and CAPEX come from the placement (water term set later)
    geo = evaluate_scheme(
        grid, injectors, producers, params=p,
        facility_xy=facility_xy, facility_ij=facility_ij,
        water_volume_m3_per_yr=0.0, k_top=k_top, k_bot=k_bot, one_based=one_based,
    )

    prof = profile[profile["t_yr"] <= p.planning_horizon_yr + 1e-9].copy()
    if npv_end_date is None:
        if len(prof) < len(profile):
            warnings.warn(
                f"{name}: dropped {len(profile) - len(prof)} period(s) beyond the "
                f"{p.planning_horizon_yr}-year horizon", stacklevel=2,
            )
    else:
        if "end_date" not in profile.columns:
            raise KeyError("profile has no 'end_date' column; reload it with "
                           "load_profile() to use npv_end_date")
        cutoff = pd.Timestamp(npv_end_date)
        last = pd.Timestamp(profile["end_date"].max())
        if last < cutoff:
            raise ValueError(
                f"profile ends {last.date()}, short of npv_end_date "
                f"{cutoff.date()}"
            )
        prof = prof[pd.to_datetime(prof["end_date"]) <= cutoff].copy()
        if prof.empty:
            raise ValueError(
                f"no period ends on or before npv_end_date {cutoff.date()}")

    capex_wells = geo.capex_drilling_usd
    capex_fac = geo.capex_surface_usd + geo.capex_flowline_usd
    capex_total = capex_wells + capex_fac

    om_base = geo.capex_surface_usd + (
        geo.capex_flowline_usd if p.om_includes_flowline else 0.0
    )

    # per-period cash flows
    prof["revenue_usd"] = prof["energy_kwh"] * p.energy_price_usd_per_kwh
    prof["opex_wells_usd"] = p.fixed_cost_per_well_usd_per_yr * len(geo.wells)
    prof["opex_water_usd"] = prof["water_m3"] * p.water_handling_usd_per_m3
    prof["opex_om_usd"] = p.facility_om_fraction_per_yr * om_base
    prof["opex_usd"] = (prof["opex_wells_usd"] + prof["opex_water_usd"]
                        + prof["opex_om_usd"])
    if discount_on == "period":
        t = prof.index.to_numpy(dtype=float)
    elif discount_on == "elapsed":
        t = prof["t_yr"].to_numpy()
    else:
        raise ValueError("discount_on must be 'period' or 'elapsed'")
    prof["discount"] = (1.0 + p.discount_rate) ** (-t)
    prof["net_cash_usd"] = prof["revenue_usd"] - prof["opex_usd"]
    prof["pv_net_usd"] = prof["discount"] * prof["net_cash_usd"]

    npv = -capex_wells - capex_fac + prof["pv_net_usd"].sum()

    return {
        "case": name,
        "injectors": ";".join(f"{i}-{j}" for i, j in injectors),
        "producers": ";".join(f"{i}-{j}" for i, j in producers),
        "facility_ij": f"{facility_ij[0]}-{facility_ij[1]}" if facility_ij else "",
        "n_wells": len(geo.wells),
        "total_well_length_m": geo.total_drilled_length_m,
        "flowline_length_m": geo.total_flowline_length_m,
        "n_years": len(prof),
        # energy / water
        "net_energy_PJ": prof["energy_pj"].sum(),
        "energy_kwh": prof["energy_kwh"].sum(),
        "water_m3": prof["water_m3"].sum(),
        # revenue
        "revenue_usd": prof["revenue_usd"].sum(),
        "revenue_pv_usd": (prof["discount"] * prof["revenue_usd"]).sum(),
        # CAPEX breakdown
        "capex_wells_usd": capex_wells,
        "capex_facility_fixed_usd": geo.capex_surface_usd,
        "capex_flowline_usd": geo.capex_flowline_usd,
        "capex_facilities_usd": capex_fac,
        "capex_total_usd": capex_total,
        # OPEX breakdown, undiscounted totals and present values
        "opex_wells_usd": prof["opex_wells_usd"].sum(),
        "opex_water_usd": prof["opex_water_usd"].sum(),
        "opex_facility_om_usd": prof["opex_om_usd"].sum(),
        "opex_total_usd": prof["opex_usd"].sum(),
        "opex_wells_pv_usd": (prof["discount"] * prof["opex_wells_usd"]).sum(),
        "opex_water_pv_usd": (prof["discount"] * prof["opex_water_usd"]).sum(),
        "opex_facility_om_pv_usd": (prof["discount"] * prof["opex_om_usd"]).sum(),
        "opex_pv_usd": (prof["discount"] * prof["opex_usd"]).sum(),
        # result
        "pv_net_cash_usd": prof["pv_net_usd"].sum(),
        "npv_usd": npv,
        "_profile": prof,     # per-period detail, dropped by cases_to_frame()
    }


def cases_to_frame(rows: Iterable[dict]) -> "pd.DataFrame":
    """Tabulate `evaluate_case` results, one row per case."""
    if pd is None:
        raise ImportError("pandas is required for cases_to_frame()")
    return pd.DataFrame([{k: v for k, v in r.items() if k != "_profile"}
                         for r in rows])


def load_case_table(path: str, case_col: str = "case") -> "pd.DataFrame":
    """Load the case-definition CSV (one row per case id)."""
    if pd is None:
        raise ImportError("pandas is required for load_case_table()")
    df = pd.read_csv(path)
    if case_col not in df.columns:
        raise KeyError(f"{case_col!r} not among the columns of {path}")
    return df


def wells_from_row(row) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Pull well cells out of a case-definition row.

    Expects paired columns named INJ1_I / INJ1_J, PRO1_I / PRO1_J, ... and
    works for any number of wells per type.
    """
    def collect(prefix):
        idx = sorted(
            int(m.group(1))
            for col in row.index
            if (m := re.fullmatch(rf"{prefix}(\d+)_I", str(col)))
        )
        return [(int(row[f"{prefix}{n}_I"]), int(row[f"{prefix}{n}_J"])) for n in idx]

    injectors, producers = collect("INJ"), collect("PRO")
    if not injectors or not producers:
        raise KeyError("no INJn_I/INJn_J or PROn_I/PROn_J columns in the case table")
    return injectors, producers


def case_id_from_name(name: str) -> int | None:
    """Trailing integer of a case name, e.g. 'case12' -> 12."""
    m = re.search(r"(\d+)\s*$", name)
    return int(m.group(1)) if m else None



def find_case_files(
    folder: str,
    pattern: str = "case*_energy_water.csv",
    suffix: str = "_energy_water",
) -> list[tuple[str, str]]:
    """
    List (case_name, path) for every matching CSV in `folder`, ordered
    naturally so case2 comes before case10.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {folder!r}")

    def natural_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return [int(tok) if tok.isdigit() else tok.lower()
                for tok in re.split(r"(\d+)", stem)]

    out = []
    for path in sorted(paths, key=natural_key):
        stem = os.path.splitext(os.path.basename(path))[0]
        name = stem[: -len(suffix)] if suffix and stem.endswith(suffix) else stem
        out.append((name, path))
    return out


def evaluate_folder(
    grid: np.ndarray,
    folder: str,
    injectors: Sequence[tuple[int, int]] | None = None,
    producers: Sequence[tuple[int, int]] | None = None,
    params: EconParams | None = None,
    facility_xy: Sequence[float] | None = None,
    facility_ij: Sequence[int] | None = None,
    placements: dict | None = None,
    case_table: "pd.DataFrame | None" = None,
    case_col: str = "case",
    merge_case_table: bool = True,
    keep_all_cases: bool = True,
    pattern: str = "case*_energy_water.csv",
    energy_basis: str = "net",
    water_basis: str = "produced",
    k_top: int = 5,
    k_bot: int = 20,
    one_based: bool = True,
    discount_on: str = "period",
    npv_end_date: str | None = None,
    out_csv: str | None = None,
    columns: Sequence[str] | None = None,
    drop_columns: Sequence[str] | None = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    NPV for every caseN_energy_water.csv in `folder`, one row per case.

    Well placement per case, in order of precedence:

    1. `placements`, mapping a case name to overrides, e.g.
       {"case7": {"injectors": [...], "producers": [...]}}
    2. `case_table`, a case-definition frame whose INJn_I/INJn_J and
       PROn_I/PROn_J columns are read for the matching case id (taken from the
       trailing integer of the file name). With `merge_case_table`, every
       column of that row is carried into the output.
    3. the `injectors` / `producers` arguments, shared by all cases.

    `npv_end_date` evaluates every case to the same cut-off date; a case whose
    profile stops short of it gets no NPV instead of a short-horizon one.

    A case whose file cannot be read is skipped with a warning rather than
    aborting the batch. With `keep_all_cases` (the default) every row of
    `case_table` still appears in the output, with the economics columns left
    empty where no NPV could be computed. Writes `out_csv` if given and returns
    the frame.
    """
    if pd is None:
        raise ImportError("pandas is required for evaluate_folder()")

    files = find_case_files(folder, pattern)
    if verbose:
        print(f"found {len(files)} case file(s) in {folder}")

    placements = placements or {}
    table = None
    if case_table is not None:
        table = case_table.set_index(case_col)

    rows, failed, ids = [], [], []
    for name, path in files:
        over = dict(placements.get(name, {}))
        cid = case_id_from_name(name)

        if table is not None and "injectors" not in over:
            if cid is None or cid not in table.index:
                failed.append((name, KeyError(f"case id {cid} not in the case table")))
                warnings.warn(f"{name}: no matching row in the case table, skipped",
                              stacklevel=2)
                continue
            trow = table.loc[cid]
            inj_c, pro_c = wells_from_row(trow)
            over.setdefault("injectors", inj_c)
            over.setdefault("producers", pro_c)

        if over.get("injectors") is None or over.get("producers") is None:
            if injectors is None or producers is None:
                raise ValueError(
                    "no placement for "
                    f"{name}: pass case_table, placements, or injectors/producers")

        try:
            prof = load_profile(path, energy_basis=energy_basis,
                                water_basis=water_basis)
            rows.append(evaluate_case(
                grid,
                over.get("injectors", injectors),
                over.get("producers", producers),
                prof,
                params=params,
                facility_xy=over.get("facility_xy", facility_xy),
                facility_ij=over.get("facility_ij", facility_ij),
                k_top=k_top, k_bot=k_bot, one_based=one_based,
                name=name, discount_on=discount_on,
                npv_end_date=npv_end_date,
            ))
            ids.append(cid)
        except Exception as exc:                      # noqa: BLE001
            failed.append((name, exc))
            warnings.warn(f"{name}: skipped ({type(exc).__name__}: {exc})",
                          stacklevel=2)

    if not rows and not (keep_all_cases and table is not None):
        raise RuntimeError("no case evaluated successfully")

    df = cases_to_frame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["case"] + ECON_COLUMNS)

    if table is not None and merge_case_table:
        # key the economics by case id, then join onto the case definitions so
        # every case survives even when its NPV could not be computed
        econ = df.drop(columns=[case_col], errors="ignore")
        econ.insert(0, case_col, ids)
        left = table.reset_index()
        keep = left if keep_all_cases else left[left[case_col].isin(ids)]
        df = keep.merge(econ, on=case_col, how="left")
        df = df[[c for c in df.columns if c != "npv_usd"] + ["npv_usd"]]

    if drop_columns is None:
        drop_columns = DROPPED_COLUMNS
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    if columns is not None:
        df = df[list(columns)]

    if out_csv:
        df.to_csv(out_csv, index=False)
        if verbose:
            n_ok = int(df["npv_usd"].notna().sum()) if "npv_usd" in df else 0
            print(f"written: {out_csv}  ({len(df)} rows, {n_ok} with an NPV)")
    if verbose and failed:
        print(f"skipped {len(failed)}: {[n for n, _ in failed]}")
    return df


# redundant with their present-value / component counterparts
DROPPED_COLUMNS = ["revenue_usd", "capex_facilities_usd", "pv_net_cash_usd"]

# every column `evaluate_case` contributes, in order; used to keep the output
# schema stable when no case could be evaluated at all
ECON_COLUMNS = [
    "injectors", "producers", "facility_ij", "n_wells",
    "total_well_length_m", "flowline_length_m", "n_years",
    "net_energy_PJ", "energy_kwh", "water_m3",
    "revenue_usd", "revenue_pv_usd",
    "capex_wells_usd", "capex_facility_fixed_usd", "capex_flowline_usd",
    "capex_facilities_usd", "capex_total_usd",
    "opex_wells_usd", "opex_water_usd", "opex_facility_om_usd", "opex_total_usd",
    "opex_wells_pv_usd", "opex_water_pv_usd", "opex_facility_om_pv_usd",
    "opex_pv_usd",
    "pv_net_cash_usd", "npv_usd",
]

SUMMARY_COLUMNS = [
    "case", "injectors", "producers", "facility_ij", "n_wells", "n_years",
    "total_well_length_m", "flowline_length_m",
    "net_energy_PJ", "energy_kwh", "water_m3",
    "revenue_pv_usd",
    "capex_wells_usd", "capex_facility_fixed_usd", "capex_flowline_usd",
    "capex_total_usd",
    "opex_wells_pv_usd", "opex_water_pv_usd", "opex_facility_om_pv_usd",
    "opex_pv_usd",
    "npv_usd",
]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _ij_arg(text: str) -> tuple[int, int]:
    """Parse an 'i,j' command-line argument."""
    i, j = text.split(",")
    return int(i), int(j)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="NPV for every caseN_energy_water.csv in a folder.")
    ap.add_argument("folder", help="folder holding the case CSVs")
    ap.add_argument("--grid", default="JD_geothermal_coor_fault.npy",
                    help="path to the (ni,nj,nk,4) coordinate/fault .npy")
    ap.add_argument("--out", default="npv_summary.csv", help="output CSV")
    ap.add_argument("--cases", default=None,
                    help="case-definition CSV; well cells are read per case from "
                         "its INJn_I/INJn_J and PROn_I/PROn_J columns and all of "
                         "its columns are carried into the output")
    ap.add_argument("--injectors", nargs="+", type=_ij_arg, default=None,
                    metavar="i,j", help="injector cells, e.g. 50,110 54,97 63,109 "
                                        "(ignored when --cases is given)")
    ap.add_argument("--producers", nargs="+", type=_ij_arg, default=None,
                    metavar="i,j", help="producer cells")
    ap.add_argument("--facility", type=_ij_arg, default=None, metavar="i,j",
                    help="shared pad by grid index; default is the well centroid")
    ap.add_argument("--k-top", type=int, default=5)
    ap.add_argument("--k-bot", type=int, default=20)
    ap.add_argument("--energy-basis", default="net", choices=["net", "produced"])
    ap.add_argument("--water-basis", default="produced",
                    choices=["produced", "injected", "both", "max"])
    ap.add_argument("--npv-end-date", default=None, metavar="YYYY-MM-DD",
                    help="evaluate every case to this date; cases whose profile "
                         "stops short of it get no NPV")
    ap.add_argument("--pattern", default="case*_energy_water.csv")
    ap.add_argument("--all-columns", action="store_true",
                    help="write every computed column, not just SUMMARY_COLUMNS")
    args = ap.parse_args(argv)

    if args.cases is None and (args.injectors is None or args.producers is None):
        ap.error("give --cases, or both --injectors and --producers")

    grid = load_grid(args.grid)
    table = load_case_table(args.cases) if args.cases else None
    df = evaluate_folder(
        grid, args.folder, args.injectors, args.producers,
        case_table=table,
        facility_ij=args.facility,
        pattern=args.pattern,
        energy_basis=args.energy_basis, water_basis=args.water_basis,
        npv_end_date=args.npv_end_date,
        k_top=args.k_top, k_bot=args.k_bot,
        out_csv=args.out,
        columns=(SUMMARY_COLUMNS if (args.cases is None and not args.all_columns)
                 else None),
    )
    mm = df["npv_usd"] / 1e6
    print(f"NPV (MM$): min {mm.min():.1f}  median {mm.median():.1f}  "
          f"max {mm.max():.1f}  best {df.loc[mm.idxmax(), 'case']}")
    return df


if __name__ == "__main__":
    main()
