import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union


def load_temp_gradients(
        sampled_params_path: Union[str, Path],
        column: str = 'Temp_grad'
        ) -> Dict[str, float]:
    """
    Read the temperature gradient sampled for each realization from a Petrel
    Uncertainty & Optimization export (e.g. 260904_realization_sampled_parameters.csv).

    Args:
        sampled_params_path: path to the Petrel csv (4 header lines, then a 'Parameter' table)
        column: name of the temperature gradient column (C/m)
    Returns:
        dict: realization name in upper case (e.g. 'JD_GEOTHERMAL_112') -> temperature gradient
    """
    df = pd.read_csv(sampled_params_path, skiprows=4)
    # rows after the header include metadata rows ('Property id', 'Unit', ...) without values
    df = df[pd.to_numeric(df[column], errors='coerce').notna()]
    return {name.upper(): float(value) for name, value in zip(df['Parameter'], df[column])}


def load_cell_depths(coor_file_path: Union[str, Path]) -> np.ndarray:
    """
    Load the cell-centre depth of every cell, flattened in the same order as the
    CMG property files (i fastest, then j in the CMG order, then k).

    The coordinate file (e.g. JD_geothermal_coor&fault&dip.npy) has shape (n_i, n_j, n_k, n)
    with j already in the CMG order and z (depth, m, positive down) in column 2.
    """
    coor = np.load(coor_file_path)
    return coor[..., 2].transpose(2, 1, 0).ravel()


def temperature_from_gradient(
        depth: np.ndarray,
        temp_grad: float,
        temp_surface: float = 11.0
        ) -> np.ndarray:
    """
    Temperature (C) from a linear gradient: TEMP = temp_grad * depth + temp_surface.

    Args:
        depth: cell depths (m, positive down)
        temp_grad: temperature gradient (C/m)
        temp_surface: temperature at zero depth (C)
    """
    if not np.all(np.isfinite(depth)):
        raise ValueError("Non-finite cell depths")
    return temp_grad * depth + temp_surface
