"""Tests for pre_generate_temperature.py.

Run from the repo root with:  python -m pytest 02_code/tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pre_generate_temperature import (  # noqa: E402
    load_cell_depths, load_temp_gradients, temperature_from_gradient)


def test_load_temp_gradients(tmp_path):
    p = tmp_path / "params.csv"
    p.write_text(
        "Ensemble name:,'x/260904'\nNumber of realizations:,2\nNumber of parameters:,2\n\n"
        "Parameter,Temp_grad,P_mean,RESULT(string)\n"
        "Property id\nUnit\nDate\nIdentifier\nSecondary identifier\n"
        "JD_geothermal_1,0.025,0.1,Complete\n"
        "JD_geothermal_2,0.035,0.2,None\n")
    assert load_temp_gradients(p) == {"JD_GEOTHERMAL_1": 0.025, "JD_GEOTHERMAL_2": 0.035}


def test_load_cell_depths_matches_property_file_order(tmp_path):
    ni, nj, nk = 3, 4, 2
    # depth that encodes the (i, j, k) index of each cell, j in the CMG order as stored in the file
    i, j, k = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    coor = np.zeros((ni, nj, nk, 6))
    coor[..., 2] = 1000 * k + 10 * j + i
    np.save(tmp_path / "coor.npy", coor)
    flat = load_cell_depths(tmp_path / "coor.npy")
    assert flat.size == ni * nj * nk
    # property files are flattened with i fastest, then j, then k
    for ii, jj, kk in [(0, 0, 0), (2, 1, 0), (1, 3, 1)]:
        assert flat[kk * nj * ni + jj * ni + ii] == 1000 * kk + 10 * jj + ii


def test_temperature_from_gradient():
    depth = np.array([0.0, 1000.0, 3000.0])
    np.testing.assert_allclose(temperature_from_gradient(depth, 0.03, 11), [11, 41, 101])
    with pytest.raises(ValueError):
        temperature_from_gradient(np.array([1.0, np.nan]), 0.03)
