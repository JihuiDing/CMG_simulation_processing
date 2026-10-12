"""Tests for the importance sampling in pre_parameter_sampling.py.

Run from the repo root with:  python -m pytest 02_code/tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pre_parameter_sampling as ps  # noqa: E402

# target distributions hard-coded in IS_SH_azi_SH_uniform
AZI_LOW, AZI_HIGH = 300, 320
SH_LOW, SH_HIGH = 18 * 0.9, 18 * 1.1


def run_is(tmp_path, monkeypatch, n_samples, alpha=0.9, beta=0.9,
           proposal_SH_azi_low=319, proposal_SH_low=18 * 1.05, seed=0):
    """Run IS_SH_azi_SH_uniform and return its array at full precision
    (the csv is written with 4 decimals, too coarse for normalised weights)."""
    captured = {}
    real_savetxt = np.savetxt

    def capture(fname, X, *args, **kwargs):
        captured["X"] = np.array(X, copy=True)
        real_savetxt(fname, X, *args, **kwargs)

    monkeypatch.setattr(ps.np, "savetxt", capture)
    ps.IS_SH_azi_SH_uniform(
        random_seed=seed,
        name_prefix="test",
        output_file_path=Path(tmp_path),
        n_samples=n_samples,
        alpha=alpha,
        beta=beta,
        proposal_SH_azi_low=proposal_SH_azi_low,
        proposal_SH_low=proposal_SH_low,
        show_summary=False,
    )
    X = captured["X"]
    # columns: SH_azi_deg, q_SH_azi, SH_MPa/km (negative), q_SH, weights
    return X[:, 0], X[:, 1], -X[:, 2], X[:, 3], X[:, 4]


def test_proposal_pdf_matches_mixture(tmp_path, monkeypatch):
    alpha, beta, azi_pl, sh_pl = 0.9, 0.8, 319, 18 * 1.05
    azi, q_azi, sh, q_sh, w = run_is(tmp_path, monkeypatch, 500, alpha, beta, azi_pl, sh_pl)

    upper = azi >= azi_pl
    np.testing.assert_allclose(q_azi[upper], alpha / (AZI_HIGH - azi_pl))
    np.testing.assert_allclose(q_azi[~upper], (1 - alpha) / (azi_pl - AZI_LOW))

    upper = sh >= sh_pl
    np.testing.assert_allclose(q_sh[upper], beta / (SH_HIGH - sh_pl))
    np.testing.assert_allclose(q_sh[~upper], (1 - beta) / (sh_pl - SH_LOW))

    assert np.all(w > 0)
    assert np.isclose(w.sum(), 1.0)


def test_weights_recover_target_uniform(tmp_path, monkeypatch):
    # with correct weights, weighted estimates under the proposal match U(target)
    azi_pl, sh_pl = 319, 18 * 1.05
    azi, _, sh, _, w = run_is(tmp_path, monkeypatch, 200_000, 0.9, 0.9, azi_pl, sh_pl)

    assert np.sum(w * azi) == pytest.approx((AZI_LOW + AZI_HIGH) / 2, abs=0.2)
    assert np.sum(w * sh) == pytest.approx((SH_LOW + SH_HIGH) / 2, abs=0.05)
    assert np.sum(w[azi < azi_pl]) == pytest.approx((azi_pl - AZI_LOW) / (AZI_HIGH - AZI_LOW), abs=0.01)
    assert np.sum(w[sh < sh_pl]) == pytest.approx((sh_pl - SH_LOW) / (SH_HIGH - SH_LOW), abs=0.01)


# ---- merge_petrel_and_sampled_parameters ----
PETREL_CSV = """Ensemble name:,'test'
Number of realizations:,3
Number of parameters:,2

Parameter,Temp_grad,LOOP,RESULT(string)
Property id
Unit
JD_geothermal_10,0.0261181951353496,1,Complete
JD_geothermal_11,0.03,2,
JD_geothermal_12,0.0277625690481277,3,Complete
"""
SAMPLED_CSV = """E_kPa,PORO_file
3.45766e+07,JD_GEOTHERMAL_12_PORO.dat
3.15114e+07,JD_GEOTHERMAL_10_PORO.dat
"""


def write_inputs(tmp_path, petrel=PETREL_CSV, sampled=SAMPLED_CSV):
    (tmp_path / "petrel.csv").write_text(petrel)
    (tmp_path / "sampled.csv").write_text(sampled)
    return tmp_path / "petrel.csv", tmp_path / "sampled.csv", tmp_path / "out.csv"


def test_merge_pairs_by_realization_and_keeps_text(tmp_path):
    petrel, sampled, out = write_inputs(tmp_path)
    ps.merge_petrel_and_sampled_parameters(petrel, sampled, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "Parameter,Temp_grad,RESULT(string),E_kPa,PORO_file"
    # order of the sampled file is kept, and digits are not reformatted
    assert lines[1] == "JD_geothermal_12,0.0277625690481277,Complete,3.45766e+07,JD_GEOTHERMAL_12_PORO.dat"
    assert lines[2] == "JD_geothermal_10,0.0261181951353496,Complete,3.15114e+07,JD_GEOTHERMAL_10_PORO.dat"


def test_merge_rejects_failed_or_missing_realization(tmp_path):
    petrel, sampled, out = write_inputs(tmp_path, sampled="E_kPa,PORO_file\n1,JD_GEOTHERMAL_11_PORO.dat\n")
    with pytest.raises(ValueError, match="not Complete"):
        ps.merge_petrel_and_sampled_parameters(petrel, sampled, out)
    petrel, sampled, out = write_inputs(tmp_path, sampled="E_kPa,PORO_file\n1,JD_GEOTHERMAL_99_PORO.dat\n")
    with pytest.raises(ValueError, match="No Petrel parameters"):
        ps.merge_petrel_and_sampled_parameters(petrel, sampled, out)
