"""Tests for pre_extract_properties_from_finit.py on small synthetic Eclipse formatted files.

Run from the repo root with:  python -m pytest 02_code/tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pre_extract_properties_from_finit import extract_properties_from_finit  # noqa: E402

PORO = [0.1, 0.2, 0.3, 0.4, 0.5]
ACTID = [3, 5, 7, 9, 11]


def finit_text(poro_declared=5, poro=PORO, extra=""):
    lines = [
        " 'INTEHEAD'           3 'INTE'",
        "           1           2           3",
        " 'LOGIHEAD'           2 'LOGI'",
        "  T  F",
        " 'POROV   '           2 'REAL'",   # name starts with PORO: must not be picked up as PORO
        "   0.90000000E+00   0.80000000E+00",
        f" 'PORO    '           {poro_declared} 'REAL'",
        "   " + "   ".join(f"{v:.8E}" for v in poro[:3]),
        "   " + "   ".join(f"{v:.8E}" for v in poro[3:]),
        " 'ACTID   '           5 'INTE'",
        "   " + "   ".join(str(v) for v in ACTID),
        " 'TAB     '           2 'DOUB'",
        "   0.10000000000000D+01   0.20000000000000D+01",
    ]
    return "\n".join(lines) + "\n" + extra


def write(tmp_path, text, name="CASE_1.FINIT"):
    p = tmp_path / name
    p.write_bytes(text.encode("ascii") if isinstance(text, str) else text)
    return p


def test_clean_file(tmp_path):
    p = write(tmp_path, finit_text())
    out = extract_properties_from_finit(p, ["PORO", "ACTID", "TAB"])
    np.testing.assert_array_equal(out["PORO"], PORO)
    np.testing.assert_array_equal(out["ACTID"], ACTID)
    np.testing.assert_array_equal(out["TAB"], [1.0, 2.0])  # D exponent


def test_nul_bytes_rejected(tmp_path):
    data = bytearray(finit_text().encode("ascii"))
    data[-40:-20] = b"\x00" * 20  # simulate a zero-filled block, same file size
    p = write(tmp_path, bytes(data))
    with pytest.raises(ValueError, match="Corrupted file: 20 NUL bytes"):
        extract_properties_from_finit(p, ["PORO", "ACTID"])


def test_nul_bytes_allowed_when_check_disabled(tmp_path):
    # NUL block in a section that is not extracted
    data = finit_text().replace("0.90000000E+00", "\x00" * 14)
    p = write(tmp_path, data)
    out = extract_properties_from_finit(p, ["PORO"], check_integrity=False)
    np.testing.assert_array_equal(out["PORO"], PORO)


def test_size_mismatch_rejected(tmp_path):
    p = write(tmp_path, finit_text(poro_declared=6))
    with pytest.raises(ValueError, match="PORO header declares 6 values but 5 were read"):
        extract_properties_from_finit(p, ["PORO"])


def test_missing_keyword_rejected(tmp_path):
    p = write(tmp_path, finit_text())
    with pytest.raises(ValueError, match="Property PERMX not found"):
        extract_properties_from_finit(p, ["PORO", "PERMX"])


def test_duplicate_section_rejected(tmp_path):
    p = write(tmp_path, finit_text(extra=" 'PORO    '           1 'REAL'\n   0.1E+00\n"))
    with pytest.raises(ValueError, match="appears more than once"):
        extract_properties_from_finit(p, ["PORO"])


def test_non_numeric_value_rejected(tmp_path):
    p = write(tmp_path, finit_text().replace("4.00000000E-01", "4.0000xxxxE-01"))
    with pytest.raises(ValueError, match="Non-numeric value"):
        extract_properties_from_finit(p, ["PORO"])


def test_nan_rejected(tmp_path):
    p = write(tmp_path, finit_text().replace("4.00000000E-01", "nan"))
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_properties_from_finit(p, ["PORO"])
