#!/usr/bin/env python3
"""
Robust version of CMG2npy.py.

Same interface as CMG2npy, but hardened against the silent partial-read failure
seen when parsing rwo files off a mapped network drive (Oak via SMB):

  Warning: Expected 139 values, got 76 for K=1, J=84, Time=0.0
  Warning: Expected 139 values, got 216 for K=1, J=246, Time=0.0

In the original the affected rows were skipped, leaving zeros in the array that
later became NaN, so the loss was invisible unless the log happened to be read.

Changes:
  1. The rwo file is read with a single read() instead of iterating ~700k lines,
     so one bulk transfer replaces one network round trip per line. This removes
     the failure mechanism and is much faster over SMB.
  2. The number of '** K =' headers is checked against n_j * n_k * n_times
     before the data is trusted.
  3. A row with the wrong number of values now raises instead of warning.
  4. run_CMG_rwd_report waits until the rwo file size stops changing, so a
     write still sitting in the Windows client cache is not read too early,
     and it checks the Report.exe exit code. os.system returns the exit code
     rather than raising, so the original try/except never fired and a failed
     run produced a missing or truncated rwo with no error.
  5. extract_case_local() runs the whole sr3 -> rwo -> npy chain on local disk
     and writes back only the npy, which keeps the large intermediate files off
     the network share.

Note: requesting several properties from one rwd (multiple *OUTPUT /
*PROPERTY-FOR blocks) was tried and does not work with CMG Results 2024.20 as
invoked here, most likely because the command line already passes a single
-o output target. One rwd per property remains the only supported form.
"""

import numpy as np
import re
import os
import time
import subprocess
from pathlib import Path


def generate_CMG_rwd(
    sr3_folder_path: str = None,
    case_name: str = None,
    property: str = 'PRES',
    sim_results_file_format: str = 'sr3',
    precision: int = 4
    ):
    """
    Write a rwd file for a CMG simulation result sr3 file. Unchanged from CMG2npy.
    """
    sr3_folder = Path(sr3_folder_path)
    if not sr3_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {sr3_folder}")

    if sim_results_file_format not in ('sr3', 'gmch.sr3'):
        raise ValueError("Use correct file format: sr3 or gmch.sr3")
    case_file = sr3_folder / f"{case_name}.{sim_results_file_format}"

    if not case_file.is_file():
        raise FileNotFoundError(f"Case not found: {case_file}")

    rwo_folder = sr3_folder / "rwo"
    rwo_folder.mkdir(parents=True, exist_ok=True)

    rwd_file = sr3_folder / f"{case_name}.rwd"

    with open(rwd_file, 'w') as f:
        f.write(f"*FILES \t '{case_name}.{sim_results_file_format}' \n")
        f.write(f"*PRECISION \t {precision} \n")
        f.write(f"*OUTPUT \t 'rwo\\{case_name}_{property}.rwo' \n")
        f.write(f"*PROPERTY-FOR \t '{property}' \t *ALL-TIMES \n")


def wait_until_stable(file_path, checks: int = 3, interval: float = 1.0, timeout: float = 300.0):
    """
    Wait until a file exists and its size stops changing.

    Report.exe can return while its output is still in the Windows client cache
    for a mapped drive, so reading immediately afterwards can catch a partly
    written file. Requiring the size to repeat `checks` times closes that window.
    """
    file_path = Path(file_path)
    t_start = time.time()
    last, stable = -1, 0
    while time.time() - t_start < timeout:
        size = file_path.stat().st_size if file_path.exists() else -1
        if size >= 0 and size == last:
            stable += 1
            if stable >= checks:
                return size
        else:
            stable = 0
        last = size
        time.sleep(interval)
    raise TimeoutError(f"File size never settled within {timeout}s: {file_path}")


def run_CMG_rwd_report(
    rwd_folder_path: str,
    case_name: str,
    property = None,
    cmg_version: str = 'ese-ts2win-v2024.20',
    wait_for_output: bool = True,
    ):
    """
    Run the rwd report, check that it succeeded, and optionally wait for the
    output to finish being written.

    `property` may be a single property name or a list of them. Every
    corresponding rwo file is waited on. Pass None to skip the wait.

    Unlike CMG2npy, the exit code is checked. os.system returns the exit code
    rather than raising, so the original try/except never fired and a failed
    Report.exe produced a missing or truncated rwo without any error.

    The working directory is set with subprocess's cwd= rather than a shell
    'cd'. On Windows a plain 'cd' does not switch drives, so running from a
    mapped network drive while the rwd sits on local disk left Report.exe in
    the wrong place and it reported "Input file 'caseN.rwd' is not accessible".
    Passing cwd= changes drive and directory together, and dropping the shell
    also removes any quoting problems with spaces in the exe path.
    """
    rwd_folder = Path(rwd_folder_path)
    if not rwd_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {rwd_folder}")
    rwd_file = rwd_folder / f"{case_name}.rwd"
    if not rwd_file.is_file():
        raise FileNotFoundError(f"rwd file not found: {rwd_file}")

    if cmg_version == 'ese-ts2win-v2024.20':
        # no surrounding quotes: the argument list form does not go through a shell
        exe_path = r"C:\Program Files\CMG\RESULTS\2024.20\Win_x64\exe\Report.exe"
    else:
        raise ValueError(f'The CMG version {cmg_version} is not implemented yet .....')

    result = subprocess.run(
        [exe_path, "-f", f"{case_name}.rwd", "-o", case_name],
        cwd=str(rwd_folder),
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{case_name}: Report.exe exited with code {result.returncode}\n"
            f"  ran in: {rwd_folder}\n"
            f"  stdout: {(result.stdout or '')[-500:]}\n"
            f"  stderr: {(result.stderr or '')[-500:]}")

    # make sure the outputs have actually landed before anyone reads them
    if wait_for_output and property is not None:
        props = [property] if isinstance(property, str) else list(property)
        for prop in props:
            wait_until_stable(rwd_folder / "rwo" / f"{case_name}_{prop}.rwo")


def CMG_rwo2npy(
    rwo_folder_path: str,
    case_name: str,
    property: str = 'PRES',
    is_save: bool = False,
    save_folder_path: str = "results",
    show_info: bool = False,
    expected_shape: tuple = None,
    expected_active: int = None,
    copy_local_first: bool = False,
    dtype = None,
    ):
    """
    Parse a CMG rwo file into a numpy array of shape (n_i, n_j, n_k, n_time).

    Extra arguments over CMG2npy.CMG_rwo2npy:
        expected_shape   : (n_i, n_j, n_k) to assert against, e.g. (139, 248, 23)
        expected_active  : number of non-zero cells at the first time step,
                           taken from a reference case; guards against a read
                           that is complete in structure but short on data
        copy_local_first : copy the rwo to a local temp file before parsing,
                           for use when the network share is unreliable
        dtype            : e.g. np.float32 to halve the array and file size.
                           CMG writes 4 significant digits (*PRECISION 4) and
                           float32 holds about 7, so nothing real is lost.
                           None keeps the float64 default.
    """
    rwo_file_path = Path(rwo_folder_path) / f"{case_name}_{property}.rwo"
    if not rwo_file_path.exists():
        raise FileNotFoundError(f"File not found: {rwo_file_path}")

    save_folder_path = Path(save_folder_path)
    save_folder_path.mkdir(parents=True, exist_ok=True)

    # ---- read the whole file in one call -------------------------------------
    # iterating ~700k lines over SMB gives one round trip per line, and a short
    # read part way through silently loses a chunk. One read() avoids both.
    if copy_local_first:
        import shutil, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / rwo_file_path.name
            shutil.copy2(rwo_file_path, local)
            src_size = rwo_file_path.stat().st_size
            if local.stat().st_size != src_size:
                raise IOError(f"{case_name}_{property}: local copy is "
                              f"{local.stat().st_size} bytes, source is {src_size}")
            raw = local.read_bytes()
    else:
        with open(rwo_file_path, 'rb') as fh:
            raw = fh.read()

    on_disk = rwo_file_path.stat().st_size
    if len(raw) != on_disk:
        raise IOError(f"{case_name}_{property}: read {len(raw)} bytes but the file "
                      f"is {on_disk} bytes — incomplete read")

    text = raw.decode('ascii', errors='ignore')
    lines = text.splitlines()

    # ---- structural check before trusting any of it --------------------------
    n_time_hdr = text.count('**  TIME =')
    n_kj_hdr = text.count('** K =')
    if n_time_hdr == 0 or n_kj_hdr == 0:
        raise ValueError(f"{case_name}_{property}: no TIME or K,J headers found")
    if expected_shape is not None:
        _, n_j_exp, n_k_exp = expected_shape
        expected_hdr = n_j_exp * n_k_exp * n_time_hdr
        if n_kj_hdr != expected_hdr:
            raise ValueError(
                f"{case_name}_{property}: found {n_kj_hdr} K,J headers but expected "
                f"{expected_hdr} ({n_j_exp} x {n_k_exp} x {n_time_hdr} times) — incomplete read")

    # ---- parse ---------------------------------------------------------------
    time_values, time_dates, pressure_data = [], [], []
    current_time, current_data = None, []

    for line in lines:
        line = line.strip()

        if line.startswith("**  TIME ="):
            if current_time is not None and current_data:
                pressure_data.append({'time': current_time,
                                      'date': time_dates[-1] if time_dates else '',
                                      'data': current_data.copy()})
            match = re.match(r'\*\*  TIME = (\d+(?:\.\d+)?)\s*(.*)', line)
            if match:
                current_time = float(match.group(1))
                time_values.append(current_time)
                time_dates.append(match.group(2).strip())
                current_data = []
                if show_info:
                    print(f"Processing time step: {current_time} ({time_dates[-1]})")

        elif line.startswith("** K ="):
            match = re.match(r'\*\* K = (\d+), J = (\d+)', line)
            if match:
                current_data.append({'k': int(match.group(1)),
                                     'j': int(match.group(2)),
                                     'values': []})

        elif line and not line.startswith("**") and not line.startswith("RESULTS") \
                and not line.startswith(property):
            try:
                values = [float(x) for x in line.split()]
            except ValueError:
                continue
            if current_data and 'values' in current_data[-1]:
                current_data[-1]['values'].extend(values)

    if current_time is not None and current_data:
        pressure_data.append({'time': current_time,
                              'date': time_dates[-1] if time_dates else '',
                              'data': current_data})

    if show_info:
        print(f"Found {len(time_values)} time steps")

    # ---- grid dimensions -----------------------------------------------------
    k_values, j_values, i_count = set(), set(), 0
    for time_data in pressure_data:
        for cell_data in time_data['data']:
            k_values.add(cell_data['k'])
            j_values.add(cell_data['j'])
            if i_count == 0:
                i_count = len(cell_data['values'])

    n_k, n_j, n_i, n_time = max(k_values), max(j_values), i_count, len(time_values)

    if expected_shape is not None and (n_i, n_j, n_k) != tuple(expected_shape):
        raise ValueError(f"{case_name}_{property}: inferred grid {(n_i, n_j, n_k)} "
                         f"does not match expected {tuple(expected_shape)}")

    if show_info:
        print(f"Grid dimensions: I={n_i}, J={n_j}, K={n_k}, Time={n_time}")

    # ---- fill ----------------------------------------------------------------
    sim_results = np.zeros((n_i, n_j, n_k, n_time))
    for time_idx, time_data in enumerate(pressure_data):
        for cell_data in time_data['data']:
            k = cell_data['k'] - 1
            j = cell_data['j'] - 1
            vals = cell_data['values']
            if len(vals) != n_i:
                # was a warning in CMG2npy, which left this row as zeros
                raise ValueError(
                    f"{case_name}_{property}: row K={k+1}, J={j+1}, Time={time_data['time']} "
                    f"has {len(vals)} values, expected {n_i} — incomplete read")
            sim_results[:, j, k, time_idx] = vals

    # ---- data-completeness check --------------------------------------------
    n_active = int((sim_results[..., 0] != 0).sum())
    if expected_active is not None and n_active != expected_active:
        raise ValueError(f"{case_name}_{property}: {n_active} active cells at t0, "
                         f"expected {expected_active} — incomplete read")
    if show_info:
        print(f"Active cells at t0: {n_active}")

    # cast after the checks, so the active-cell count is done on the parsed values
    if dtype is not None:
        sim_results = sim_results.astype(dtype)

    if is_save:
        np.save(save_folder_path / f"{case_name}_{property}.npy", sim_results)

    return sim_results


def extract_case_local(
    sr3_folder_path: str,
    case_name: str,
    property_list: list,
    save_folder_path: str,
    sim_results_file_format: str = 'sr3',
    precision: int = 4,
    cmg_version: str = 'ese-ts2win-v2024.20',
    local_work_dir: str = None,
    expected_shape: tuple = None,
    expected_active: int = None,
    dtype = None,
    keep_rwo: bool = False,
    show_info: bool = False,
    ):
    """
    Run the whole sr3 -> rwd -> rwo -> npy chain for one case on LOCAL disk.

    Running in place on a mapped network drive moves a lot of data over SMB per
    case: the sr3 is read once per property, and each rwo is written to the
    share and then read straight back to be parsed. For 5 properties that is
    roughly

        5 x 90 MB sr3 read + 5 x 64 MB rwo written + 5 x 64 MB rwo read
        ~ 1.1 GB across the network per case

    Doing the work locally leaves only the sr3 copy in and the npy files out:

        1 x 90 MB sr3 read + 5 x ~19-38 MB npy written

    Properties are processed one at a time and each rwo is deleted after it is
    parsed, so local disk use stays around sr3 + one rwo (~155 MB) rather than
    sr3 + all rwo.

    Args
        local_work_dir : where to stage the case. None uses the system temp dir.
                         Point this at a fast local disk (not the mapped drive).
        dtype          : e.g. np.float32 to halve the npy size. CMG writes 4
                         significant digits, so float32 loses nothing real.
        keep_rwo       : keep the local rwo files instead of deleting them
                         (they are still discarded with the temp directory
                         unless local_work_dir is set).

    Returns a dict {property: npy_path}.
    """
    import shutil, tempfile

    sr3_folder = Path(sr3_folder_path)
    src_sr3 = sr3_folder / f"{case_name}.{sim_results_file_format}"
    if not src_sr3.is_file():
        raise FileNotFoundError(f"Case not found: {src_sr3}")

    save_folder_path = Path(save_folder_path)
    save_folder_path.mkdir(parents=True, exist_ok=True)

    tmp_ctx = None
    if local_work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory()
        work = Path(tmp_ctx.name)
    else:
        work = Path(local_work_dir) / case_name
        work.mkdir(parents=True, exist_ok=True)

    written = {}
    try:
        # ---- copy the sr3 in once, and check it arrived whole ----------------
        local_sr3 = work / src_sr3.name
        if show_info:
            print(f"{case_name}: copying {src_sr3.stat().st_size/1e6:.0f} MB sr3 to {work}")
        shutil.copy2(src_sr3, local_sr3)
        if local_sr3.stat().st_size != src_sr3.stat().st_size:
            raise IOError(f"{case_name}: local sr3 is {local_sr3.stat().st_size} bytes, "
                          f"source is {src_sr3.stat().st_size}")

        (work / "rwo").mkdir(exist_ok=True)

        # ---- one property at a time, deleting each rwo once parsed ----------
        for property in property_list:
            generate_CMG_rwd(
                sr3_folder_path = work,
                case_name = case_name,
                property = property,
                sim_results_file_format = sim_results_file_format,
                precision = precision,
            )
            run_CMG_rwd_report(
                rwd_folder_path = work,
                case_name = case_name,
                property = property,
                cmg_version = cmg_version,
                wait_for_output = True,
            )
            (work / f"{case_name}.rwd").unlink(missing_ok=True)

            sim_results = CMG_rwo2npy(
                rwo_folder_path = work / "rwo",
                case_name = case_name,
                property = property,
                is_save = False,
                save_folder_path = work,
                show_info = show_info,
                expected_shape = expected_shape,
                expected_active = expected_active,
                dtype = dtype,
            )

            out = save_folder_path / f"{case_name}_{property}.npy"
            np.save(out, sim_results)
            written[property] = out

            if not keep_rwo:
                (work / "rwo" / f"{case_name}_{property}.rwo").unlink(missing_ok=True)
            del sim_results
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        elif not keep_rwo:
            # local_work_dir was given: clean the staged case but leave the root
            shutil.rmtree(work, ignore_errors=True)

    return written
