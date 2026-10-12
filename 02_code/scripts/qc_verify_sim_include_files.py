"""Read-only QC of the include files next to a dat-file folder (data_gridfiles/, data_properties/).

Checks:
1. data_properties/ is byte-identical to processed/pre_properties/ (verified on 2026-10-09/11)
2. data_gridfiles/: no NUL bytes; md5 equals all other copies found under `reference_root`
3. corners: 24*ni*nj*nk values; cell centres equal the CMG-ordered centres in the
   coordinate npy (confirms the CMG i, j order)
4. null_all: ni*nj*nk 0/1 values, active cells = PORO > 0 in every realization
5. pinchoutarray_all: ni*nj*nk 0/1 values, relation to null_all
6. transf_fault*: cells inside the grid, and agreement with the fault ids in the npy
7. well perforations (from the case list) are not null or pinched out

Usage (Anaconda python):
    python 02_code/scripts/qc_verify_sim_include_files.py <output_dir>
"""
import collections
import filecmp
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_root = project_root / '01_data' / 'geothermal_Vienna'

################## Inputs ##############################
sim_root = data_root / 'processed' / 'pre_sim_dat_files'
grid_folder = sim_root / 'data_gridfiles'
property_copy_folder = sim_root / 'data_properties'
property_folder = data_root / 'processed' / 'pre_properties'
case_list_path = sim_root / 'param_clustering_case_list.csv'
coor_path = data_root / 'processed' / 'post_coor_fault' / 'JD_geothermal_coor&fault&dip.npy'
reference_root = Path('Z:/jding/project_OMV_geothermal')   # earlier copies of the grid files
grid_shape = (139, 248, 23)  # (ni, nj, nk)
perf_layers = range(5, 21)
WELLS = ['INJ1', 'INJ2', 'INJ3', 'PRO1', 'PRO2', 'PRO3']
################## End of Inputs #######################


def read_cmg_array(path):
    values = []
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith('**'):
            continue
        for token in line.split():
            if '*' in token:
                count, value = token.split('*')
                values.extend([float(value)] * int(count))
            else:
                values.append(float(token))
    return np.array(values)


def to_ijk(flat, shape):
    ni, nj, nk = shape
    return flat.reshape(nk, nj, ni).transpose(2, 1, 0)  # -> [i, j, k], 0-based


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ni, nj, nk = grid_shape
    n = ni * nj * nk
    problems = []

    # 1. property copies
    names = sorted(p.name for p in property_folder.glob('*.dat'))
    copies = sorted(p.name for p in property_copy_folder.glob('*.dat'))
    _, mismatch, errors = filecmp.cmpfiles(property_folder, property_copy_folder, names, shallow=False)
    print(f'[props] {len(copies)} files in data_properties, {len(names)} in pre_properties; same list: '
          f'{names == copies}; byte-identical: {len(names) - len(mismatch) - len(errors)}/{len(names)}')
    if names != copies or mismatch or errors:
        problems.append(f'property copies differ: {mismatch + errors}')

    # 2. grid files: NUL and md5 vs other copies
    rows = []
    for f in sorted(grid_folder.glob('*.dat')):
        b = f.read_bytes()
        md5 = hashlib.md5(b).hexdigest()
        others = [p for p in reference_root.rglob(f.name) if p.is_file()]
        hashes = collections.Counter(hashlib.md5(p.read_bytes()).hexdigest() for p in others)
        rows.append(dict(file=f.name, n_bytes=len(b), nul_bytes=b.count(b'\x00'), md5=md5,
                         n_other_copies=len(others), n_other_copies_same_md5=hashes[md5],
                         n_distinct_md5_other_copies=len(hashes)))
    df_md5 = pd.DataFrame(rows)
    print(df_md5.to_string(index=False))
    df_md5.to_csv(out_dir / 'verify_gridfiles_md5.csv', index=False)
    if (df_md5['nul_bytes'] > 0).any() or (df_md5['n_other_copies_same_md5'] < df_md5['n_other_copies']).any():
        problems.append('grid file with NUL bytes or differing from other copies')

    coor = np.load(coor_path)

    # 3. corners
    c = read_cmg_array(grid_folder / '260403_JD_geothermal_flow_corners.dat')
    print(f'[corners] {c.size} values (expected 24*N = {24 * n})')
    centres = []
    for axis in range(3):
        a = c[axis * 8 * n:(axis + 1) * 8 * n].reshape(nk, 2, 2 * nj, 2 * ni)  # k, top/bottom, 2nj, 2ni
        a = a.reshape(nk, 2, nj, 2, ni, 2).mean(axis=(1, 3, 5))               # k, j, i
        centres.append(a.transpose(2, 1, 0))
    centres = np.stack(centres, axis=-1)

    # 4. null
    null = to_ijk(read_cmg_array(grid_folder / '260403_JD_geothermal_flow_null_all.dat'), grid_shape)
    active = null == 1
    print(f'[null] {null.size} values (expected {n}), values {np.unique(null)}, active cells {active.sum()}')
    d = np.abs(centres - coor[..., :3])[active]
    print(f'[corners] active cells: max |centre - npy| x={d[:, 0].max():.3g} m, y={d[:, 1].max():.3g} m, '
          f'z={d[:, 2].max():.3g} m')
    # z agrees exactly; x/y differ by up to ~0.2 m (0.5 % of a 50 m cell), as the npy centres are
    # not the plain 8-corner mean. A reversed j would give differences of km
    if d.max() > 1.0:
        problems.append('corner centres differ from coordinate npy')
    poro_files = sorted(pd.read_csv(case_list_path)['PORO_file'].unique())
    zero_active, poro_in_null = [], 0
    for f in poro_files:
        poro = to_ijk(read_cmg_array(property_folder / f), grid_shape)
        poro_in_null += int(np.sum((poro > 0) & ~active))
        zero_active += [f'{f[:17]} ({i + 1},{j + 1},{k + 1})' for i, j, k in np.argwhere((poro <= 0) & active)]
    print(f'[null] over {len(poro_files)} realizations: PORO > 0 in null cells: {poro_in_null}; '
          f'active cells with PORO = 0: {len(zero_active)} {zero_active}')
    if poro_in_null:
        problems.append('PORO > 0 in null cells')

    # 5. pinch-out
    pinch = to_ijk(read_cmg_array(grid_folder / '260403_JD_geothermal_flow_pinchoutarray_all.dat'), grid_shape)
    print(f'[pinch] {pinch.size} values, values {np.unique(pinch)}; flagged 0 = {np.sum(pinch == 0)}, '
          f'of which null {np.sum((pinch == 0) & ~active)}, active {np.sum((pinch == 0) & active)}; '
          f'flagged 1 but null {np.sum((pinch == 1) & ~active)}')

    # 6. fault transmissibility lists
    for k_fault in (0, 1):
        f = grid_folder / f'260720_JD_geothermal_flow_transf_fault{k_fault}.dat'
        entries = re.findall(r'(\d+)\s+(\d+)\s+(\d+)\s+([IJK]DIR[+-])', f.read_text())
        ijk = np.array([[int(a), int(b), int(c)] for a, b, c, _ in entries])
        dirs = collections.Counter(e[3] for e in entries)
        inside = ((ijk >= 1) & (ijk <= np.array(grid_shape))).all()
        fid = coor[ijk[:, 0] - 1, ijk[:, 1] - 1, ijk[:, 2] - 1, 3]
        print(f'[transf] fault{k_fault}: {len(entries)} faces {dict(dirs)}, inside grid {inside}, '
              f'npy fault id at these cells: {dict(collections.Counter(np.where(np.isnan(fid), -1, fid).astype(int)))} '
              f'(-1 = none), cells in active blocks {active[ijk[:, 0] - 1, ijk[:, 1] - 1, ijk[:, 2] - 1].sum()}')
        if not inside:
            problems.append(f'transf_fault{k_fault} outside grid')

    # 7. well perforations
    cases = pd.read_csv(case_list_path)
    lay = cases.drop_duplicates('layout_id')
    bad = []
    for _, r in lay.iterrows():
        for w in WELLS:
            i, j = int(r[f'{w}_I']), int(r[f'{w}_J'])
            for k in perf_layers:
                if not active[i - 1, j - 1, k - 1] or pinch[i - 1, j - 1, k - 1] == 0:
                    bad.append((r['layout_id'], w, i, j, k))
    print(f'[wells] perforated cells null or pinched out: {bad if bad else "none"} '
          f'({len(lay)} layouts x {len(WELLS)} wells x {len(perf_layers)} layers)')
    if bad:
        problems.append('well perforations in null/pinched cells')

    print('\nPROBLEMS:' if problems else '\nNo problems found.', problems if problems else '')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
