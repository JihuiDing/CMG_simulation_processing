"""Read-only QC of a set of generated CMG dat files (template + case list).

Checks:
1. case list = every well layout x every sampled parameter set, case numbers 1..n
2. every dat file matches the template exactly outside the placeholders; the value at
   each placeholder equals the case list (and the sampled-parameter csv); no '--X--' left
3. referenced property files exist in processed/pre_properties and belong to one realization
4. wells: inside the grid, distinct columns, perforated cells active (PORO > 0) in every
   realization used, fault cells hit, well spacing
5. reference block depth (GEORBLOCK) vs the top/bottom depths used for the stress state

Usage (Anaconda python):
    python 02_code/scripts/qc_verify_dat_files.py <output_dir>
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_root = project_root / '01_data' / 'geothermal_Vienna'

################## Inputs ##############################
name_prefix = 'param_clustering'
dat_folder = data_root / 'processed' / 'pre_sim_dat_files' / f'{name_prefix}_dat_files'
case_list_path = data_root / 'processed' / 'pre_sim_dat_files' / f'{name_prefix}_case_list.csv'
template_path = data_root / 'raw' / 'pre_dat_file_templates' / 'JD_geothermal_base_model_GA.dat'
layout_path = data_root / 'raw' / 'pre_well_layouts' / 'selected_layouts_to_simulate.csv'
sampled_path = data_root / 'processed' / 'pre_parameter_sampling' / 'master_sampled_params_seed11.csv'
property_folder = data_root / 'processed' / 'pre_properties'
coor_path = data_root / 'processed' / 'post_coor_fault' / 'JD_geothermal_coor&fault&dip.npy'
grid_shape = (139, 248, 23)            # (ni, nj, nk), CMG order
ref_block_top_depth, ref_block_bottom_depth = 2365.807, 2383.534
################## End of Inputs #######################

WELLS = ['INJ1', 'INJ2', 'INJ3', 'PRO1', 'PRO2', 'PRO3']


def read_cmg_array(path):
    """Values of a CMG include file, expanding N*value (also with exponents)."""
    values = []
    for token in Path(path).read_text().split():
        if token.startswith('**'):
            continue
        if '*' in token:
            count, value = token.split('*')
            values.extend([float(value)] * int(count))
        else:
            values.append(float(token))
    return np.array(values)


def template_regex(template):
    """Regex of the template with one capture group per placeholder occurrence."""
    parts = re.split(r'(--[^\s-][^\s]*?--)', template)
    names, pattern = [], ''
    for k, part in enumerate(parts):
        if k % 2:
            names.append(part[2:-2])
            pattern += r'(\S+?)'
        else:
            pattern += re.escape(part)
    return re.compile(pattern, re.S), names


def same_value(text, expected):
    if isinstance(expected, str):
        return text == expected
    try:
        return np.isclose(float(text), float(expected), rtol=1e-12, atol=0)
    except ValueError:
        return False


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = []

    # 1. case list
    cases = pd.read_csv(case_list_path)
    layouts = pd.read_csv(layout_path)
    layouts.columns = [c if c == 'layout_id' else c.upper() for c in layouts.columns]
    sampled = pd.read_csv(sampled_path)
    expected = layouts.merge(sampled.assign(param_id=range(1, len(sampled) + 1)), how='cross')
    expected.insert(0, 'case', range(1, len(expected) + 1))
    expected = expected[['case', 'layout_id'] + list(layouts.columns[1:]) + ['param_id'] + list(sampled.columns)]
    ok = list(cases.columns) == list(expected.columns) and len(cases) == len(expected)
    if ok:
        for c in cases.columns:
            if cases[c].dtype == object:
                ok &= (cases[c].values == expected[c].values).all()
            else:
                ok &= np.allclose(cases[c].values, expected[c].values, rtol=1e-12, atol=0)
    print(f'[cases] {len(layouts)} layouts x {len(sampled)} parameter sets = {len(cases)} cases; '
          f'case list = cross join of layouts and {sampled_path.name}: {ok}')
    if not ok:
        problems.append('case list differs from layouts x sampled parameters')

    # 2. dat files vs template
    template = template_path.read_text()
    regex, names = template_regex(template)
    missing_cols = sorted(set(names) - set(cases.columns))
    print(f'[template] {len(names)} placeholder occurrences, {len(set(names))} distinct; '
          f'not in case list: {missing_cols}')
    unused = sorted(set(cases.columns) - set(names))
    print(f'[template] case-list columns not used by the template: {unused}')
    dat_files = sorted(dat_folder.glob('case*.dat'), key=lambda p: int(re.search(r'(\d+)', p.name).group(1)))
    print(f'[dat] {len(dat_files)} files in {dat_folder.name}; names case1..case{len(cases)}: '
          f'{[p.name for p in dat_files] == [f"case{c}.dat" for c in cases["case"]]}')
    rows = []
    for _, case in cases.iterrows():
        path = dat_folder / f'case{case["case"]}.dat'
        raw = path.read_bytes()
        text = path.read_text()
        m = regex.fullmatch(text)
        bad = []
        if m is None:
            bad.append('text outside placeholders differs from template')
        else:
            for name, value in zip(names, m.groups()):
                if not same_value(value, case[name]):
                    bad.append(f'{name}={value} (expected {case[name]})')
        if re.search(r'--[A-Za-z][^\s]*--', text):
            bad.append('placeholder left')
        if b'\x00' in raw or re.search(r'\bnan\b', text, re.I):
            bad.append('NUL or nan')
        rows.append(dict(case=case['case'], file=path.name, n_bytes=len(raw), ok=not bad, issues='; '.join(bad)))
    df_dat = pd.DataFrame(rows)
    print(f'[dat] files matching template + case list: {df_dat["ok"].sum()}/{len(df_dat)}')
    for _, r in df_dat[~df_dat['ok']].head(10).iterrows():
        print(f'   {r["file"]}: {r["issues"][:300]}')
    if not df_dat['ok'].all():
        problems.append(f'{(~df_dat["ok"]).sum()} dat files do not match')
    df_dat.to_csv(out_dir / 'verify_dat_files.csv', index=False)

    # 3. property files
    for c in ['PORO_file', 'PERMX_file', 'TEMP_file']:
        absent = [f for f in cases[c].unique() if not (property_folder / f).exists()]
        print(f'[props] {c}: {cases[c].nunique()} distinct, missing in pre_properties: {absent}')
    one_real = (cases['PORO_file'].str[:-9] == cases['PERMX_file'].str[:-10]).all() and \
               (cases['PORO_file'].str[:-9] == cases['TEMP_file'].str[:-9]).all()
    print(f'[props] PORO/PERMX/TEMP of each case from the same realization: {one_real}')

    # 4. wells
    ni, nj, nk = grid_shape
    perf_k = sorted({int(k) for k in re.findall(r'--PRO1_I-- --PRO1_J-- (\d+)', template)})
    print(f'[wells] perforated layers: {perf_k[0]}-{perf_k[-1]} ({len(perf_k)} cells per well)')
    coor = np.load(coor_path)
    poro_files = sorted(cases['PORO_file'].unique())
    poro = np.stack([read_cmg_array(property_folder / f) for f in poro_files])  # (n_real, ni*nj*nk)
    well_rows = []
    for _, lay in layouts.iterrows():
        cols = {w: (int(lay[f'{w}_I']), int(lay[f'{w}_J'])) for w in WELLS}
        dup = len(set(cols.values())) < len(cols)
        for w, (i, j) in cols.items():
            inside = 1 <= i <= ni and 1 <= j <= nj
            idx = [(k - 1) * ni * nj + (j - 1) * ni + (i - 1) for k in perf_k]
            p = poro[:, idx]
            fault = coor[i - 1, j - 1, [k - 1 for k in perf_k], 3]
            well_rows.append(dict(
                layout_id=lay['layout_id'], well=w, i=i, j=j, inside_grid=inside,
                x=coor[i - 1, j - 1, perf_k[len(perf_k) // 2] - 1, 0], y=coor[i - 1, j - 1, perf_k[len(perf_k) // 2] - 1, 1],
                top_depth_k5=coor[i - 1, j - 1, perf_k[0] - 1, 2], depth_k20=coor[i - 1, j - 1, perf_k[-1] - 1, 2],
                cells_inactive_any_real=int((p <= 0).any(axis=0).sum()),
                min_poro=p.min(), mean_poro=p.mean(),
                fault_cells=int(np.sum(~np.isnan(fault))), duplicate_column_in_layout=dup))
    df_w = pd.DataFrame(well_rows)
    print(df_w.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    for lid, g in df_w.groupby('layout_id'):
        xy = g.set_index('well')[['x', 'y']]
        d = {(a, b): np.hypot(*(xy.loc[a] - xy.loc[b])) for a in WELLS for b in WELLS if a < b}
        inj_pro = {k: v for k, v in d.items() if k[0].startswith('INJ') and k[1].startswith('PRO')}
        print(f'[wells] layout {lid}: min distance between any wells = {min(d.values()):.0f} m '
              f'{min(d, key=d.get)}; nearest producer to each injector: ' +
              ', '.join(f'{a}->{min((v, b) for (aa, b), v in inj_pro.items() if aa == a)[1]} '
                        f'{min(v for (aa, b), v in inj_pro.items() if aa == a):.0f} m' for a in WELLS[:3]))
    df_w.to_csv(out_dir / 'verify_well_locations.csv', index=False)
    if (df_w['cells_inactive_any_real'] > 0).any() or not df_w['inside_grid'].all():
        problems.append('well perforations outside grid or in inactive cells')

    # 5. reference block
    m = re.search(r'\*GEORBLOCK\s+(\d+)\s+(\d+)\s+(\d+)', template)
    i, j, k = map(int, m.groups())
    z = coor[i - 1, j - 1, k - 1, 2]
    print(f'[ref] GEORBLOCK ({i},{j},{k}) centre depth = {z:.3f} m; (top+bottom)/2 used for STRESS3D = '
          f'{(ref_block_top_depth + ref_block_bottom_depth) / 2:.3f} m; '
          f'PORO in ref block (min over realizations) = {poro[:, (k - 1) * ni * nj + (j - 1) * ni + (i - 1)].min():.4g}')

    print('\nPROBLEMS:' if problems else '\nNo problems found.', problems if problems else '')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
