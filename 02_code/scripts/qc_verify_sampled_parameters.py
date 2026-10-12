"""Read-only QC of a sampled-parameter master csv (Petrel parameters + CMG LHS parameters).

Checks:
1. rows: realization name matches the PORO/PERMX/TEMP file names, no duplicates,
   every Complete Petrel realization present
2. Petrel columns equal the Petrel ensemble csv
3. CMG columns equal the `latin_hypercube_sampling` output csv, and that csv is
   reproduced exactly by re-running the sampler with the given bounds/seed
4. LHS: values within bounds, one sample per 1/n stratum in each dimension,
   pairwise correlations (CMG-CMG, Petrel-CMG)
5. derived stress columns recomputed from the master values; |Sv| > |SH| > |Sh|
6. Petrel ensemble: one sample per 1/n stratum per parameter; failed realizations

Usage (Anaconda python, needs scipy):
    python 02_code/scripts/qc_verify_sampled_parameters.py <output_dir>
"""
import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / '02_code' / 'src'))
from pre_parameter_sampling import latin_hypercube_sampling

data_root = project_root / '01_data' / 'geothermal_Vienna'

################## Inputs ##############################
master_path = data_root / 'processed' / 'pre_parameter_sampling' / '260904_sampled_parameters_master.csv'
lhs_path = data_root / 'processed' / 'pre_parameter_sampling' / 'master_sampled_params_seed11.csv'
petrel_path = data_root / 'raw' / 'from_Petrel' / '260904_realization_sampled_parameters.csv'
file_names_path = data_root / 'processed' / 'pre_property_file_names' / '260904_realization_property_filenames.csv'
# same inputs as cell 4 of pre3_generate_sim_files_Vienna_GA.ipynb
random_seed = 11
param_bounds = {
    'KVKH': [0.01, 0.5],
    'PRES_init_kPa': [24000, 25000],
    'FTM0': [0.0, 0.5],
    'FTM1': [0.0, 0.5],
    'HCAP_res_J/m3-C': [2.304e6, 3.117e6],
    'THCOND_res_J/m-day-C': [2.387e5, 2.920e5],
    'THEXPCOEF_1/C': [6.0e-6, 1.1e-5],
    'E_kPa': [30e6, 35e6],
    'PR': [0.2, 0.4],
    'SH_MPa/km': [-8.68 * 1.05, -8.68 * 0.95],
    'Sh_MPa/km': [-5.52 * 1.05, -5.52 * 0.95],
    'Sv_MPa/km': [-12.47 * 1.05, -12.47 * 0.95],
    'SH_azi_deg': [158, 178],
}
ref_block_top_depth, ref_block_bottom_depth = 2365.807, 2383.534
# Petrel ensemble bounds (inferred from the samples: every 1/100 stratum is filled)
petrel_bounds = {
    'Temp_grad': [0.025, 0.035], 'P_std': [0.01, 0.02], 'P_mean': [0.065, 0.135],
    'AC': [0.2, 0.5], 'Zmin': [0.045, 0.085], 'Zmax': [0.085, 0.195],
    'PV': [10, 100], 'PMax': [1000, 5000], 'AZ': [0, 359],
}
################## End of Inputs #######################


def read_petrel_csv(path):
    lines = Path(path).read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith('Parameter,'))
    rows = [line for line in lines[i + 1:] if line.startswith('JD_')]
    return pd.read_csv(io.StringIO('\n'.join([lines[i]] + rows)))


def n_strata(x, lo, hi, n):
    return len(set(np.floor((np.asarray(x) - lo) / (hi - lo) * n).astype(int)))


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(master_path)
    lhs = pd.read_csv(lhs_path)
    petrel = read_petrel_csv(petrel_path)
    n = len(master)
    band = 1.96 / np.sqrt(n)
    rows = []  # per-column summary

    # 1. rows
    num = lambda s: int(re.search(r'(\d+)', s).group(1))
    for c in ['PORO_file', 'PERMX_file', 'TEMP_file']:
        print(f'[rows] {c}: rows with a different realization number = '
              f'{(master["Parameter"].map(num) != master[c].map(num)).sum()}')
    complete = set(petrel.loc[petrel['RESULT(string)'] == 'Complete', 'Parameter'])
    print(f'[rows] n={n}, duplicates={master["Parameter"].duplicated().sum()}, '
          f'Complete missing={sorted(complete - set(master["Parameter"]))}, '
          f'non-Complete included={sorted(set(master["Parameter"]) - complete)}')

    # 2. Petrel columns
    p = petrel.set_index('Parameter').loc[master['Parameter']]
    for c in petrel_bounds:
        rel = np.max(np.abs(master[c].values - p[c].values) / np.abs(p[c].values))
        lo, hi = petrel_bounds[c]
        rows.append(dict(column=c, source='Petrel', max_rel_diff=rel, lower=lo, upper=hi,
                         min=master[c].min(), max=master[c].max(), n_unique=master[c].nunique(),
                         strata_filled=n_strata(petrel[c], lo, hi, len(petrel)), n_strata=len(petrel)))

    # 3. CMG columns vs LHS csv, and LHS csv vs re-run
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        res = latin_hypercube_sampling(
            random_seed=random_seed, name_prefix='qc', property_file_names_path=file_names_path,
            output_file_path=tmp, n_samples=n, param_names=list(param_bounds),
            lower_bounds=[b[0] for b in param_bounds.values()],
            upper_bounds=[b[1] for b in param_bounds.values()],
            property_list=['PORO', 'PERMX', 'TEMP'],
            ref_block_top_depth=ref_block_top_depth, ref_block_bottom_depth=ref_block_bottom_depth,
            show_results=False)
        regen = pd.read_csv(Path(tmp) / f'qc_sampled_params_seed{random_seed}.csv')
    print(f'[lhs] {lhs_path.name} reproduced exactly by re-running the sampler: '
          f'{regen.equals(lhs)}')
    full = res['param_dataframe']
    for c in param_bounds:
        lo, hi = sorted(param_bounds[c])
        rel = np.max(np.abs(master[c] - lhs[c]) / np.abs(lhs[c]).clip(1e-300))
        rows.append(dict(column=c, source='LHS', max_rel_diff=rel, lower=lo, upper=hi,
                         min=master[c].min(), max=master[c].max(), n_unique=master[c].nunique(),
                         strata_filled=n_strata(master[c], lo, hi, n), n_strata=n,
                         strata_filled_full_precision=n_strata(full[c], lo, hi, n)))
    for c in [c for c in lhs.columns if lhs[c].dtype == object]:
        print(f'[lhs] {c} identical in master: {(master[c].values == lhs[c].values).all()}')

    # 4. correlations
    X = full[list(param_bounds)].values
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(len(param_bounds), 1)
    print(f'[corr] CMG-CMG: max |r| = {np.abs(C[iu]).max():.3f}; '
          f'{np.sum(np.abs(C[iu]) > band)}/{len(iu[0])} pairs above |r| > {band:.3f} '
          f'(about {0.05 * len(iu[0]):.0f} expected by chance)')
    CC = np.array([[np.corrcoef(master[a], master[b])[0, 1] for b in param_bounds] for a in petrel_bounds])
    print(f'[corr] Petrel-CMG: max |r| = {np.abs(CC).max():.3f}; '
          f'{np.sum(np.abs(CC) > band)}/{CC.size} pairs above {band:.3f} '
          f'(about {0.05 * CC.size:.0f} expected by chance)')

    # 5. derived columns
    beta = master['SH_azi_deg'] - 90
    c2, s2 = np.cos(np.radians(2 * beta)), np.sin(np.radians(2 * beta))
    SH, Sh, Sv = master['SH_MPa/km'], master['Sh_MPa/km'], master['Sv_MPa/km']
    sx = (SH + Sh) / 2 + (SH - Sh) / 2 * c2
    sy = (SH + Sh) / 2 - (SH - Sh) / 2 * c2
    txy = -(SH - Sh) / 2 * s2
    z = (ref_block_top_depth + ref_block_bottom_depth) / 2
    derived = {'beta': beta, 'cos_2beta': c2, 'sin_2beta': s2, 'sigma_x_grad': sx, 'sigma_y_grad': sy,
               'tau_xy_grad': txy, 'sigma_x_ref': -sx * z, 'sigma_y_ref': -sy * z,
               'sigma_z_ref': -Sv * z, 'tau_xy_ref': -txy * z}
    err = max(np.max(np.abs(master[k] - v) / np.abs(v).clip(1e-12)) for k, v in derived.items())
    print(f'[derived] max rel err of recomputed stress columns = {err:.1e}')
    print(f'[derived] |Sv| > |SH| > |Sh| in all rows: '
          f'{((Sv.abs() > SH.abs()) & (SH.abs() > Sh.abs())).all()}')

    # 6. failed Petrel realizations
    failed = petrel.loc[petrel['RESULT(string)'] != 'Complete', 'Parameter'].tolist()
    print(f'[petrel] {len(petrel)} realizations, not Complete: {failed}; LOOP dropped from master')

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'verify_sampled_parameters_master.csv', index=False, float_format='%.6g')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
