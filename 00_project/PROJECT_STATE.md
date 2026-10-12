# Project State

## Objective

Pre- and post-processing for CMG coupled flow-geomechanical simulations

## Current Status

- check parameter sampling: importance-sampling weight bug fixed (2026-10-08)
- duplicate source files in `02_code/src/` consolidated (2026-10-08)
- 25 skipped FINIT extractions diagnosed: file corruption from copying over `Z:`, not a parser bug (2026-10-09)

## Completed

- Set up the repo
- Fixed the proposal pdf in `IS_SH_azi_SH_uniform` (`02_code/src/pre_parameter_sampling.py`); regression test in `02_code/tests/test_pre_parameter_sampling.py`
- Moved duplicate src files to `02_code/src/delete_src/` and pointed notebook imports at the kept files (see `06_logs/change_log.md`)
- Integrity scan of all FINIT/F0000 files (`03_results/2026-10-09_finit_integrity/`)
- TEMP for all 96 realizations regenerated from the sampled `Temp_grad` (TEMP = Temp_grad * Z + 11, every cell) and verified (`03_results/2026-10-11_temperature_verification/`) (2026-10-10/11)
- Re-extracted the 25 skipped realizations (human ran the notebook). All 96 realizations / 288 property files verified against raw data (`03_results/2026-10-09_property_verification/`)
- Checked the new parameter master `260904_sampled_parameters_master.csv`: sampling correct; the Excel-saved master had 3 columns rounded to 3 significant digits, so it was rebuilt with `merge_petrel_and_sampled_parameters` and passes the QC (`03_results/2026-10-12_parameter_sampling_check/`) (2026-10-12)
- Checked the 192 `param_clustering` dat files (2 GA layouts × 96 parameter sets): all correct (`03_results/2026-10-12_dat_file_check/`) (2026-10-12)
- FINIT parser hardened (NUL check, declared-size check, exact keyword match); pre2 notebook (`..._Vienna2.ipynb`) no longer copies files and reads the clean in-repo `raw/pre_Petrel_realizations/` directly (2026-10-09)

## Current Work

- check property extraction

## Current Findings

1. Before the fix, the importance weights from `IS_SH_azi_SH_uniform` were wrong:
   - the lower-component proposal density had an operator-precedence error, giving `(1-alpha)/320 - 300`, which is negative;
   - both components divided by the full target width instead of their own width.
   With alpha = 0.9 and proposal_SH_azi_low = 319, the weighted mean SH azimuth was about 319.5 deg instead of the target mean of 310 deg. Any earlier results that used these weights (via `latin_hypercube_and_importance_sampling`) are biased. The sampled parameter values themselves were not affected, only the `q_SH_azi`, `q_SH` and `weights` columns. None of the 15 Vienna parameter sets in `01_data/geothermal_Vienna/raw/pre_sampled_parameters/` has `weights` or `q_*` columns, so the Vienna runs did not use importance sampling. The code comments say it was written for Sula CCS, which may be affected.
2. Only 71 of 100 Petrel realizations (112-211) have extracted properties. 4 failed in Petrel (122, 132, 186, 207) and 25 were skipped during FINIT extraction. The cause is **not the parser**: the parsed copies contained 1 MiB zero-filled (NUL) blocks introduced when copying over the `Z:` SMB mount. All 25 are clean in the original source (`Z:/jding/project_OMV_geothermal/github_geothermal/preCMGsim/data/Vienna_geothermal/Petrel_realizations`) and can be re-extracted. Details: `03_results/2026-10-09_finit_integrity/README.md`.
3. The repo copies in `01_data/geothermal_Vienna/raw/pre_Petrel_realizations/` were corrupted (53 of 96). On 2026-10-09 the human recopied them on Sherlock with `rsync -avc`; a rescan shows all 96 clean (`03_results/2026-10-09_finit_integrity/scan_repo_raw_after_sherlock_recopy.csv`). The 213 extracted property files and `raw/from_Petrel/` are clean.

## Unresolved Questions

- Initial temperature (F0000 TEMP) was identical in all realizations: Petrel used a fixed 31.5 C/km instead of the sampled `Temp_grad`. Replaced by `pre_generate_temperature.py` (2026-10-10). Any simulations already run with the old TEMP did not include temperature-gradient uncertainty

- Did any non-Vienna work (e.g. Sula CCS) use the old importance weights and need re-weighting?
- Which other files copied over `Z:` may carry NUL blocks (e.g. dat files sent to Sherlock, sr3 results)?

## Next Actions


## Last Updated

2026-10-12

## Last Agent

Claude
