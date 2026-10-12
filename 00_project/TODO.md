# TODO

## High Priority

- After uploading `pre_sim_dat_files/` (dat files, data_gridfiles, data_properties) to Sherlock, compare checksums with the verified copies on Oak
- Check the .out of case 6 (realization 117, 2 active cells with PORO = 0) for zero-porosity warnings

- Do not open/save sampled-parameter csv files in Excel (it rounds to the displayed digits). Rebuild the master with `merge_petrel_and_sampled_parameters` after any re-sampling

- Re-run parameter sampling / CMG dat generation with the new TEMP files (earlier simulations used the identical F0000 TEMP)

- Set up the repo
- Check whether non-Vienna work (e.g. Sula CCS) used the old (wrong) importance weights from `IS_SH_azi_SH_uniform` and re-weight if needed. The Vienna parameter sets do not use IS weights

## Medium Priority

- Check in Petrel whether AZ, AC, PV, LOOP reach the property modelling (realization similarity does not depend on them)
- Check other files copied over `Z:` (dat files, sr3 results) for NUL blocks

## Low Priority

- `CMG_format_compress` merges values within 1e-8 and writes values < 1e-8 as 0 (`np.isclose` default `atol`); negligible but could pass `atol=0`

- The extraction log records an old output path (`z:\jding\project_OMV_geothermal\...`)
- `IS_SH_azi_SH_uniform` writes `weights` with `fmt='%.4f'`. With ~100 samples the normalised weights are ~0.01, so they keep only 2-3 significant digits
- `02_code/src/pre_sampling.py` is an empty file
