# Decisions

## 2026-10-08 set up a new repo

Decisions:
- Set up a new repo to combine pre- and post-processing for coupled CMG simulations
- Set up the repo as suggested by Peng Li
- Check parameter sampling

Reasons:
- Simplify processing
- Effectively use AI
- Make sure parameter sampling is robust

Affected files:
- all

Decision owner:
Human

## 2026-10-08 consolidate duplicate source files

Decisions:
- Keep only the file with the consistent `pre_`/`post_` name for each duplicated module; move the others to `02_code/src/delete_src/`
- `latin_hypercube_sampling` pairs property files with samples in sorted order (first n realizations). This is intentional; do not change it

Reasons:
- Avoid divergent copies of the same code

Affected files:
- `02_code/src/pre_parameter_sampling.py`, `pre_generate_dat_files.py`, `post_CMG2npy_robust.py`
- notebooks importing them (see `06_logs/change_log.md`)

Decision owner:
Human

## 2026-10-10 initial temperature from the sampled gradient

Decisions:
- TEMP for every cell (including overburden layers 1-4, underburden 21-23 and inactive cells) = Temp_grad * Z + 11, with Temp_grad sampled per realization in Petrel and Z the cell-centre depth from `JD_geothermal_coor&fault&dip.npy`
- No depth datum offset and no temperature cap
- Implemented as optional step 2 of the pre2 notebook, after the normal extraction (step 1)

Reasons:
- Petrel's F0000 TEMP ignored the sampled `Temp_grad` (fixed 31.5 C/km, capped at 137 C)

Affected files:
- `02_code/src/pre_generate_temperature.py`, `02_code/notebooks/pre2_extract_rels_from_finit_Vienna2.ipynb`

Decision owner:
Human
