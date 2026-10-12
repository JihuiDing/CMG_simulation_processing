# Facts

## Dataset

- pre-processing
1. When exporting Petrel grid and properties as rescue files, the default cell origin seems to be (I=0, J=max, K). A cell (i, j, k) in Petrel corresponds to (i, j, k).
2. When exporting Petrel grid properties, if setting cell origin to be (I=0, J=0, K), J is actually reversed. Therefore, a cell (i, j, k) in Petrel corresponds to (i, totalj-j+1, k).
3. In Petrel, i goes from left to right (west to east), j goes from top to bottom (north to south). In CMG, i is the same, but j goes from bottom to top (south to north).
- when generating geological realizations using a Petrel workflow, there are often times failed realizations
- CMG coupled simulations may have crashed runs
- Copying large files over the `Z:` (Oak SMB) mount can silently replace 1 MiB blocks, at MiB-aligned offsets, with NUL bytes. The file size is unchanged, and each copy is damaged in different places. Observed in FINIT/F0000 copies made Oak-to-Oak from a Stanford Windows computer over SMB: in Sep 2026 by `shutil.copy2` in the pre2 notebook, and on 2026-10-08 by a manual copy from the Petrel sim folder to the repo. Damage was stable across re-reads a day apart. Evidence: `03_results/2026-10-09_finit_integrity/README.md`

## Processing

- None

## Literature

- None

