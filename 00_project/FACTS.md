# Facts

## Dataset

- pre-processing
1. When exporting Petrel grid and properties as rescue files, the default cell origin seems to be (I=0, J=max, K). A cell (i, j, k) in Petrel corresponds to (i, j, k).
2. When exporting Petrel grid properties, if setting cell origin to be (I=0, J=0, K), J is actually reversed. Therefore, a cell (i, j, k) in Petrel corresponds to (i, totalj-j+1, k).
3. In Petrel, i goes from left to right (west to east), j goes from top to bottom (north to south). In CMG, i is the same, but j goes from bottom to top (south to north).

## Processing

- None

## Literature

- None

