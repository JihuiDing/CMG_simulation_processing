import pandas as pd
import numpy as np

def extract_npy_properties(
    fault_table_file_path: str,
    npy_property_file_path: str,
    sim_year: list[int],
    n_i: int,
    n_j: int,
    k_layer_start: int,
    k_layer_end: int
    ) -> pd.DataFrame:

    fault_table = pd.read_csv(fault_table_file_path)
    npy_property = np.load(npy_property_file_path)
    k_layer = [k for k in range(k_layer_start , k_layer_end + 1)]

    for year in sim_year:
        fault_table[f'{year}'] = np.nan
        for k in k_layer:
            for i in range(n_i):
                for j in range(n_j):
                    fault_table.loc[(fault_table['i'] == i) & (fault_table['j'] == j) & (fault_table['k'] == k), f'{year}'] = npy_property[i,j, k - k_layer_start, sim_year.index(year)]

    fault_table.to_csv(f'{npy_property_file_path.split(".")[0]}_fault_table.csv', index=False)

    return fault_table
    


