import pandas as pd
import numpy as np

def extract_npy_properties(
    fault_table: pd.DataFrame,
    npy_property_file_path: str,
    save_path: str,
    sim_year: list[int],
    n_i: int,
    n_j: int,
    k_layer_start: int,
    k_layer_end: int
) -> pd.DataFrame:
    
    # Load data
    # fault_table = pd.read_csv(fault_table_file_path)
    npy_property = np.load(npy_property_file_path)
    
    # Pre-calculate year indices to avoid repeated .index() calls
    year_indices = {year: idx for idx, year in enumerate(sim_year)}
    
    # Create a mapping for faster lookups
    # Group fault table by i, j, k coordinates for vectorized operations
    fault_coords = fault_table[['i', 'j', 'k']].values
    
    # Vectorized assignment for each simulation year
    for year in sim_year:
        year_idx = year_indices[year]
        fault_table[f'{year}'] = np.nan
        
        # Get all unique k values that exist in the fault table
        unique_k_values = fault_table['k'].unique()
        
        for k in unique_k_values:
            if k_layer_start <= k <= k_layer_end:
                # Find all fault locations for this k value
                k_mask = fault_table['k'] == k
                
                # Get i, j coordinates for this k layer
                k_faults = fault_table[k_mask]
                i_coords = k_faults['i'].values
                j_coords = k_faults['j'].values
                
                # Vectorized extraction of property values
                # Adjust k index for array access
                k_array_idx = k - k_layer_start
                if 0 <= k_array_idx < npy_property.shape[2]:
                    # Adjust coordinates to be 0-indexed for array access
                    # Fault table coordinates are 1-indexed, so subtract 1
                    i_array_coords = i_coords - 1
                    j_array_coords = j_coords - 1
                    
                    # Ensure coordinates are within bounds
                    valid_mask = (i_array_coords >= 0) & (i_array_coords < n_i) & \
                               (j_array_coords >= 0) & (j_array_coords < n_j)
                    
                    if np.any(valid_mask):
                        # Only process valid coordinates
                        valid_i = i_array_coords[valid_mask]
                        valid_j = j_array_coords[valid_mask]
                        
                        property_values = npy_property[valid_i, valid_j, k_array_idx, year_idx]
                        
                        # Assign values back to the fault table, but only for valid coordinates
                        valid_k_mask = k_mask.copy()
                        valid_k_mask[k_mask] = valid_mask
                        fault_table.loc[valid_k_mask, f'{year}'] = property_values
    
    # Save results
    fault_table.to_csv(save_path, index=False)
    
    return fault_table