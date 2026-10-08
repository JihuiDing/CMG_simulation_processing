import numpy as np
from pathlib import Path
from typing import Union

def generate_full_properties(
        property_dict: dict,
        property_list: list = ['PORO', 'PERMX'], 
        grid_shape: tuple = (107, 117, 79),
        is_save: bool = True,
        save_dir: Union[str, Path] = 'results',
        save_name: str = 'full',
        show_summary: bool = False,
        reverse_j: bool = False
        ):
    """
    Generate property data files for all cells, filling inactive cells with 0.
    
    Args:
        property_dict (dict): Dictionary of property arrays
        property_list (list): List of properties to generate
        total_cells (int): Total number of cells in the grid
        is_save (bool): Whether to save the full property arrays to a file
        output_file (str): Path to the output property data file
        show_summary (bool): Whether to show summary statistics
        is_j_reversed (bool): Whether to reverse the j-direction (CMG could reverse the j-direction)
        grid_shape (tuple): Shape of the grid
    
    Returns:
        dict: Summary statistics of the generated data
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    total_cells = np.prod(grid_shape) 
    
    # Check if input property arrays exist
    if 'ACTID' not in property_dict:
        raise ValueError("Active cell ID array not found in property_dict")
    
    for key in property_list:
        if key not in property_dict:
            raise ValueError(f"Property {key} not found in property_dict")
    
    # Convert active cell IDs to integers if needed
    if property_dict['ACTID'].dtype != np.int64 and property_dict['ACTID'].dtype != np.int32:
        property_dict['ACTID'] = property_dict['ACTID'].astype(np.int64)
    
    # Verify that the number of active cells matches the number of property values
    for key in property_list:
        if len(property_dict['ACTID']) != len(property_dict[key]):
            raise ValueError(f"Mismatch: {len(property_dict['ACTID'])} active cells but {len(property_dict[key])} {key} values")
    
    # Check that cell IDs are within valid range
    min_id = np.min(property_dict['ACTID'])
    max_id = np.max(property_dict['ACTID'])
    
    if min_id < 1 or max_id > total_cells:
        raise ValueError(f"Cell IDs out of range: {min_id} to {max_id} (should be 1 to {total_cells})")
    
    full_property_dict = {}

    for key in property_list:
        # Create full property array initialized with zeros
        full_property = np.zeros(total_cells, dtype=np.float64)

        # Fill in property values for active cells
        full_property[property_dict['ACTID'] - 1] = property_dict[key]  # Subtract 1 because cell IDs are 1-based
    
        if reverse_j:
            # Petrel grid (i,j,k) corresponds to 3D numpy array(k,i,j)
            # reshape then transpose because Petrel export traverse first along i then along j
            full_property = full_property.reshape(grid_shape[2], grid_shape[1], grid_shape[0])
            full_property = np.transpose(full_property, axes=(0, 2, 1)) 
            full_property = full_property[:, :, ::-1] # reverse j columns
            full_property = np.transpose(full_property, axes=(0, 2, 1)) # transpose because flatten will be along rows
            full_property = full_property.flatten() # flatten back to 1D array

        # Save the full property array
        if is_save:
            np.save(save_dir / f'{save_name}_{key}.npy', full_property)
    
        # Store in dictionary with key as name
        full_property_dict[key] = full_property
    
        # Print summary
        if show_summary:
            print("\n" + "="*60)
            print(f"FULL {key} SUMMARY")
            print("="*60)
            print(f"Total cells: {total_cells:,}")
            print(f"Active cells: {len(property_dict['ACTID']):,}")
            print(f"Active cell IDs data type: {property_dict['ACTID'].dtype}")
            print(f"{key} values data type: {property_dict[key].dtype}")
            print(f"Active {key} - Mean: {np.mean(property_dict[key]):.6f}")
            print(f"Active {key} - Min: {np.min(property_dict[key]):.6f}")
            print(f"Active {key} - Max: {np.max(property_dict[key]):.6f}")
            print(f"Full {key} - Mean: {np.mean(full_property):.6f}")
            print(f"Full {key} - Min: {np.min(full_property):.6f}")
            print(f"Full {key} - Max: {np.max(full_property):.6f}")
            print(f"Saved full {key} data to: {save_dir / f'{save_name}_{key}.npy'}")
    
    return full_property_dict