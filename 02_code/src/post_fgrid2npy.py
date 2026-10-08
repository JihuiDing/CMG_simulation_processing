#!/usr/bin/env python3
"""
Extract grid coordinates from .fgrid file and save to numpy array.
The .fgrid file contains grid cell definitions with i,j,k indices and corner coordinates.
This script extracts the average x,y,z coordinates for each cell and organizes them into a numpy array.
"""

import numpy as np
import re
import os
from typing import Tuple

def fgrid2npy(
    fgrid_file_path: str,
    save_dir: str = "results/fgrid_coordinates.npy"
    ) -> np.ndarray:
    """
    Extract grid coordinates from .fgrid file and organize into numpy array.
    
    Args:
        fgrid_file_path: Path to the .fgrid file
        save_dir: Output directory
    Returns:
        coordinates_array: numpy array with shape (n_i, n_j, n_k, 3) containing [x_ave, y_ave, z_ave]
    """
    
    if not os.path.exists(fgrid_file_path):
        raise FileNotFoundError(f"File not found: {fgrid_file_path}")
    
    print(f"Parsing the fgrid file: {fgrid_file_path}")
    
    # Initialize variables
    coordinates_data = []
    
    with open(fgrid_file_path, 'r') as file:
        content = file.read()
    
    # Extract grid dimensions
    dims_match = re.search(r"'DIMENS  '\s+3\s+'INTE'\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", content)
    if dims_match:
        n_i, n_j, n_k = map(int, dims_match.groups())
    else:
        raise ValueError("Could not find grid dimensions in .fgrid file")
    
    # Extract map units
    units_match = re.search(r"'MAPUNITS'\s+1\s+'CHAR'\s*\n\s*'([^']+)'", content)
    if units_match:
        print(f"Units: {units_match.group(1)}")
    
    # Split content into COORDS sections
    sections = content.split("'COORDS  '")
    
    print(f"Processing {len(sections)-1} coordinate sections...")
    
    for i, section in enumerate(sections[1:], 1):
        # Extract COORDS values (i, j, k, cell_id)
        coords_match = re.search(r"'INTE'\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", section)
        if coords_match:
            i, j, k, cell_id = map(int, coords_match.groups())
        else:
            continue
        
        # Extract CORNERS values
        corners_match = re.search(r"'CORNERS '\s+24\s+'REAL'\s*\n(.*?)(?='COORDS  '|$)", section, re.DOTALL)
        if corners_match:
            corners_text = corners_match.group(1).strip()
            # Extract all numeric values from corners
            corners_values = re.findall(r'-?\d+\.?\d*E?[+-]?\d*', corners_text)
            corners_values = [float(val) for val in corners_values]
            
            if len(corners_values) >= 24:
                # Calculate averages for x, y, z
                # Corners are organized as: [x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4, x5,y5,z5, x6,y6,z6, x7,y7,z7, x8,y8,z8]
                x_ave = np.mean(corners_values[0::3])  # Every 3rd value starting from 0
                y_ave = np.mean(corners_values[1::3])  # Every 3rd value starting from 1
                z_ave = np.mean(corners_values[2::3])  # Every 3rd value starting from 2
                
                coordinates_data.append({
                    'i': i,
                    'j': j,
                    'k': k,
                    'cell_id': cell_id,
                    'x_ave': x_ave,
                    'y_ave': y_ave,
                    'z_ave': z_ave
                })
    
    print(f"Total cells processed: {len(coordinates_data)}")
    
    # Create the coordinates array with shape (n_i, n_j, n_k, 3)
    coordinates_array = np.zeros((n_i, n_j, n_k, 3))
    
    # Fill the array
    for cell_data in coordinates_data:
        i, j, k = cell_data['i'] - 1, cell_data['j'] - 1, cell_data['k'] - 1  # Convert to 0-based indexing
        
        # Check bounds
        if 0 <= i < n_i and 0 <= j < n_j and 0 <= k < n_k:
            coordinates_array[i, j, k, 0] = cell_data['x_ave']  # x coordinate
            coordinates_array[i, j, k, 1] = cell_data['y_ave']  # y coordinate
            coordinates_array[i, j, k, 2] = cell_data['z_ave']  # z coordinate
        else:
            print(f"Warning: Cell indices ({i+1}, {j+1}, {k+1}) out of bounds for grid ({n_i}, {n_j}, {n_k})")
    
    # Check for missing cells
    total_expected = n_i * n_j * n_k
    cells_filled = np.count_nonzero(np.any(coordinates_array != 0, axis=3))
    print(f"Cells filled: {cells_filled}/{total_expected} ({cells_filled/total_expected*100:.1f}%)")
    
    # Save numpy array
    np.save(save_dir, coordinates_array)
    
    # Print information
    print(f"Coordinates array shape: {coordinates_array.shape}")
    print(f"Coordinates array dtype: {coordinates_array.dtype}")
    print(f"Coordinate Statistics:")
    print(f"  X range: {np.min(coordinates_array[:,:,:,0]):.2f} to {np.max(coordinates_array[:,:,:,0]):.2f}")
    print(f"  Y range: {np.min(coordinates_array[:,:,:,1]):.2f} to {np.max(coordinates_array[:,:,:,1]):.2f}")
    print(f"  Z range: {np.min(coordinates_array[:,:,:,2]):.2f} to {np.max(coordinates_array[:,:,:,2]):.2f}")
    
    print(f"Data saved to {save_dir}")

    return coordinates_array