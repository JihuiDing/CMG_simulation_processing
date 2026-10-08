#!/usr/bin/env python3
"""
Extract pressure data from .rwo file and organize into numpy array.
The .rwo file contains reservoir simulation results with pressure for i,j,k cells at different times.
"""

import numpy as np
import re
from typing import Tuple, List
import os
from pathlib import Path


def generate_CMG_rwd(
    sr3_folder_path: str = None,
    case_name: str = None,
    property: str = 'PRES',
    sim_results_file_format: str = 'sr3',
    precision: int = 4
    ):
    """
    Write a rwd file for a CMG simulation result sr3 file.

    Args:
        sr3_folder_path: Path to the sr3 folder
        case_name: Name of the case, should be the same as the sr3 file name
        property: Property to extract, should be one of the properties in the sr3 file
        precision: Precision of the output

    Returns:
        None
    """
    # print(f"{case_name}: writing rwd file ...")
    # check if the sr3 folder exists
    sr3_folder = Path(sr3_folder_path)
    if not sr3_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {sr3_folder}")

    if sim_results_file_format not in ('sr3', 'gmch.sr3'):
        raise ValueError("Use correct file format: sr3 or gmch.sr3")
    case_file = sr3_folder / f"{case_name}.{sim_results_file_format}" 

    if not case_file.is_file():
        raise FileNotFoundError(f"Case not found: {case_file}")

    # create a new folder for the rwo files
    rwo_folder = sr3_folder / "rwo"
    rwo_folder.mkdir(parents=True, exist_ok=True)

    # create a new rwd file
    rwd_file = sr3_folder / f"{case_name}.rwd"

    # write the rwd file
    with open(rwd_file, 'w') as f:
        f.write(f"*FILES \t '{case_name}.{sim_results_file_format}' \n")  
        f.write(f"*PRECISION \t {precision} \n")
        f.write(f"*OUTPUT \t 'rwo\\{case_name}_{property}.rwo' \n")
        f.write(f"*PROPERTY-FOR \t '{property}' \t *ALL-TIMES \n")


def run_CMG_rwd_report(
    rwd_folder_path: str,
    case_name: str,
    cmg_version: str = 'ese-ts2win-v2024.20',
    ):
    """
    Run the rwd report.
    """
    # print(f"{case_name}: generating rwo file ...")

    # check if the rwd folder exists
    rwd_folder = Path(rwd_folder_path)
    if not rwd_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {rwd_folder}")
    # check if the case name exists in the rwd folder
    rwd_file = rwd_folder / f"{case_name}.rwd"
    if not rwd_file.is_file():
        raise FileNotFoundError(f"rwd file not found: {rwd_file}")

    if cmg_version == 'ese-ts2win-v2024.20':
        exe_path='"C:\\Program Files\\CMG\\RESULTS\\2024.20\\Win_x64\\exe\\Report.exe"'
        cd_path = rwd_folder
    else:
        print(f'The CMG version {cmg_version} is not implemented yet .....')
        
    # Execute the CMG Results Report on the rwd file to generate the rwo file
    cmd_line = f"cd {cd_path}  & {exe_path} -f {case_name}.rwd -o {case_name}" # 250821 JD: add output file name otherwise it keeps waiting.
    try:
        os.system(cmd_line)
    except:
        raise ValueError(f'{case_name} run rwd step encounters an error ...')



def CMG_rwo2npy(
    rwo_folder_path: str,
    case_name: str,
    property: str = 'PRES',
    is_save: bool = False,
    save_folder_path: str = "results",
    show_info: bool = False
    ):
    """
    Parse the .rwo file and extract pressure data.
    
    Args:
        rwo_folder_path: Path to the .rwo file
        case_name: Name of the case
        property: Property to extract
        save_folder_path: Path to save the numpy array
        
    Returns:
        pressure_array: numpy array with shape (n_i, n_j, n_k, n_time)
    """
    # print(f"{case_name}: converting rwo file to numpy array ...")

    rwo_file_path = os.path.join(rwo_folder_path, f"{case_name}_{property}.rwo")
    if not os.path.exists(rwo_file_path):
        raise FileNotFoundError(f"File not found: {rwo_file_path}")

    # check if the save folder exists
    save_folder_path = Path(save_folder_path)
    save_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists to store data
    time_values = []
    time_dates = []
    pressure_data = []
    
    current_time = None
    current_k = None
    current_j = None
    current_data = []
    
    with open(rwo_file_path, 'r') as file:
        for line_num, line in enumerate(file):
            line = line.strip()
            
            # Check for time step
            if line.startswith("**  TIME ="):
                # Save previous time step data if exists
                if current_time is not None and current_data:
                    pressure_data.append({
                        'time': current_time,
                        'date': time_dates[-1] if time_dates else '',
                        'data': current_data.copy()
                    })
                
                # Parse new time step
                match = re.match(r'\*\*  TIME = (\d+(?:\.\d+)?)\s+(.+)', line)
                if match:
                    current_time = float(match.group(1))
                    current_date = match.group(2)
                    time_values.append(current_time)
                    time_dates.append(current_date)
                    current_data = []
                    if show_info:
                        print(f"Processing time step: {current_time} ({current_date})")
                
            # Check for K, J header
            elif line.startswith("** K ="):
                match = re.match(r'\*\* K = (\d+), J = (\d+)', line)
                if match:
                    current_k = int(match.group(1))
                    current_j = int(match.group(2))
                    current_data.append({
                        'k': current_k,
                        'j': current_j,
                        'values': []
                    })
            
            # Parse pressure values (skip empty lines and headers)
            elif line and not line.startswith("**") and not line.startswith("RESULTS") and not line.startswith(property):
                # Split line and convert to float
                try:
                    values = [float(x) for x in line.split()]
                    if current_data and 'values' in current_data[-1]:
                        current_data[-1]['values'].extend(values)
                except ValueError:
                    # Skip lines that can't be parsed as numbers
                    continue
    
    # Add the last time step
    if current_time is not None and current_data:
        pressure_data.append({
            'time': current_time,
            'date': time_dates[-1] if time_dates else '',
            'data': current_data
        })
    
    if show_info:
        print(f"Found {len(time_values)} time steps")
    
    # Determine grid dimensions
    k_values = set()
    j_values = set()
    i_count = 0
    
    for time_data in pressure_data:
        for cell_data in time_data['data']:
            k_values.add(cell_data['k'])
            j_values.add(cell_data['j'])
            if i_count == 0:
                i_count = len(cell_data['values'])
    
    n_k = max(k_values)
    n_j = max(j_values)
    n_i = i_count
    n_time = len(time_values)
    
    if show_info:
        print(f"Grid dimensions: I={n_i}, J={n_j}, K={n_k}, Time={n_time}")
    
    # Create the pressure array
    sim_results = np.zeros((n_i, n_j, n_k, n_time))
    
    # Fill the array
    for time_idx, time_data in enumerate(pressure_data):
        for cell_data in time_data['data']:
            k = cell_data['k'] - 1  # Convert to 0-based indexing
            j = cell_data['j'] - 1  # Convert to 0-based indexing
            
            if len(cell_data['values']) == n_i:
                sim_results[:, j, k, time_idx] = cell_data['values']
            else:
                print(f"Warning: Expected {n_i} values, got {len(cell_data['values'])} for K={k+1}, J={j+1}, Time={time_data['time']}")

    # Save numpy array
    if is_save:
        save_file_path = save_folder_path / f"{case_name}_{property}.npy"
        np.save(save_file_path, sim_results)

    return sim_results




