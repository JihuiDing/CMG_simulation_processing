import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Union

def extract_properties_from_finit(
    finit_file_path: str,
    keywords: List[str],
    is_save: bool = False,
    save_dir: Union[str, Path] = 'results',
    save_name: str = 'extracted',
    show_summary: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract property arrays from a FINIT file and save them as numpy files.
    
    Args:
        finit_file_path (str): Path to the FINIT file
        keywords (List[str]): List of property keywords to extract (e.g., ['PORO', 'PERMX'])
        is_save (bool): Whether to save the numpy files (default: False)
        save_dir (Union[str, Path]): Directory to save the numpy files (default: 'results')
        show_summary (bool): Whether to show the summary of the extracted properties (default: False)
    Returns:
        Dict[str, np.ndarray]: Dictionary containing the extracted property arrays
    """
    # Create save directory if it doesn't exist
    if is_save:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # Initialize storage for property arrays
    sections = {key: [] for key in keywords}
    current_key = None
    
    # Read the FINIT file
    with open(finit_file_path, 'r') as file:
        for line in file:
            stripped = line.strip()
            
            # Check for section header
            for key in keywords:
                if f"'{key}" in stripped:
                    current_key = key
                    # Extract size and type from the header line
                    match = re.search(r"'(\w+)'\s+(\d+)\s+'(\w+)'", stripped)
                    if match:
                        keyword, size, dtype = match.groups()
                        print(f"Found {keyword} section: size={size}, type={dtype}")
                    break
            else:
                if current_key:
                    # Split the line and process each part
                    parts = stripped.split()
                    for part in parts:
                        # Stop if we encounter a non-numerical part
                        if not re.match(r'^[-+]?\d*\.?\d+([eE][-+]?\d+)?$', part):
                            current_key = None
                            break
                        try:
                            value = float(part)
                            sections[current_key].append(value)
                        except ValueError:
                            current_key = None
                            break
    
    # Convert sections to numpy arrays and save
    extracted_property_dict = {}
    for key, values in sections.items():
        if values:
            # Convert to numpy array
            arr = np.array(values)
            if show_summary:
                print(f"\n{key} statistics:")
                print(f"Number of values: {len(arr)}")
                print(f"Min: {arr.min():.4e}")
                print(f"Max: {arr.max():.4e}")
                print(f"Mean: {arr.mean():.4e}")
            
            if is_save:
                # Save to numpy file
                output_file = save_dir / f"{save_name}_{key.lower()}.npy"
                np.save(output_file, arr)
                if show_summary:
                    print(f"Saved to {output_file}")
            
            # Store in results dictionary
            extracted_property_dict[key] = arr
    
    return extracted_property_dict
