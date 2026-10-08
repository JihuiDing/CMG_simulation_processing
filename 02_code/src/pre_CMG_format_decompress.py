import numpy as np
from pathlib import Path
import re

def CMG_format_decompress(file_path):
    """
    Decompress a CMG format file and return the numerical values as a numpy array.
    
    Args:
        file_path (str or Path): Path to the CMG format file
        
    Returns:
        numpy.ndarray: Array of numerical values
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    values = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('**'):
                continue
            
            # Split the line into tokens
            tokens = line.split()
            
            for token in tokens:
                # Check if token matches pattern N*value
                match = re.match(r'(\d+)\*([+-]?\d*\.?\d*)', token)
                if match:
                    count = int(match.group(1))
                    value = float(match.group(2))
                    values.extend([value] * count)
                else:
                    # If it's just a single number, add it once
                    try:
                        value = float(token)
                        values.append(value)
                    except ValueError:
                        # Skip non-numeric tokens
                        continue
    
    if not values:
        raise ValueError(f"No numerical values found in {file_path}. Check if the file format is correct.")
    
    return np.array(values, dtype=np.float64)
