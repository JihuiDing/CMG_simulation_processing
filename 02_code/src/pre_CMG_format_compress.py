import numpy as np
from pathlib import Path

def CMG_format_compress(
        array: np.ndarray,
        keyword: str, 
        max_line_length: int = 80,
        show_summary: bool = False,
        save_dir: str = 'results',
        save_name: str = 'compressed'
        ):
    """
    Compress a numpy array to CMG format (N*value).
    
    Args:
        array (numpy.ndarray): Input array to compress
        max_line_length (int): Maximum characters per line
        
    Returns:
        list: List of compressed lines
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    compressed_lines = []
    current_line = ""
    
    # Find consecutive runs of the same value
    i = 0
    while i < len(array):
        value = array[i]
        count = 1
        
        # Count consecutive occurrences of the same value
        while i + count < len(array) and np.isclose(array[i + count], value, rtol=1e-10):
            count += 1
        
        # Create compressed token - format zero values as "0" instead of "0.000000"
        if np.isclose(value, 0.0, rtol=1e-10):
            if count == 1:
                token = "0"
            else:
                token = f"{count}*0"
        else:
            if count == 1:
                # token = f"{value:.6f}"
                token = f"{value}"
            else:
                # token = f"{count}*{value:.6f}"
                token = f"{count}*{value}"
        
        # Add to current line or start new line
        if current_line and len(current_line + " " + token) > max_line_length:
            compressed_lines.append(current_line)
            current_line = token
        else:
            if current_line:
                current_line += " " + token
            else:
                current_line = token
        
        i += count
    
    # Add the last line
    if current_line:
        compressed_lines.append(current_line)
    
    
    # Create output file path
    output_file = save_dir / f"{save_name}_{keyword}.dat"
    
    # # Create header
    # header = "**POR *ALL"
    
    # Write compressed file
    with open(output_file, 'w') as f:
        # f.write(header + '\n')
        for line in compressed_lines:
            f.write(line + '\n')
    
    if show_summary:
        print("\n" + "="*60)
        print(f"COMPRESSED {keyword} SUMMARY")
        print("="*60)
        print(f"Compressed {len(array):,} values into {len(compressed_lines)} lines")
        compression_ratio = len(array) / len(compressed_lines)
        print(f"Compression ratio: {compression_ratio:.1f}:1")
        print(f"Total cells: {len(array):,}")
        print(f"{keyword} - Mean: {np.mean(array):.6f}")
        print(f"{keyword} - Min: {np.min(array):.6f}")
        print(f"{keyword} - Max: {np.max(array):.6f}")
    print(f'Saved compressed {keyword} data to: {output_file}')

    