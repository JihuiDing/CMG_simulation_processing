import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Union

# section header of an Eclipse formatted file, e.g.  'PORO    '      401741 'REAL'
HEADER_PATTERN = re.compile(r"^\s*'([^']{1,8})'\s+(\d+)\s+'(\w{4})'\s*$")


def check_file_integrity(file_path: Union[str, Path]) -> bytes:
    """
    Read a file as bytes and raise ValueError if it contains NUL bytes.

    Copies made over the Z: (Oak SMB) mount have been found with 1 MiB blocks
    silently replaced by NUL bytes (file size unchanged). An ASCII FINIT/F0000
    file never contains NUL bytes, so any NUL byte means the file is corrupted.

    Returns:
        bytes: the file content, so the caller does not need to read it again
    """
    data = Path(file_path).read_bytes()
    first_nul = data.find(b"\x00")
    if first_nul != -1:
        n_nul = data.count(b"\x00")
        raise ValueError(
            f"Corrupted file: {n_nul:,} NUL bytes, first at byte {first_nul:,} "
            f"({first_nul / 2**20:.2f} MiB). Use a clean copy of {Path(file_path).name}"
        )
    return data


def extract_properties_from_finit(
    finit_file_path: str,
    keywords: List[str],
    is_save: bool = False,
    save_dir: Union[str, Path] = 'results',
    save_name: str = 'extracted',
    show_summary: bool = False,
    check_integrity: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract property arrays from a FINIT (or F0000) file and save them as numpy files.

    Every requested keyword must appear exactly once, and the number of values read
    must equal the size declared in its section header; otherwise a ValueError is raised.

    Args:
        finit_file_path (str): Path to the FINIT file
        keywords (List[str]): List of property keywords to extract (e.g., ['PORO', 'PERMX'])
        is_save (bool): Whether to save the numpy files (default: False)
        save_dir (Union[str, Path]): Directory to save the numpy files (default: 'results')
        show_summary (bool): Whether to show the summary of the extracted properties (default: False)
        check_integrity (bool): Whether to raise on NUL bytes, i.e. a file corrupted during copy (default: True)
    Returns:
        Dict[str, np.ndarray]: Dictionary containing the extracted property arrays
    """
    finit_file_path = Path(finit_file_path)

    # Create save directory if it doesn't exist
    if is_save:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    # Read the file, checking for corruption first
    if check_integrity:
        data = check_file_integrity(finit_file_path)
    else:
        data = finit_file_path.read_bytes()
    lines = data.decode('ascii', errors='replace').splitlines()

    # Collect the data lines of each requested section; a section ends at the next header
    sections = {}  # key -> (declared size, header line number, data lines)
    current_key = None
    for line_no, line in enumerate(lines, 1):
        match = HEADER_PATTERN.match(line)
        if match:
            name, size, dtype = match.group(1).strip(), int(match.group(2)), match.group(3)
            current_key = name if name in keywords else None
            if current_key:
                if current_key in sections:
                    raise ValueError(f"Property {current_key} appears more than once in {finit_file_path.name}")
                sections[current_key] = (size, line_no, [])
                if show_summary:
                    print(f"Found {name} section: size={size}, type={dtype}")
        elif current_key:
            sections[current_key][2].append(line)

    # Convert sections to numpy arrays, check their sizes and save
    extracted_property_dict = {}
    for key in keywords:
        if key not in sections:
            raise ValueError(f"Property {key} not found in {finit_file_path.name}")
        declared, header_line, data_lines = sections[key]

        # DOUB arrays use a 'D' exponent, which float() does not accept
        tokens = ' '.join(data_lines).replace('D', 'E').split()
        try:
            arr = np.array(tokens, dtype=np.float64)
        except ValueError:
            bad = next(t for t in tokens if not _is_number(t))
            raise ValueError(f"Non-numeric value {bad[:20]!r} in {key} section (header at line {header_line}) of {finit_file_path.name}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"NaN or Inf in {key} section (header at line {header_line}) of {finit_file_path.name}")

        if len(arr) != declared:
            raise ValueError(f"Size mismatch in {finit_file_path.name}: {key} header declares {declared} values but {len(arr)} were read")

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


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False
