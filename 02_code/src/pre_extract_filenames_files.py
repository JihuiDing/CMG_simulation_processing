import numpy as np
import re
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_sorted_filenames(
        folder_dir=None,
        is_save=False,
        save_dir=None,
        save_name='filenames.csv',
        show_filenames: bool = False,
):
    """
    Extract file names from a folder and sort them numerically.
    Returns a list of sorted file names (excluding hidden files like .DS_Store).
    """
    
    # Create the save directory if it doesn't exist
    if is_save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Get all files, excluding hidden files
    filenames = [
        file_path.name for file_path in folder_dir.iterdir()
        if file_path.is_file() and not file_path.name.startswith('.')
    ]

    # Sort numerically
    filenames = sorted(filenames, key=extract_number)
    
    if is_save:
        np.savetxt(save_dir/f'{save_name}', np.array(filenames), delimiter=',', fmt='%s')

    if show_filenames:
        print("Extracted and sorted filenames:")
        for fname in filenames:
            print(fname)

    return filenames


def extract_sorted_files(
        folder_dir=None,
        save_dir=None,
        file_extension=('.FINIT',),
        show_summary: bool = True,
        show_filenames: bool = False
):
    """
    Extract files of one or more extensions from a folder and sort them numerically.

    file_extension: a single extension string (e.g. '.FINIT') or an iterable of
                    extensions (e.g. ['.FINIT', '.F0000']).
    """

    # Normalize to an uppercase set for matching
    if isinstance(file_extension, str):
        file_extension = (file_extension,)
    extensions = {ext.upper() for ext in file_extension}

    # Create destination directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    # Iterate through each simulation case folder
    for case_path in tqdm(folder_dir.iterdir(), desc='Extracting files'):
        if case_path.is_dir():
            for src_file in case_path.iterdir():
                if src_file.is_file() and src_file.suffix.upper() in extensions:
                    dst_file = save_dir / src_file.name
                    shutil.copy2(src_file, dst_file)
                    filenames.append(src_file.name)

    # Sort numerically by extracted number
    filenames = sorted(filenames, key=extract_number)

    if show_summary:
        print(f"Total files copied ({', '.join(sorted(extensions))}): {len(filenames)}")

    if show_filenames:
        print("Extracted and sorted filenames:")
        for fname in filenames:
            print(fname)


# Function to extract number inside filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')