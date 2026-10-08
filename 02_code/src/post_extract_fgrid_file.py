import pandas as pd
import numpy as np
import re

def extract_fgrid_file(file_path):
    """
    Extract data from mining.txt file:
    - First 4 values below each COORDS section as i, j, k, cell_id
    - Average first 8, second 8, third 8 values under CORNERS as x_ave, y_ave, z_ave
    """
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split content into sections
    sections = content.split("'COORDS  '")
    
    data = []
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract COORDS values (first 4 values after the header)
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
                x_ave = np.mean(corners_values[0::3])
                y_ave = np.mean(corners_values[1::3])
                z_ave = np.mean(corners_values[2::3])
                
                data.append({
                    'i': i,
                    'j': j,
                    'k': k,
                    'cell_id': cell_id,
                    'x_ave': x_ave,
                    'y_ave': y_ave,
                    'z_ave': z_ave
                })
    
    df = pd.DataFrame(data)
    print(f"Total cells processed: {len(df)}")
    # # Save to CSV
    # output_file = 'results/fgrid_extracted.csv'
    # df.to_csv(output_file, index=False)
    # print(f"Data saved to {output_file}")


    return df


