import numpy as np
import pandas as pd
import os

def generate_dat_files(
    df_parameters: pd.DataFrame,
    template_file_path: str,
    save_folder_path: str
    ) -> None:

    # 1. Create the output directory if it doesn't exist
    os.makedirs(save_folder_path, exist_ok=True)

    # 2. Read the content of the template file
    try:
        with open(template_file_path, 'r') as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file '{template_file_path}' not found.")
        exit()

    # 3. Loop through each row of the DataFrame to create a new file
    for index, row in df_parameters.iterrows():
        # Start with the original template content for each new file
        new_content = template_content

        # Replace each placeholder with the value from the current row
        for col_name in df_parameters.columns:
            placeholder = f'--{col_name}--'
            new_content = new_content.replace(placeholder, str(row[col_name]))

        # Define the new filename using the 'CASEID' column
        output_filename = f"case{index+1}.dat"
        output_path = os.path.join(save_folder_path, output_filename)

        # Write the new content to the output file
        with open(output_path, 'w') as f:
            f.write(new_content)

    print(f"Generated {len(df_parameters)} dat files successfully.")



