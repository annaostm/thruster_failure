import pandas as pd
import os
import re

def extract_experiment_key(filename):
    """ Extracts sorting key based on suffix first (a, b, c) then experiment number (1, 2, 3) """
    match = re.search(r"(\d+)([a-zA-Z])_", filename)
    if match:
        experiment_number = int(match.group(1))  # Extract number
        suffix = match.group(2)  # Extract letter
        return (suffix, experiment_number)  # Sorting primarily by suffix, then number
    return ("z", float('inf'))  # Place any non-matching files at the end

def merge_experiment_csvs(input_folder, output_csv):
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Sort by suffix first (a, b, c...) then by experiment number (1, 2, 3...)
    sorted_files = sorted(all_files, key=extract_experiment_key)

    df_list = []
    for file in sorted_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)  # Keep order as is
    combined_df.to_csv(output_csv, index=False)
    print(f"Merged CSV saved as: {output_csv}")

# Usage
input_folder = '/workspace/Test Data 0.5'  # Update with the correct path
output_csv = '/workspace/test_data_0.5_timestamps_wrongfaultcolumn_test_data.csv'
merge_experiment_csvs(input_folder, output_csv)
