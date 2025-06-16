import pandas as pd
import os

def merge_experiment_csvs(input_folder, output_csv):
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".csv")]
    
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)  # Keep order as is
    combined_df.to_csv(output_csv, index=False)
    print(f"Merged CSV saved as: {output_csv}")

# Usage
input_folder = '/workspace/Thruster Data Correct Yaw/'  # Update with the correct path
output_csv = '/workspace/thruster_data_combined_correct_yaw_no_shuffle.csv'
merge_experiment_csvs(input_folder, output_csv)
