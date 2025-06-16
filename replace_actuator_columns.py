import pandas as pd

# Load the dataset
file_path = "/workspace/test_data_0.5_timestamps_wrongfaultcolumn_test_data.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Create a new column 'fault_label'
def assign_fault_label(row):
    if row["fault"] == 0:
        return 0  # No fault
    elif row["/actuator_ref_1"] == 1:
        return 1  # Fault in actuator 1
    elif row["/actuator_ref_2"] == 1:
        return 2  # Fault in actuator 2
    elif row["/actuator_ref_3"] == 1:
        return 3  # Fault in actuator 3
    elif row["/actuator_ref_4"] == 1:
        return 4  # Fault in actuator 4
    else:
        return -1  # Should not happen if data is correct

df["fault_label"] = df.apply(assign_fault_label, axis=1)

# Drop old fault columns
df.drop(columns=["fault", "/actuator_ref_1", "/actuator_ref_2", "/actuator_ref_3", "/actuator_ref_4"], inplace=True)

# Save the cleaned dataset
output_file = "/workspace/test_data_0.5_timestamps_rightfaultcolumn_test_data.csv"
df.to_csv(output_file, index=False)

print(f"Processed dataset saved as {output_file}")
