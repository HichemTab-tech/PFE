import pandas as pd
import json

# Load the dataset
data = pd.read_csv('HomeC.csv', low_memory=False)

# Load the devices from JSON
with open('../devices.json') as f:
    devices = json.load(f)

# Dictionary to store minimum values for each device
min_values = {}

# Calculate the minimum non-zero value for each device
for device in devices:
    # Filter non-zero values for the device
    non_zero_values = data[data[device] > 0][device]

    if not non_zero_values.empty:  # Check if there are any valid values
        min_values[device] = non_zero_values.min()
    else:
        min_values[device] = None  # If no non-zero values, assign None

# Display the results
for device, min_val in min_values.items():
    print(f"Device: {device}, Minimum Non-Zero Power: {min_val}")