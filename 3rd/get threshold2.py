import pandas as pd
import json

# Load dataset
data = pd.read_csv("HomeC.csv", low_memory=False)

# Load device names
with open('../devices.json') as f:
    devices = json.load(f)

# Initialize dictionary to store thresholds
thresholds = {}

# Calculate thresholds using mean + k * std
k = 1.5  # You can adjust k based on sensitivity
for device in devices:
    # Filter data where power is above 0
    non_zero_data = data[data[device] > 0][device]

    if not non_zero_data.empty:  # Check if data is available
        mean = non_zero_data.mean()  # Calculate mean
        std = non_zero_data.std()  # Calculate standard deviation
        threshold = mean + (k * std)  # Calculate threshold
        thresholds[device] = threshold
    else:
        thresholds[device] = None  # No data available

# Save thresholds to a JSON file
with open('../statistical_thresholds.json', 'w') as json_file:
    json.dump(thresholds, json_file, indent=4)

print("Thresholds calculated and saved to 'statistical_thresholds.json'")