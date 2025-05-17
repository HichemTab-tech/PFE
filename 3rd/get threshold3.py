import pandas as pd
import json

# Load dataset
data = pd.read_csv("HomeC.csv", low_memory=False)

# Load device names
with open('../devices.json') as f:
    devices = json.load(f)

# Settings: Adjust these as needed
PERCENT_OF_MAX_THRESHOLD = 0.2  # Use 20% of max for binary devices
STD_MULTIPLIER = 3  # k-value for mean + k * std thresholding
MIN_POWER_THRESHOLD = 0.001  # Ignore devices below 0.001 kW max power as noise


# Method to calculate mean
def calculate_mean(data):
    return sum(data) / len(data)


# Method to calculate standard deviation
def calculate_std(data):
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5


# Hybrid threshold calculation
def calculate_thresholds(device_data):
    results = {}

    for device, readings in device_data.items():
        # Remove zeros and filter out noise (e.g., negligible values)
        non_zero_readings = [x for x in readings if x > 0]
        if not non_zero_readings:
            results[device] = {"max": None, "mean": None, "threshold": None}
            continue  # Skip devices with no meaningful power consumption

        max_value = max(non_zero_readings)
        mean_value = calculate_mean(non_zero_readings)
        std_value = calculate_std(non_zero_readings)

        threshold = PERCENT_OF_MAX_THRESHOLD * max_value

        # Store results
        results[device] = {
            "max": max_value,
            "mean": mean_value,
            "threshold": threshold
        }

    return results


# Run the threshold calculations
threshold_results = calculate_thresholds(data[devices])

# Save thresholds to a JSON file
with open('../statistical_thresholds.json', 'w') as json_file:
    json.dump({device: stats["threshold"] for device, stats in threshold_results.items()}, json_file, indent=4)

print("Thresholds calculated and saved to 'statistical_thresholds.json'")

# Display the results
import plotly.graph_objects as go

# Convert stats to a DataFrame and include device names
stats_df = pd.DataFrame.from_dict(threshold_results, orient='index').reset_index()
stats_df.rename(columns={'index': 'Device'}, inplace=True)

# Create a pretty table using Plotly
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(stats_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[stats_df[col] for col in stats_df.columns], fill_color='lavender', align='left')
)])

fig_table.update_layout(title_text="Device Statistics")
fig_table.show()