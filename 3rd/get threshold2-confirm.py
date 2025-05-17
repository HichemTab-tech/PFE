import pandas as pd
import json

# Load dataset
data = pd.read_csv("HomeC.csv", low_memory=False)

# Load device names
with open('../devices.json') as f:
    devices = json.load(f)
with open('../devices-thresholds-2.json') as th:
    thresholds = json.load(th)



# Function to calculate max, min, and mean for each dataset
def calculate_stats(data):
    results = {}
    for device, readings in data.items():
        # Filter out zero values (optional, depending on your context)
        non_zero_readings = [x for x in readings if x > 0]

        if non_zero_readings:  # Ensure the list is not empty after filtering
            max_value = max(non_zero_readings)
            min_value = min(non_zero_readings)
            mean_value = sum(non_zero_readings) / len(non_zero_readings)

            # Store results
            results[device] = {
                "max": max_value,
                "min": min_value,
                "mean": mean_value,
                "threshold": thresholds[device] if device in thresholds else None
            }
        else:
            # If no non-zero data exists, indicate as 'No data'
            results[device] = {
                "max": None,
                "min": None,
                "mean": None,
                "threshold": thresholds[device] if device in thresholds else None
            }

    return results


# Call the function and display stats
stats = calculate_stats(data[devices])

# Display the results
import plotly.graph_objects as go

# Convert stats to a DataFrame and include device names
stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index()
stats_df.rename(columns={'index': 'Device'}, inplace=True)

# Create a pretty table using Plotly
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(stats_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[stats_df[col] for col in stats_df.columns], fill_color='lavender', align='left')
)])

fig_table.update_layout(title_text="Device Statistics")
fig_table.show()


for device, values in stats.items():
    print(f"Device: {device}")
    print(f"  Max: {values['max']}")
    print(f"  Min: {values['min']}")
    print(f"  Mean: {values['mean']}")
    print(f"  Threshold: {values['threshold']}")
    print()