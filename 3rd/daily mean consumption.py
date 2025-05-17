import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../HomeC.csv', low_memory=False)

# Recreate the 'time' column with the actual frequency of one minute
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

# Add useful columns for analysis
data['date'] = data['time'].dt.date
data['hour'] = data['time'].dt.hour

# Filter data for the specific day
specific_day = pd.to_datetime('2016-01-05').date()
data_specific_day = data[data['date'] == specific_day]

# Calculate the hourly mean consumption for the specific day
hourly_mean = data_specific_day.groupby('hour')['use [kW]'].mean()

# Plot the results
hourly_mean.plot(figsize=(10, 5), marker='o')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Consumption (kW)')
plt.title(f'Hourly Mean Consumption for {specific_day}')
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.show()