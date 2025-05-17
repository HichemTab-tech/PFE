import pandas as pd

# Load the CSV file
df = pd.read_csv("HomeC.csv", low_memory=False)

# Handle and clean the `time` column
df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S", errors="coerce")

df = df.dropna(subset=['time'])

# Set 'time' as the index
df.set_index('time', inplace=True)

# Resample data to 15-minute intervals
df_15min = df.resample('15min').mean()  # Use 'min' instead of 'T'

print(df_15min.head())

