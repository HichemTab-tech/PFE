import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../HomeC.csv', low_memory=False)
data.dropna(subset=['time'], inplace=True)
data['time'] = pd.to_numeric(data['time'], errors='coerce')
data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
data.dropna(subset=['time'], inplace=True)
data['date'] = data['time'].dt.date

data['hour'] = data['time'].dt.hour

devices = [
    'Dishwasher [kW]',
    'Furnace 1 [kW]',
    'Furnace 2 [kW]',
    'Home office [kW]',
    'Fridge [kW]',
    'Wine cellar [kW]',
    'Garage door [kW]',
    'Kitchen 12 [kW]',
    'Kitchen 14 [kW]',
    'Kitchen 38 [kW]',
    'Barn [kW]',
    'Well [kW]',
    'Living room [kW]',
    'Microwave [kW]',
    'Furnace 1 [kW]'
]

seventh_day_data = data[data['date'] == pd.to_datetime('2016-01-07').date()]
print(seventh_day_data)

# On génère un graphique pour chaque jour individuellement
for current_date in data['date'].unique():
    daily_data = data[data['date'] == current_date]
    hourly_device_usage = daily_data.groupby('hour')[devices].mean()

    day_name = pd.to_datetime(current_date).strftime('%A')
    if pd.to_datetime(current_date).weekday() >= 5:
        day_name += ' (Weekend)'

    hourly_device_usage.plot(figsize=(12, 6))
    plt.xlabel('Heure de la journée')
    plt.ylabel('Consommation (kW)')
    plt.title(f'Consommation par appareil pour le {day_name}')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend(title='Appareils')
    plt.tight_layout()
    plt.show()


