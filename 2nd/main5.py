import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../HomeC.csv', low_memory=False)
data.dropna(subset=['time'], inplace=True)
data['time'] = pd.to_numeric(data['time'], errors='coerce')
data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
data.dropna(subset=['time'], inplace=True)
data['date'] = data['time'].dt.date

# Ajouter colonne 'hour' pour regrouper par heure
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

for device in devices:
    hourly_device_usage = data.groupby(['date', 'hour'])[device].mean().unstack(level=0)

    hourly_device_usage.plot(figsize=(12, 6))
    plt.xlabel('Heure de la journée')
    plt.ylabel(f'Consommation ({device})')
    plt.title(f'Consommation horaire par jour : {device}')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend(title='Jour')
    plt.tight_layout()
    plt.show()


