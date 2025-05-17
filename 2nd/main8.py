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

summary_table = pd.DataFrame(columns=['Appareil', 'Conso Moyenne (kW)', 'Conso Max (kW)', 'Heure du Pic Moyen'])

for device in devices:
    mean_conso = data[device].mean()
    max_conso = data[device].max()
    # Heure moyenne où la consommation max est atteinte
    hour_of_max = data.groupby('hour')[device].mean().idxmax()

    summary_table = summary_table._append({
        'Appareil': device,
        'Conso Moyenne (kW)': round(mean_conso, 3),
        'Conso Max (kW)': round(max_conso, 3),
        'Heure du Pic Moyen': hour_of_max
    }, ignore_index=True)

print(summary_table)


