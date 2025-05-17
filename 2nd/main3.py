import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min')) #upgrade the time column with a readable date
# data.dropna(subset=['time'], inplace=True)
# data['time'] = pd.to_numeric(data['time'], errors='coerce')
# data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
# data.dropna(subset=['time'], inplace=True)
data['date'] = data['time'].dt.date

# Calcul de la consommation horaire moyenne (24h)
data['hour'] = data['time'].dt.hour

hourly_usage = data.groupby('hour')['use [kW]'].mean().reset_index()

# Affichage clair de la consommation horaire
plt.figure(figsize=(10, 5))
plt.bar(hourly_usage['hour'], hourly_usage['use [kW]'], color='skyblue')
plt.xlabel('Heure de la journée')
plt.ylabel('Consommation moyenne (kW)')
plt.title('Consommation horaire moyenne (sur 7 jours)')
plt.xticks(range(0, 24))
plt.grid(axis='y')
plt.tight_layout()
plt.show()

