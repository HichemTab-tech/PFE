import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min')) #upgrade the time column with a readable date
# data.dropna(subset=['time'], inplace=True)
# data['time'] = pd.to_numeric(data['time'], errors='coerce')
# data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
# data.dropna(subset=['time'], inplace=True)
data['date'] = data['time'].dt.date

# Ajouter colonne 'hour' pour regrouper par heure
data['hour'] = data['time'].dt.hour

# Grouper par date et heure pour chaque jour indépendamment
hourly_usage_per_day = data.groupby(['date', 'hour'])['use [kW]'].mean().unstack(level=0)

# Afficher clairement les résultats sous forme graphique (chaque jour une ligne)
hourly_usage_per_day.plot(figsize=(12, 6))
plt.xlabel('Heure de la journée')
plt.ylabel('Consommation (kW)')
plt.title('Consommation horaire par jour')
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend(title='Jour')
plt.tight_layout()
plt.show()


