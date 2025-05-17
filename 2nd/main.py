import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../HomeC.csv', low_memory=False)

data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min')) #upgrade the time column with a readable date
# data.dropna(subset=['time'], inplace=True)
# data['time'] = pd.to_numeric(data['time'], errors='coerce')
# data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
# data.dropna(subset=['time'], inplace=True)
data['date'] = data['time'].dt.date

# Calcul correct de l'énergie journalière (kWh)
daily_energy = data.groupby('date')['use [kW]'].mean() * (86400 / 3600)
daily_energy = daily_energy.reset_index(name='daily_energy_kWh')

# Ajouter le nom court des jours de la semaine
# daily_energy['day_name'] = pd.to_datetime(daily_energy['date']).dt.strftime('%a')

# Identifier jours max/min
max_day = daily_energy.loc[daily_energy['daily_energy_kWh'].idxmax()]
min_day = daily_energy.loc[daily_energy['daily_energy_kWh'].idxmin()]

# Weekday vs Weekend
data['weekday'] = data['time'].dt.weekday
data['day_type'] = data['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
week_type_usage = data.groupby('day_type')['use [kW]'].mean().reset_index()

# Plot clair et simple par jours
plt.figure(figsize=(10, 5))
plt.plot(daily_energy['date'], daily_energy['daily_energy_kWh'], marker='o')
plt.xlabel('Jour de la semaine')
plt.ylabel('Énergie journalière (kWh)')
plt.title('Énergie consommée par jour')
plt.grid(True)
plt.tight_layout()
plt.show()

# Affichage de weekday vs weekend
week_type_usage = data.groupby('day_type')['use [kW]'].mean().reset_index()
week_type_usage.plot.bar(x='day_type', y='use [kW]', legend=False)
plt.xlabel('Type de jour')
plt.ylabel('Consommation moyenne (kW)')
plt.title('Consommation Moyenne : Weekday vs Weekend')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
