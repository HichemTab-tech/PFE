import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min')) #upgrade the time column with a readable date
# data.dropna(subset=['time'], inplace=True)
# data['time'] = pd.to_numeric(data['time'], errors='coerce')
# data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
# data.dropna(subset=['time'], inplace=True)
data['date'] = data['time'].dt.date

# Supposant que data['time'] est déjà convertie en datetime
data['weekday'] = data['time'].dt.weekday
data['type_jour'] = data['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Calculer consommation moyenne par type de jour
conso_type_jour = data.groupby('type_jour')['use [kW]'].mean()

# Affichage simple en graphique barre
conso_type_jour.plot(kind='bar', color=['skyblue', 'orange'])
plt.xlabel('Type de jour')
plt.ylabel('Consommation moyenne (kW)')
plt.title('Comparaison consommation : Weekday vs Weekend')
plt.grid(axis='y')
plt.show()