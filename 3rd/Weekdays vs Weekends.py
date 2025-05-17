import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv('HomeC.csv', low_memory=False)

# Recréer la colonne 'time' avec la fréquence réelle d'une minute
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

# Ajouter des colonnes utiles pour l'analyse
data['weekday'] = data['time'].dt.weekday
data['hour'] = data['time'].dt.hour

# Identifier clairement les jours weekends et weekdays
data['day_type'] = data['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Calculer la consommation horaire moyenne regroupée par weekend et weekday
hourly_daytype = data.groupby(['day_type', 'hour'])['use [kW]'].mean().unstack(0)

# Tracer clairement les résultats
hourly_daytype.plot(figsize=(12, 6))
plt.xlabel('Heure de la journée')
plt.ylabel('Consommation moyenne (kW)')
plt.title('Consommation horaire moyenne : Weekday vs Weekend')
plt.xticks(range(0,24))
plt.grid(True)
plt.legend(title='Type de Jour')
plt.tight_layout()
plt.show()
