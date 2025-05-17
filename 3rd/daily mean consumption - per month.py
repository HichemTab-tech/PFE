import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv('HomeC.csv', low_memory=False)

# Recréer la colonne 'time' avec la fréquence réelle d'une minute
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

# Ajouter des colonnes utiles pour l'analyse
data['month'] = data['time'].dt.month
data['hour'] = data['time'].dt.hour

# Calculer la consommation horaire moyenne regroupée par mois
hourly_monthly = data.groupby(['month', 'hour'])['use [kW]'].mean().unstack(0)

# Tracer clairement les résultats
hourly_monthly.plot(figsize=(12, 6))
plt.xlabel('Heure de la journée')
plt.ylabel('Consommation moyenne (kW)')
plt.title('Consommation horaire moyenne par mois')
plt.xticks(range(0,24))
plt.grid(True)
plt.legend(title='Mois')
plt.tight_layout()
plt.show()