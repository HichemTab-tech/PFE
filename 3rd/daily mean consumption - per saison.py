import pandas as pd
import plotly.express as px

# Chargement des données
data = pd.read_csv('HomeC.csv', low_memory=False)

# Recréer la colonne 'time' avec la fréquence réelle d'une minute
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

# Ajouter des colonnes utiles pour l'analyse
data['month'] = data['time'].dt.month
data['hour'] = data['time'].dt.hour

# Définir clairement les saisons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Hiver'
    elif month in [3, 4, 5]:
        return 'Printemps'
    elif month in [6, 7, 8]:
        return 'Été'
    else:
        return 'Automne'

data['season'] = data['month'].apply(get_season)

# Calculer la consommation horaire moyenne par saison
hourly_seasonal = data.groupby(['season', 'hour'])['use [kW]'].mean().reset_index()

# Tracer clairement les résultats
fig = px.line(hourly_seasonal, x='hour', y='use [kW]', color='season',
              title='Consommation horaire moyenne par saison',
              labels={'hour': 'Heure de la journée', 'use [kW]': 'Consommation moyenne (kW)', 'season': 'Saison'})
fig.update_layout(xaxis=dict(tickmode='linear'))
fig.show()