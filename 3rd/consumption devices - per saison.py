import pandas as pd
import plotly.express as px

data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

# Colonnes utiles
data['month'] = data['time'].dt.month
data['hour'] = data['time'].dt.hour

# Définir saisons
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

# Appareils à étudier
import json

with open('../devices.json') as f:
    devices = json.load(f)

# Consommation horaire moyenne par appareil et saison
device_hourly_seasonal = data.groupby(['season', 'hour'])[devices].mean().reset_index()

# Conversion pour Plotly
df_melted = device_hourly_seasonal.melt(id_vars=['season', 'hour'], value_vars=devices,
                                        var_name='Appareil', value_name='Consommation (kW)')

# Graphique clair
fig = px.line(df_melted, x='hour', y='Consommation (kW)', color='Appareil',
              facet_col='season', facet_col_wrap=2,
              title='Consommation horaire des appareils par saison')
fig.update_layout(xaxis=dict(tickmode='linear'))

fig.show()
