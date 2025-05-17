import pandas as pd
import plotly.express as px

data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

data['month'] = data['time'].dt.strftime('%B')  # Nom des mois
data['hour'] = data['time'].dt.hour


# Appareils à étudier
import json
with open('../devices.json') as f:
    devices = json.load(f)

# Consommation horaire moyenne par appareil et par mois
monthly_devices = data.groupby(['month', 'hour'])[devices].mean().reset_index()

# Conversion pour Plotly
df_melted = monthly_devices.melt(id_vars=['month', 'hour'], value_vars=devices,
                                 var_name='Appareil', value_name='Consommation (kW)')

# Graphique interactif clair
fig = px.line(df_melted, x='hour', y='Consommation (kW)', color='Appareil',
              animation_frame='month',
              title='Consommation horaire moyenne par appareil pour chaque mois')

fig.update_layout(xaxis=dict(tickmode='linear'))
fig.show()
