import pandas as pd
import plotly.express as px

data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

data['month'] = data['time'].dt.strftime('%B')  # Noms des mois
data['month_num'] = data['time'].dt.month       # Pour l'ordre
data['hour'] = data['time'].dt.hour

# Appareils à étudier
import json
with open('../devices.json') as f:
    devices = json.load(f)

# Moyenne mensuelle globale par appareil
monthly_device_avg = data.groupby(['month', 'month_num'])[devices].mean().reset_index()

# Transformer en long format
df_melted = monthly_device_avg.melt(id_vars=['month', 'month_num'],
                                    value_vars=devices,
                                    var_name='Appareil',
                                    value_name='Conso Moyenne (kW)')

# 3D Plot
fig = px.scatter_3d(df_melted,
                    x='Appareil',
                    y='month_num',
                    z='Conso Moyenne (kW)',
                    color='Appareil',
                    hover_name='month',
                    title='Consommation moyenne mensuelle par appareil (3D)',
                    labels={'month_num': 'Mois'})

# Rendre les mois dans le bon ordre
fig.update_layout(scene=dict(yaxis=dict(tickmode='array',
                                        tickvals=list(range(1,13)),
                                        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])))

fig.show()
