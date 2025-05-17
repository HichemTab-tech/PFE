import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# === 1. LOAD & PREPARE DATA ===
data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))
data['hour'] = data['time'].dt.hour

# Appareils à étudier
import json
with open('../devices.json') as f:
    devices = json.load(f)
with open('../devices-thresholds.json') as th:
    thresholds = json.load(th)

flex_results = []

for device in devices:
    actif = data[data[device] > thresholds[device]]
    hours = actif['hour']
    if len(hours) < 10:
        continue

    mean_hour = hours.mean()
    std_hour = hours.std()
    label = 'Flexible' if std_hour > 2 else ('Strict' if std_hour < 1.5 else 'Moyen')

    flex_results.append({
        'Appareil': device,
        'Heure Moyenne': round(mean_hour, 2),
        'Écart-type': round(std_hour, 2),
        'Profil': label
    })

    # Graphique de distribution (histogramme interactif)
    fig = px.histogram(hours, nbins=24, title=f"Distribution horaire de l'appareil: {device}",
                       labels={'value': 'Heure de la journée', 'count': 'Fréquence'})
    fig.update_layout(bargap=0.1)
    fig.show()

# Résumé en tableau interactif
flex_df = pd.DataFrame(flex_results)
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(flex_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[flex_df[col] for col in flex_df.columns], fill_color='lavender', align='left')
)])

fig_table.update_layout(title_text="Résumé de la flexibilité horaire par appareil")
fig_table.show()


## print a table in console too
from tabulate import tabulate
print(tabulate(flex_df, headers='keys', tablefmt='psql'))