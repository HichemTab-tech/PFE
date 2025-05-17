import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# === 1. LOAD & PREPARE DATA ===
data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))
data['hour'] = data['time'].dt.hour
data['month'] = data['time'].dt.month

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

# === 2. DEFINE DEVICES ===
import json
with open('../devices.json') as f:
    devices = json.load(f)
with open('../devices-thresholds-3.json') as th:
    thresholds = json.load(th)

# slice devices to one only
#devices = [devices[1]]

# === 3. FLEXIBILITY PER SEASON ===
all_flex_results = []

for season in data['season'].unique():
    season_data = data[data['season'] == season]
    flex_results = []

    for device in devices:
        actif = season_data[season_data[device] > thresholds[device]]
        hours = actif['hour']
        if len(hours) < 10:
            continue

        mean_hour = hours.mean()
        std_hour = hours.std()
        label = 'Flexible' if std_hour > 4 else ('Strict' if std_hour < 1.5 else 'Moyen')

        flex_results.append({
            'Saison': season,
            'Appareil': device,
            'Heure Moyenne': round(mean_hour, 2),
            'Écart-type': round(std_hour, 2),
            'Profil': label
        })

        # Histogramme interactif par saison
        # fig = px.histogram(hours, nbins=24,
        #                    title=f"Distribution horaire ({device}) - {season}",
        #                    labels={'value': 'Heure de la journée', 'count': 'Fréquence'})
        # fig.update_layout(bargap=0.1)
        # fig.show()

    all_flex_results.extend(flex_results)

# === 4. DISPLAY SUMMARY TABLE ===
flex_df = pd.DataFrame(all_flex_results)
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(flex_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[flex_df[col] for col in flex_df.columns], fill_color='lavender', align='left')
)])

fig_table.update_layout(title_text="Résumé de la flexibilité horaire par appareil et par saison")
fig_table.show()
