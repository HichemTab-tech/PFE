import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# === 1. LOAD & PREPARE DATA ===
data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))
data['hour'] = data['time'].dt.hour
data['date'] = data['time'].dt.date


# === 2. DEFINE DEVICES ===
import json
with open('../devices.json') as f:
    devices = json.load(f)
with open('../devices-thresholds-3.json') as th:
    thresholds = json.load(th)

flex_results = []

for device in devices:
    # Filtrer les jours où l'appareil a été utilisé
    daily_sessions = data[data[device] > thresholds[device]].groupby('date')
    start_hours = []

    for day, group in daily_sessions:
        start_hour = group['hour'].iloc[0]  # Première heure d'activité dans la journée
        start_hours.append(start_hour)

    if len(start_hours) < 3:
        continue  # Pas assez de données pour évaluer

    start_hours_series = pd.Series(start_hours)
    mean_hour = start_hours_series.mean()
    std_hour = start_hours_series.std()
    label = 'Flexible' if std_hour > 4 else ('Strict' if std_hour < 1.5 else 'Moyen')

    flex_results.append({
        'Appareil': device,
        'Jours analysés': len(start_hours),
        'Heure Moyenne de Début': round(mean_hour, 2),
        'Écart-type': round(std_hour, 2),
        'Profil': label
    })

    # Histogramme des heures de démarrage par jour
    # fig = px.histogram(start_hours_series, nbins=24,
    #                    title=f"Heures de démarrage journalières - {device}",
    #                    labels={'value': 'Heure de démarrage', 'count': 'Nombre de jours'})
    # fig.update_layout(bargap=0.1)
    # fig.show()

# === 3. DISPLAY SUMMARY TABLE ===
flex_df = pd.DataFrame(flex_results)
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(flex_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[flex_df[col] for col in flex_df.columns], fill_color='lavender', align='left')
)])

fig_table.update_layout(title_text="Flexibilité basée sur l'heure de démarrage quotidienne par appareil")
fig_table.show()
