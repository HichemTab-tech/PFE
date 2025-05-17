import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# === 1. LOAD & PREPARE DATA ===
data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))
data['hour'] = data['time'].dt.hour
data['date'] = data['time'].dt.date
data['weekday'] = data['time'].dt.weekday

def get_season(month):
    if month in [12, 1, 2]:
        return 'Hiver'
    elif month in [3, 4, 5]:
        return 'Printemps'
    elif month in [6, 7, 8]:
        return 'Été'
    else:
        return 'Automne'


data['month'] = data['time'].dt.month
data['season'] = data['month'].apply(get_season)
data['day_type'] = data['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')


# === 2. DEFINE DEVICES ===
import json
with open('../devices.json') as f:
    devices = json.load(f)
with open('../devices-thresholds-3.json') as th:
    thresholds = json.load(th)

all_flex_results = []

for season in data['season'].unique():
    for day_type in ['Weekday', 'Weekend']:
        filtered = data[(data['season'] == season) & (data['day_type'] == day_type)]

        for device in devices:
            daily_sessions = filtered[filtered[device] > thresholds[device]].groupby('date')
            start_hours = []

            for day, group in daily_sessions:
                start_hour = group['hour'].iloc[0]
                start_hours.append(start_hour)

            if len(start_hours) < 3:
                continue

            series = pd.Series(start_hours)
            mean_hour = series.mean()
            std_hour = series.std()
            label = 'Flexible' if std_hour > 4 else ('Strict' if std_hour < 2 else 'Moyen')

            all_flex_results.append({
                'Saison': season,
                'Type de jour': day_type,
                'Appareil': device,
                'Jours analysés': len(start_hours),
                'Heure Moyenne de Début': round(mean_hour, 2),
                'Écart-type': round(std_hour, 2),
                'Profil': label
            })

            # Graphique pour visualiser
            # fig = px.histogram(series, nbins=24,
            #                    title=f"{device} - {season} ({day_type})",
            #                    labels={'value': 'Heure de démarrage', 'count': 'Nombre de jours'})
            # fig.update_layout(bargap=0.1)
            # fig.show()

# === 3. DISPLAY SUMMARY TABLE ===
flex_df = pd.DataFrame(all_flex_results)
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(flex_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[flex_df[col] for col in flex_df.columns], fill_color='lavender', align='left')
)])

fig_table.update_layout(title_text="Flexibilité horaire par saison et type de jour (session-based)")
fig_table.show()
