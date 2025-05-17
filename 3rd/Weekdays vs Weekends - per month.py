import pandas as pd
import plotly.express as px

# Chargement des données
data = pd.read_csv('HomeC.csv', low_memory=False)

# Recréer la colonne 'time' avec la fréquence réelle d'une minute
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))

# Ajouter des colonnes utiles pour l'analyse
data['month'] = data['time'].dt.month
data['hour'] = data['time'].dt.hour
data['weekday'] = data['time'].dt.weekday

# Identifier clairement les jours weekends et weekdays
data['day_type'] = data['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Calculer la consommation horaire moyenne regroupée par mois et type de jour
hourly_monthly_daytype = data.groupby(['month', 'day_type', 'hour'])['use [kW]'].mean()
hourly_monthly_daytype = hourly_monthly_daytype.reset_index()  # Reset index for Plotly compatibility

# Tracer des résultats interactifs avec Plotly
fig = px.line(
    hourly_monthly_daytype,
    x='hour',
    y='use [kW]',
    color='month',
    line_group='day_type',
    title='Consommation horaire moyenne par mois (Weekdays vs Weekends)',
    labels={
        'hour': 'Heure de la journée',
        'use [kW]': 'Consommation moyenne (kW)',
        'month': 'Mois'
    }
)
fig.update_layout(
    hovermode='x',  # Reactivity on hover
    xaxis=dict(tickmode='linear', dtick=1),  # Ensure hour ticks are regular
    legend_title="Month and Day Type"
)
fig.show()