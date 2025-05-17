import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("HomeC.csv", low_memory=False)
# Ensure the 'time' column is numeric before using pd.to_datetime()
df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Then, safely parse the UNIX timestamp to datetime using the desired unit
df['time'] = pd.to_datetime(df['time'], unit='s')


# Définir "time" comme index
df.set_index("time", inplace=True)

# Ajouter une colonne "date" pour grouper par jour
df['date'] = df.index.date

alpha_list = []  # liste pour stocker α (en secondes depuis minuit) pour chaque jour
beta_list = []   # liste pour stocker β
equipements = ["Dishwasher [kW]", "Microwave [kW]", "Garage door [kW]"]

device = equipements[2]

# Initialiser une liste pour stocker le LOT (en heures) pour chaque jour
lot_list = []
# Supposons que chaque enregistrement représente la consommation instantanée sur un intervalle constant.
# Si l'intervalle est de 5 minutes, le facteur de conversion est constant (peut être ignoré pour le calcul des quantiles).
for jour, group in df.groupby('date'):
    group = group.sort_index().copy()
    # On considère la consommation de l'appareil, ici on prend directement la valeur,
    # éventuellement multiplier par la durée si besoin (mais ici on travaille en proportions)
    group['consumption'] = group[device]
    total_consumption = group['consumption'].sum()
    if total_consumption == 0:
        continue

    # Calculer la consommation cumulée
    group['cumsum'] = group['consumption'].cumsum()

    # Définir α comme le moment où le cumul atteint 10% du total
    alpha_time = group[group['cumsum'] >= 0.1 * total_consumption].index[0]
    # Définir β comme le moment où le cumul atteint 90% du total
    beta_time = group[group['cumsum'] >= 0.9 * total_consumption].index[0]

    # Convertir ces instants en secondes depuis minuit
    alpha_sec = alpha_time.hour * 3600 + alpha_time.minute * 60 + alpha_time.second
    beta_sec = beta_time.hour * 3600 + beta_time.minute * 60 + beta_time.second
    # Calculer LOT : durée totale d'utilisation en heures
    lot = (beta_time - alpha_time).total_seconds() / 3600.0
    lot_list.append(lot)

    alpha_list.append(alpha_sec)
    beta_list.append(beta_sec)

# Calculer la moyenne (ou la médiane) de α et β sur l'ensemble des jours
if alpha_list and beta_list:
    avg_alpha_sec = np.mean(alpha_list)  # ou np.median(alpha_list)
    avg_beta_sec = np.mean(beta_list)    # ou np.median(beta_list)

    # Convertir en format HH:MM:SS
    avg_alpha = f"{int(avg_alpha_sec // 3600):02d}:{int((avg_alpha_sec % 3600) // 60):02d}:{int(avg_alpha_sec % 60):02d}"
    avg_beta = f"{int(avg_beta_sec // 3600):02d}:{int((avg_beta_sec % 3600) // 60):02d}:{int(avg_beta_sec % 60):02d}"

    print("Alpha (10% cumul) =", avg_alpha)
    print("Beta (90% cumul) =", avg_beta)
else:
    print("Aucune utilisation détectée pour", device)

if lot_list:
    avg_lot = np.mean(lot_list)
    print("LOT moyen (durée d'opération) :", avg_lot, "heures")
else:
    print("Aucune utilisation détectée pour le calcul du LOT.")