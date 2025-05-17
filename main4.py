import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger le fichier et convertir explicitement la colonne "time" en datetime
df = pd.read_csv("HomeC.csv", low_memory=False)
# Ensure the 'time' column is numeric before using pd.to_datetime()
df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Then, safely parse the UNIX timestamp to datetime using the desired unit
df['time'] = pd.to_datetime(df['time'], unit='s')


# Définir "time" comme index
df.set_index("time", inplace=True)

# Supposons que ton DataFrame df est déjà chargé et indexé par "time" (DatetimeIndex)
# et que la colonne "Dishwasher [kW]" représente l'usage du lave-vaisselle.
# On définit un seuil pour considérer que l'appareil est en fonctionnement.
seuil = 0.001

# Ajouter une colonne "date" pour grouper par jour
df['date'] = df.index.date

# Initialiser des listes pour stocker α et β (en secondes depuis minuit) pour chaque jour
alpha_list = []
beta_list = []

# Initialiser une liste pour stocker le LOT (en heures) pour chaque jour
lot_list = []
i = 0
# Parcourir chaque jour
for jour, group in df.groupby('date'):
    print(group)

    i = i + 1
    if i > 4:
        exit(0)

    # Filtrer les enregistrements où l'appareil est utilisé
    usage = group[group["Dishwasher [kW]"] > seuil]
    if not usage.empty:
        # α : le premier instant d'utilisation de la journée
        alpha = usage.index.min()
        # β : le dernier instant d'utilisation de la journée
        beta = usage.index.max()
        # Calculer LOT : durée totale d'utilisation en heures
        lot = (beta - alpha).total_seconds() / 3600.0
        lot_list.append(lot)

        # Convertir l'heure en secondes depuis minuit
        alpha_seconds = alpha.hour * 3600 + alpha.minute * 60 + alpha.second
        beta_seconds = beta.hour * 3600 + beta.minute * 60 + beta.second

        alpha_list.append(alpha_seconds)
        beta_list.append(beta_seconds)

# Calcul de la moyenne (ou médiane) de α et β sur tous les jours
if alpha_list and beta_list:
    avg_alpha_seconds = np.mean(alpha_list)  # Remplacer par np.median(alpha_list) si besoin
    avg_beta_seconds = np.mean(beta_list)  # Remplacer par np.median(beta_list) si besoin

    # Conversion des secondes en heure, minute, seconde
    avg_alpha_hour = int(avg_alpha_seconds // 3600)
    avg_alpha_minute = int((avg_alpha_seconds % 3600) // 60)
    avg_alpha_second = int(avg_alpha_seconds % 60)

    avg_beta_hour = int(avg_beta_seconds // 3600)
    avg_beta_minute = int((avg_beta_seconds % 3600) // 60)
    avg_beta_second = int(avg_beta_seconds % 60)

    print(f"Valeur moyenne de α (heure de début) : {avg_alpha_hour:02d}:{avg_alpha_minute:02d}:{avg_alpha_second:02d}")
    print(f"Valeur moyenne de β (heure limite) : {avg_beta_hour:02d}:{avg_beta_minute:02d}:{avg_beta_second:02d}")
else:
    print("Aucune utilisation détectée pour le lave-vaisselle sur les jours analysés.")

if lot_list:
    avg_lot = np.mean(lot_list)
    print("LOT moyen (durée d'opération) :", avg_lot, "heures")
else:
    print("Aucune utilisation détectée pour le calcul du LOT.")