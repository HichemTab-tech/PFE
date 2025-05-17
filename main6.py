import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Chargement et préparation des données ---
# On charge le fichier CSV et on convertit la colonne 'time' en DatetimeIndex.
df = pd.read_csv("HomeC.csv", low_memory=False)
# Ici, on suppose que la colonne 'time' est un timestamp UNIX (en secondes)
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index("time", inplace=True)

# Pour cet exemple, nous considérons que la donnée est échantillonnée par intervalles de 30 minutes.
tau_interval = 0.5  # Durée d'un intervalle en heures

# --- 2. Définition des appareils fixes et de leurs paramètres ---
# D'après l'article, pour les Fixed Loads, on dispose de :
#   ρ_f : la puissance nominale (en kW)
#   X_f,n : l'état de l'appareil à l'intervalle n (1 si allumé, 0 si éteint)
# Ici, nous supposons que pour les appareils fixes, on peut utiliser directement les valeurs mesurées.
# Par exemple, pour le Fridge et le Home office, qui fonctionnent généralement en continu.
fixed_appliances = {
    "Dishwasher [kW]": {"alpha": "07:00:00", "beta": "21:00:00"},       # fonctionnement 24h
    "Microwave [kW]": {"alpha": "08:00:00", "beta": "18:00:00"},    # plages d'utilisation typique
    "Garage door [kW]": {"alpha": "00:00:00", "beta": "24:00:00"}
}

# --- 3. Calcul de la consommation quotidienne par appareil fixe ---
# Pour chaque appareil, on calcule l'énergie consommée sur la journée en multipliant la consommation par
# la durée d'un intervalle (en h) et en faisant la somme sur tous les intervalles de la journée.
# On regroupe par jour (on utilise df.index.date).

daily_energy = {}  # dictionnaire pour stocker l'énergie quotidienne (en kWh) de chaque appareil fixe

for appliance in fixed_appliances.keys():
    # On suppose que la colonne contient la consommation instantanée en kW pour l'intervalle
    energy_by_day = df.groupby(df.index.date)[appliance].sum() * tau_interval
    daily_energy[appliance] = energy_by_day

    print(f"\nConsommation quotidienne pour {appliance} (en kWh) :")
    print(energy_by_day)

    # --- 4. Visualisation ---
    plt.figure(figsize=(10, 6))
    energy_by_day.plot(marker='o', linestyle='-')
    plt.title(f"Consommation quotidienne de {appliance}")
    plt.xlabel("Date")
    plt.ylabel("Énergie (kWh)")
    plt.grid(True)
    plt.show()

# --- 5. Calcul global sur tous les appareils fixes ---
# On peut également calculer l'énergie totale consommée par tous les Fixed Loads sur la journée.
total_energy_daily = None
for appliance, energy in daily_energy.items():
    if total_energy_daily is None:
        total_energy_daily = energy.copy()
    else:
        total_energy_daily += energy

print("\nConsommation quotidienne totale pour les appareils fixes :")
print(total_energy_daily)

plt.figure(figsize=(10,6))
total_energy_daily.plot(marker='o', linestyle='-', color='purple')
plt.title("Consommation quotidienne totale pour les appareils fixes")
plt.xlabel("Date")
plt.ylabel("Énergie (kWh)")
plt.grid(True)
plt.show()
