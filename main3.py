import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier et convertir explicitement la colonne "time" en datetime
df = pd.read_csv("HomeC.csv", low_memory=False)
# Ensure the 'time' column is numeric before using pd.to_datetime()
df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Then, safely parse the UNIX timestamp to datetime using the desired unit
df['time'] = pd.to_datetime(df['time'], unit='s')


# Définir "time" comme index
df.set_index("time", inplace=True)

# Vérifier que l'index est bien un DatetimeIndex
print(type(df.index))  # Doit afficher <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

# Extraire l'heure de l'index
df['hour'] = df.index.hour

# Agréger l'utilisation de l'appareil par heure (exemple avec "Dishwasher [kW]")
equipements = ["Dishwasher [kW]", "Microwave [kW]", "Garage door [kW]"]

# loop
for equipement in equipements:


    avg_usage_by_hour = df.groupby('hour')[equipement].mean()

    # Tracer la moyenne d'utilisation par heure sur tous les jours
    plt.figure(figsize=(10,6))
    plt.plot(avg_usage_by_hour.index, avg_usage_by_hour, marker='o', linestyle='-')
    plt.title(f"Usage moyen de {equipement} par heure sur tous les jours")
    plt.xlabel("Heure de la journée")
    plt.ylabel("Usage moyen (kW)")
    plt.xticks(range(0,24))
    plt.grid(True)
    plt.show()
