import pandas as pd
import random

data = pd.read_csv('../HomeC.csv', low_memory=False)

data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min')) #upgrade the time column with a readable date
# data.dropna(subset=['time'], inplace=True)
# data['time'] = pd.to_numeric(data['time'], errors='coerce')
# data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
# data.dropna(subset=['time'], inplace=True)
data['date'] = data['time'].dt.date
data['hour'] = data['time'].dt.hour

# data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
# data.dropna(subset=['time'], inplace=True)
# data['hour'] = data['time'].dt.hour

devices = [
    'Dishwasher [kW]',
    'Furnace 1 [kW]',
    'Furnace 2 [kW]',
    'Home office [kW]',
    'Fridge [kW]',
    'Wine cellar [kW]',
    'Garage door [kW]',
    'Kitchen 12 [kW]',
    'Kitchen 14 [kW]',
    'Kitchen 38 [kW]',
    'Barn [kW]',
    'Well [kW]',
    'Living room [kW]',
    'Microwave [kW]',
    'Furnace 1 [kW]'
]

planning_initial = {}

for device in devices:
    # Filtrer les moments où l'appareil fonctionne
    appareil_data = data[data[device] > 0]

    # Heure moyenne de démarrage
    heure_moyenne = int(appareil_data['hour'].mean())

    # Durée typique (nombre d'heures moyennes par jour d'utilisation)
    duree_typique = appareil_data.groupby(appareil_data['time'].dt.date).size().mean() / 3600  # en heures

    # Puissance moyenne réelle
    puissance_moyenne = appareil_data[device].mean()

    planning_initial[device] = {
        'heure_debut': heure_moyenne,
        'duree_h': round(duree_typique, 2),
        'puissance_kW': round(puissance_moyenne, 2)
    }

print("\nPlanning initial basé sur les données réelles :")
print(planning_initial)

# Tarif énergétique simplifié (Canada)
def cout_energie(heure):
    if 7 <= heure < 11:  # Morning peak
        return 0.20
    elif 11 <= heure < 17:  # Mid-peak
        return 0.15
    elif 17 <= heure < 19:  # Evening peak
        return 0.22
    else:  # Off-peak
        return 0.10


# Fonction pour calculer le coût total
def calculer_cout_total(planning):
    cout_total = 0
    for appareil, params in planning.items():
        heure_debut = params['heure_debut']
        duree = params['duree_h']
        puissance = params['puissance_kW']

        increments = int(duree * 2)  # Incréments de 0.5 heure
        for h in range(increments):
            heure_actuelle = (heure_debut + h * 0.5) % 24
            cout_horaire = cout_energie(int(heure_actuelle))
            cout_total += puissance * cout_horaire * 0.5  # (kW * $/kWh * h)

    return cout_total


# Fonction fitness pour CSA
def fitness(planning):
    return calculer_cout_total(planning)

fitness_sans_optimization = fitness(planning_initial)

print(f"Valeur sans optimisation : {fitness_sans_optimization:.2f} CAD")

# Exemple minimaliste d'algorithme CSA simplifié
def csa_optimisation(planning_initial_, iterations=200):
    best_planning = planning_initial_.copy()
    best_fitness = fitness(best_planning)

    for _ in range(iterations):
        nouveau_planning = best_planning.copy()

        # Perturber aléatoirement l'heure de démarrage des appareils
        appareil = random.choice(list(nouveau_planning.keys()))
        nouveau_planning[appareil]['heure_debut'] = random.randint(0, 23)

        nouveau_fitness = fitness(nouveau_planning)

        if nouveau_fitness < best_fitness:
            best_planning = nouveau_planning
            best_fitness = nouveau_fitness

    return best_planning, best_fitness





planning_optimise, fitness_optimise = csa_optimisation(planning_initial)

print(f"Planning optimisé basé sur données réelles : {planning_optimise}")
print(f"Coût optimisé : {fitness_optimise:.2f} CAD")