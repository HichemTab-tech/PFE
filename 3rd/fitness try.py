import pandas as pd
import random
import plotly.express as px

# === 1. LOAD & PREPARE DATA ===
data = pd.read_csv('HomeC.csv', low_memory=False)
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data), freq='min'))
data['hour'] = data['time'].dt.hour
data['date'] = data['time'].dt.date

# === 2. SELECT DEVICES & BUILD INITIAL PLANNING ===
devices = [
    'Dishwasher [kW]', 'Furnace 1 [kW]', 'Microwave [kW]'
]

planning_initial = {}

for device in devices:
    appareil_data = data[data[device] > 0]
    if appareil_data.empty:
        continue

    heure_moyenne = int(appareil_data['hour'].mean())
    duree_typique = appareil_data.groupby(appareil_data['time'].dt.date).size().mean() / 60  # en heures
    duree_typique = min(max(duree_typique, 0.5), 3)  # clamp entre 0.5h et 3h
    puissance_moyenne = appareil_data[device].mean()

    planning_initial[device] = {
        'heure_debut': heure_moyenne,
        'duree_h': round(duree_typique, 2),
        'puissance_kW': round(puissance_moyenne, 2)
    }

# === 3. TARIFF FUNCTION (Canada-like) ===
def cout_energie(heure):
    if 7 <= heure < 11:
        return 0.20
    elif 11 <= heure < 17:
        return 0.15
    elif 17 <= heure < 19:
        return 0.22
    else:
        return 0.10

# === 4. FITNESS FUNCTION ===
def calculer_cout_total(planning):
    cout_total = 0
    for appareil, params in planning.items():
        heure_debut = params['heure_debut']
        duree = params['duree_h']
        puissance = params['puissance_kW']

        increments = int(duree * 2)
        for h in range(increments):
            heure_actuelle = (heure_debut + h * 0.5) % 24
            cout_horaire = cout_energie(int(heure_actuelle))
            cout_total += puissance * cout_horaire * 0.5

    return cout_total

def fitness(planning):
    return calculer_cout_total(planning)

# === 5. CROW SEARCH ALGORITHM ===
def csa_optimisation(planning_initial, n_corbeaux=10, iterations=100):
    corbeaux = [planning_initial.copy() for _ in range(n_corbeaux)]
    memoire = corbeaux.copy()
    fitness_memoire = [fitness(c) for c in corbeaux]

    meilleure_solution = corbeaux[fitness_memoire.index(min(fitness_memoire))]
    meilleure_fitness = min(fitness_memoire)

    for _ in range(iterations):
        for i in range(n_corbeaux):
            j = random.randint(0, n_corbeaux - 1)
            nouveau = {}

            for appareil in corbeaux[i]:
                if random.random() < 0.5:
                    nouveau[appareil] = memoire[j][appareil].copy()
                else:
                    nouveau[appareil] = corbeaux[i][appareil].copy()
                    nouveau[appareil]['heure_debut'] = random.randint(0, 23)

            nouvelle_fitness = fitness(nouveau)

            if nouvelle_fitness < fitness_memoire[i]:
                memoire[i] = nouveau
                fitness_memoire[i] = nouvelle_fitness
                if nouvelle_fitness < meilleure_fitness:
                    meilleure_solution = nouveau
                    meilleure_fitness = nouvelle_fitness

    return meilleure_solution, meilleure_fitness

# === 6. EXECUTION ===
print("Planning initial :")
print(planning_initial)
print("Coût initial :", round(fitness(planning_initial), 3), "CAD")

planning_opt, fitness_opt = csa_optimisation(planning_initial)

print("\nPlanning optimisé :")
print(planning_opt)
print("Coût optimisé :", round(fitness_opt, 3), "CAD")