import pandas as pd
import random


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


# Exemple simplifié de planning initial
planning_initial = {
    'Lave-linge': {'heure_debut': 15, 'duree_h': 1.5, 'puissance_kW': 1.0},
    'Four': {'heure_debut': 19, 'duree_h': 1.0, 'puissance_kW': 1.5},
    'Chauffage': {'heure_debut': 6, 'duree_h': 3.0, 'puissance_kW': 2.0}
}


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
def csa_optimisation(planning_initial, iterations=20):
    best_planning = planning_initial.copy()
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


# Exécution concrète du CSA
planning_optimise, fitness_optimise = csa_optimisation(planning_initial)

print(f"Planning optimisé : {planning_optimise}")
print(f"Valeur Fitness optimisée : {fitness_optimise:.2f} CAD")
