import numpy as np
import matplotlib.pyplot as plt

# --- 1. Paramètres des appareils (en secondes) ---
devices = {
    "Dishwasher": {
        "alpha": 9 * 3600,  # 09:00:00
        "beta": 17 * 3600,  # 17:00:00
        "LOT": 2 * 3600,  # 2 heures
        "power": 1.2
    },
    "Microwave": {
        "alpha": 8 * 3600,  # 08:00:00
        "beta": 11 * 3600,  # 11:00:00
        "LOT": int(0.5 * 3600),  # 0.5 heure
        "power": 0.8
    },
    "Garage_door": {
        "alpha": 8 * 3600,  # 08:00:00
        "beta": 10 * 3600,  # 10:00:00
        "LOT": int(0.25 * 3600),  # 0.25 heure
        "power": 0.5
    }
}

# Calcul de la plage autorisée pour le démarrage : [α, β - LOT]
for device in devices:
    a = devices[device]["alpha"]
    b = devices[device]["beta"]
    LOT = devices[device]["LOT"]
    devices[device]["min_eta"] = a
    devices[device]["max_eta"] = b - LOT

print("Plages autorisées (en secondes depuis minuit) :")
for device, params in devices.items():
    print(f"{device}: de {params['min_eta']} à {params['max_eta']}")


# --- 2. Représentation d'une solution (chromosome) ---
def random_solution(devices):
    sol = {}
    for device, params in devices.items():
        low = params["min_eta"]
        high = params["max_eta"]
        sol[device] = np.random.randint(low, high + 1)
    return sol


# --- 3. Nouvelle fonction fitness non triviale ---
def fitness(solution, devices, lambda1=0.5, lambda2=0.5):
    total = 0.0
    norm_times = []
    for device, params in devices.items():
        a = params["alpha"]
        max_eta = params["max_eta"]
        eta = solution[device]
        # Calcul du temps d'attente normalisé
        denom = (max_eta - a)
        waiting = (eta - a) / denom if denom > 0 else 0
        norm_times.append(waiting)

        # Prix non linéaire : prix = p_max - (p_max - p_min) * (waiting)^2
        p_max = 0.35
        p_min = 0.15
        price = p_max - (p_max - p_min) * (waiting ** 2)
        cost = price * params["power"]

        total += lambda1 * waiting + lambda2 * cost
    # On peut éventuellement ajouter un terme de dispersion (facultatif)
    variance = np.var(norm_times)
    total -= 0.1 * variance  # petit bonus pour synchroniser (à ajuster)
    return total


# --- 4. Algorithme Hybride TG-MFO Simplifié ---
pop_size = 100
iterations = 200

# Génération de la population initiale
population = [random_solution(devices) for _ in range(pop_size)]
fitness_values = [fitness(sol, devices) for sol in population]

best_fitness_history = []
best_solution_history = []

for it in range(iterations):
    best_idx = np.argmin(fitness_values)
    best_sol = population[best_idx]
    best_fit = fitness_values[best_idx]
    best_fitness_history.append(best_fit)
    best_solution_history.append(best_sol)

    # Phase MFO : mise à jour inspirée de la spirale logarithmique
    new_population = []
    b_constant = 1
    for sol in population:
        new_sol = {}
        for device, params in devices.items():
            eta = sol[device]
            best_eta = best_sol[device]
            d = abs(best_eta - eta)
            t = np.random.uniform(-1, 1)
            new_eta = best_eta + d * np.exp(b_constant * t) * np.cos(2 * np.pi * t)
            low = params["min_eta"]
            high = params["max_eta"]
            new_sol[device] = int(np.clip(new_eta, low, high))
        new_population.append(new_sol)

    # Phase GA : croisement et mutation
    ga_population = []
    for _ in range(pop_size):
        p1 = population[np.random.randint(pop_size)]
        p2 = population[np.random.randint(pop_size)]
        child = {}
        for device in devices.keys():
            child[device] = p1[device] if np.random.rand() < 0.5 else p2[device]
            if np.random.rand() < 0.1:  # mutation 10%
                mutation = np.random.randint(-300, 300)  # ±5 minutes
                child[device] += mutation
            low = devices[device]["min_eta"]
            high = devices[device]["max_eta"]
            child[device] = int(np.clip(child[device], low, high))
        ga_population.append(child)

    # Combinaison et sélection
    population = new_population + ga_population
    population = sorted(population, key=lambda sol: fitness(sol, devices))[:pop_size]
    fitness_values = [fitness(sol, devices) for sol in population]

# --- 5. Affichage du résultat ---
best_idx = np.argmin(fitness_values)
best_sol = population[best_idx]
best_fit = fitness_values[best_idx]

print("Meilleure solution trouvée :")
for device, eta in best_sol.items():
    h = eta // 3600
    m = (eta % 3600) // 60
    s = eta % 60
    print(f"{device}: {h:02d}:{m:02d}:{s:02d}")
print("Fitness =", best_fit)

# --- 6. Visualisation de l'évolution du fitness ---
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_history, marker='o')
plt.title("Évolution du meilleur fitness")
plt.xlabel("Itération")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()
