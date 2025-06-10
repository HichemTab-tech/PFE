# solvers/csa_solver.py
import numpy as np
from typing import Callable # Added for repair_function typing

from .base import Solver

class CsaSolver(Solver):
    def __init__(
        self,
        fitness,
        params: dict, # This `params` dictionary contains α, β, LOT, P, W, L, M, valid_hours (now valid_slots)
        repair_function: Callable = None, # ADDED: Repair function for constraints
    ):
        self.fitness = fitness
        self.params = params
        # ADDED: Store repair function, defaulting to an identity function if None
        self.repair_function = repair_function if repair_function else lambda s: s
        # Extract device names once from any of the parameter dictionaries
        # All α, β, etc. should have the same set of keys (device names)
        self.devices = list(self.params['α'].keys())

    # Corrected 'run' signature: it now expects 'devices' (list of names) not 'params'
    def run(self, devices_to_schedule: list, seed: int = None, max_iter: int = 100):  # Updated default max_iter
        # delegate directly to run_csa, passing the list of device names
        return self.run_csa(devices_to_schedule, seed=seed, max_iter=max_iter)

    def run_csa(self, devices: list,
                # This 'devices' parameter is the list of device names (e.g., ["Dishwasher [kW]", ...])
                n_crows: int = 150, P_aw: float = 0.3, seed: int = None, FL: int = 1,
                max_iter: int = 100):  # max_iter added here

        if seed is not None:
            np.random.seed(seed)

        population = []
        memories = []
        for _ in range(n_crows):
            η = {d: (self.params['α'][d] if self.params['W'][d] == 0 else int(
                np.random.choice(self.params['valid_hours'][d]))) for d in devices}
            population.append(η.copy())
            memories.append(η.copy())

        best_mem = min(memories, key=self.fitness)
        best_fit = self.fitness(best_mem)
        fitness_history = [best_fit]  # Initialize fitness history

        for _ in range(max_iter):
            for k in range(n_crows):
                j = np.random.choice([i for i in range(n_crows) if i != k])
                if np.random.rand() > P_aw:
                    # η_new = {
                    #     d: memories[j][d] if np.random.rand() < FL else population[k][d]
                    #     for d in devices
                    # }
                    η_new = {d: memories[j][d] for d in devices}
                else:
                    η_new = {d: int(np.random.choice(self.params['valid_hours'][d])) for d in devices}

                # enforce immovable (w=0) → lock at α[d]
                for d in η_new:
                    if self.params['W'][d] == 0:
                        η_new[d] = self.params['α'][d]  # Use alpha_slot

                # ADDED: Repair the new schedule to satisfy hard constraints (like picLimit)
                η_new = self.repair_function(η_new)

                if self.fitness(η_new) < self.fitness(memories[k]):
                    memories[k] = η_new.copy()
                    population[k] = η_new.copy()
                    f_new = self.fitness(η_new)
                    if f_new < best_fit:
                        best_fit = f_new
                        best_mem = η_new.copy()
            fitness_history.append(best_fit)  # Append best fitness of current iteration

        return best_mem, fitness_history  # Return schedule AND history