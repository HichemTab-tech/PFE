# solvers/csa_solver.py
import sys
import numpy as np
# sys.path.append('../../main_program') # Consider if this path is truly necessary; remove if not needed.

from .base import Solver

class CsaSolver(Solver):
    def __init__(
        self,
        fitness,
        params: dict, # This `params` dictionary contains α, β, LOT, P, W, L, M, valid_hours (now valid_slots)
    ):
        self.fitness = fitness
        self.params = params
        # Extract device names once from any of the parameter dictionaries
        # All α, β, etc. should have the same set of keys (device names)
        self.devices = list(self.params['α'].keys())

    # Corrected 'run' signature: it now expects 'devices' (list of names) not 'params'
    def run(self, devices_to_schedule, seed=None):
        # delegate directly to run_csa, passing the list of device names
        return self.run_csa(devices_to_schedule, seed=seed)

    def run_csa(self, devices, # This 'devices' parameter is the list of device names (e.g., ["Dishwasher [kW]", ...])
                n_crows=150, max_iter=500, P_aw=0.3, seed=None):
        if seed is not None:
            np.random.seed(seed)

        population = []
        memories = []
        for _ in range(n_crows):
            # η = {d: int(np.random.choice(valid_hours[d])) for d in devices}
            # Use self.params correctly for α (alpha_slot), W, valid_hours (valid_slots)
            η = {d: (self.params['α'][d] if self.params['W'][d] == 0 else int(np.random.choice(self.params['valid_hours'][d]))) for d in devices}
            population.append(η.copy())
            memories.append(η.copy())

        best_mem = min(memories, key=self.fitness)
        best_fit = self.fitness(best_mem)

        for _ in range(max_iter):
            for k in range(n_crows):
                j = np.random.choice([i for i in range(n_crows) if i != k])
                if np.random.rand() > P_aw:
                    η_new = {d: memories[j][d] for d in devices}
                else:
                    η_new = {d: int(np.random.choice(self.params['valid_hours'][d])) for d in devices}

                # enforce immovable (w=0) → lock at α[d]
                for d in η_new:
                    if self.params['W'][d] == 0:
                        η_new[d] = self.params['α'][d] # Use alpha_slot

                if self.fitness(η_new) < self.fitness(memories[k]):
                    memories[k] = η_new.copy()
                    population[k] = η_new.copy()
                    f_new = self.fitness(η_new)
                    if f_new < best_fit:
                        best_fit = f_new
                        best_mem = η_new.copy()

        return best_mem