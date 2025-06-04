# solvers/csa_solver.py
import numpy as np

from .base import Solver


class CsaSolver(Solver):
    def __init__(
            self,
            fitness,
            params: dict,
    ):
        self.fitness = fitness
        self.params = params
        self.devices = list(self.params['α'].keys())

    def run(self, devices_to_schedule, seed=None, max_iter=100):
        # Pass max_iter directly to run_csa
        return self.run_csa(devices_to_schedule, seed=seed, max_iter=max_iter)

    def run_csa(self, devices, n_crows=150, P_aw=0.3, seed=None, FL=1, max_iter=100):
        # max_iter is now correctly received and used.
        # Removed the redundant hardcoded `max_iter=100` that was overriding the passed argument.

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

        for _ in range(max_iter):  # Loop now correctly uses the received max_iter
            for k in range(n_crows):
                j = np.random.choice([i for i in range(n_crows) if i != k])
                if np.random.rand() > P_aw:
                    η_new = {
                        d: memories[j][d] if np.random.rand() < FL else population[k][d]
                        for d in devices
                    }
                else:
                    η_new = {d: int(np.random.choice(self.params['valid_hours'][d])) for d in devices}

                for d in η_new:
                    if self.params['W'][d] == 0:
                        η_new[d] = self.params['α'][d]

                if self.fitness(η_new) < self.fitness(memories[k]):
                    memories[k] = η_new.copy()
                    population[k] = η_new.copy()
                    f_new = self.fitness(η_new)
                    if f_new < best_fit:
                        best_fit = f_new
                        best_mem = η_new.copy()

        return best_mem