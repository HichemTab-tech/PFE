import sys
import numpy as np
sys.path.append('../../main_program')

from .base import Solver

class CsaSolver(Solver):
    def __init__(
        self,
        fitness,
        params: dict,
    ):
        self.fitness = fitness
        self.params = params



    def run(self, params, seed=None):
        # delegate directly to your existing function
        return self.run_csa(params, seed=seed)

    def run_csa(self, devices,
                n_crows=30, max_iter=500, P_aw=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # init
        population = []
        memories = []
        for _ in range(n_crows):
            # η = {d: int(np.random.choice(valid_hours[d])) for d in devices}
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
                        η_new[d] = self.params['α'][d]

                if self.fitness(η_new) < self.fitness(memories[k]):
                    memories[k] = η_new.copy()
                    population[k] = η_new.copy()
                    f_new = self.fitness(η_new)
                    if f_new < best_fit:
                        best_fit = f_new
                        best_mem = η_new.copy()

        return best_mem