import sys
import random
sys.path.append('../../main_program')

from .base import Solver

class GaSolver(Solver):
    def __init__(
        self,
        fitness,
        params: dict,
        pop_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
    ):
        # fitness: a callable mapping schedule→cost (lower is better)
        # params: dict with keys 'α', 'beta', 'valid_hours', 'W', etc.
        self.fitness       = fitness
        self.params        = params
        self.pop_size      = pop_size
        self.generations   = generations
        self.crossover_rate= crossover_rate
        self.mutation_rate = mutation_rate

    def run(self, devices, seed=None):
        return self._run_ga(devices, seed=seed)

    def _run_ga(self, devices, seed=None):
        if seed is not None:
            random.seed(seed)

        # 1. Initialize population of schedules
        population = []
        for _ in range(self.pop_size):
            indiv = {}
            for d in devices:
                if self.params['W'][d] == 0:
                    # immovable device locked at α[d]
                    indiv[d] = self.params['α'][d]
                else:
                    indiv[d] = int(random.choice(self.params['valid_hours'][d]))
            population.append(indiv.copy())

        # 2. Find initial best
        best       = min(population, key=self.fitness)
        best_score = self.fitness(best)

        # 3. Evolve for given generations
        for _ in range(self.generations):
            new_pop = []

            # pair off to produce pop_size children
            for _ in range(self.pop_size // 2):
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)

                # crossover
                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # mutate
                self._mutate(c1)
                self._mutate(c2)

                new_pop.extend([c1, c2])

            # if odd, carry the best over
            if len(new_pop) < self.pop_size:
                new_pop.append(best.copy())

            population = new_pop[: self.pop_size]

            # update best individual
            for indiv in population:
                score = self.fitness(indiv)
                if score < best_score:
                    best_score, best = score, indiv.copy()

        return best

    def _tournament_selection(self, population, k: int = 3):
        """Pick k random schedules and return the one with lowest cost."""
        contenders = random.sample(population, k)
        return min(contenders, key=self.fitness)

    def _crossover(self, p1, p2):
        """Single-point crossover over the list of devices."""
        keys = list(p1.keys())
        pt   = random.randint(1, len(keys) - 1)
        c1, c2 = {}, {}
        for i, d in enumerate(keys):
            if i < pt:
                c1[d], c2[d] = p1[d], p2[d]
            else:
                c1[d], c2[d] = p2[d], p1[d]
        return c1, c2

    def _mutate(self, indiv):
        """For each device, with mutation_rate prob, reassign a valid hour."""
        for d in indiv:
            if self.params['W'][d] != 0 and random.random() < self.mutation_rate:
                indiv[d] = int(random.choice(self.params['valid_hours'][d]))