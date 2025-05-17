import random
from .base import Solver

class GaSolver(Solver):
    def __init__(
        self,
        pop_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.05,
    ):
        self.pop_size       = pop_size
        self.generations    = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate

    def run(
        self,
        params: dict[str, dict],
        price_profile,      # pd.Series indexed by hour
        load_profile=None,  # pd.Series if you later add load penalties
        seed: int = None
    ) -> dict[str, int]:
        """Evolve a population of schedules and return the best one."""
        if seed is not None:
            random.seed(seed)

        # 1. Initialize population
        population = [self._random_solution(params) for _ in range(self.pop_size)]

        best_sol, best_fit = None, float('-inf')
        for gen in range(self.generations):
            # 2. Evaluate fitnesses
            fitnesses = [
                self._fitness(sol, params, price_profile, load_profile)
                for sol in population
            ]

            # 3. Track the global best
            for sol, fit in zip(population, fitnesses):
                if fit > best_fit:
                    best_fit, best_sol = fit, sol

            # 4. Create next generation
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                # 5. Crossover
                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # 6. Mutation
                self._mutate(c1, params)
                self._mutate(c2, params)

                new_pop.extend([c1, c2])

            population = new_pop[: self.pop_size]

        return best_sol

    def _random_solution(self, params):
        sol = {}
        for device, p in params.items():
            lo = p['alpha'] // 3600
            hi = p['beta']  // 3600
            # clamp to avoid empty range
            if hi < lo:
                hi = lo
            sol[device] = random.randint(lo, hi)
        return sol

    def _fitness(self, sol, params, price_profile, load_profile):
        total_cost = 0.0
        for device, start_h in sol.items():
            duration_h = params[device]['LOT'] / 3600
            rate       = price_profile.get(start_h, price_profile.mean())
            weight     = params[device].get('lambda', 1)
            total_cost += duration_h * rate * weight
        return -total_cost

    def _tournament_selection(self, pop, fits, k: int = 3):
        competitors = random.sample(list(zip(pop, fits)), k)
        return max(competitors, key=lambda cf: cf[1])[0]

    def _crossover(self, p1, p2):
        keys = list(p1.keys())
        pt   = random.randint(1, len(keys) - 1)
        c1, c2 = {}, {}
        for i, device in enumerate(keys):
            if i < pt:
                c1[device], c2[device] = p1[device], p2[device]
            else:
                c1[device], c2[device] = p2[device], p1[device]
        return c1, c2

    def _mutate(self, indiv, params):
        for device, p in params.items():
            if random.random() < self.mutation_rate:
                lo = p['alpha'] // 3600
                hi = p['beta']  // 3600
                if hi < lo:
                    hi = lo
                indiv[device] = random.randint(lo, hi)
