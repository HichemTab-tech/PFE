# solvers/__init__.py
from .csa_solver import CsaSolver
from .ga_solver import GaSolver

def SolverFactory(name: str, **kwargs):
    name = name.lower()
    if name == 'csa':
        return CsaSolver(**kwargs)
    elif name in ('ga', 'genetic', 'geneticalgorithm'):
      # pull out GA hyperparams (or use defaults)
        return GaSolver(
            pop_size      = kwargs.get('pop_size', 50),
            generations = kwargs.get('generations', 100),
            crossover_rate = kwargs.get('crossover_rate', 0.8),
            mutation_rate = kwargs.get('mutation_rate', 0.1),
        )
    else:
        raise ValueError(f"Unknown solver: {name}")