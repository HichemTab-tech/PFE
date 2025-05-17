# solvers/__init__.py
from .csa_solver import CsaSolver
from .ga_solver import GaSolver

def SolverFactory(name: str, **kwargs):
    name = name.lower()
    if name == 'csa':
        return CsaSolver(**kwargs)
    elif name in ('ga', 'genetic', 'geneticalgorithm'):
      # pull out GA hyperparams (or use defaults)
        return GaSolver(**kwargs)
    else:
        raise ValueError(f"Unknown solver: {name}")