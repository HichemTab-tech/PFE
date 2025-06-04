# solvers/base.py
from typing import Dict, Any, Sequence, Tuple, List

class Solver:
    def run(self,
            devices_to_schedule: List[str], # Renamed for clarity to distinguish from self.params
            seed: int = None,
            max_iter: int = 100
    ) -> Tuple[Dict[str, int], List[float]]: # Now returns best schedule AND fitness history
        """Return a tuple: (a dict mapping device→start_hour, a list of fitness values over iterations)."""
        raise NotImplementedError