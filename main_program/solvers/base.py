from typing import Dict, Any, Sequence
import pandas as pd

class Solver:
    def run(self,
            params: Dict[str, Dict[str, Any]],
            price_profile: pd.Series,
            load_profile: pd.Series,
            seed: int = None
    ) -> Dict[str, int]:
        """Return a dict mapping device→start_hour."""
        raise NotImplementedError