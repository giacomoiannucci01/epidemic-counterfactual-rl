"""Locate bundled inputs in a source checkout or an installed wheel."""
from pathlib import Path
import sys

_checkout = Path(__file__).resolve().parents[1]
ROOT = (_checkout if (_checkout / "data/england/pf_observations.csv").is_file()
        else Path(sys.prefix) / "share/epidemic-counterfactual-rl")
