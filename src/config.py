"""
Central config loader. Reads config.yaml and exposes a typed settings object.
"""
import yaml
from pathlib import Path
from functools import lru_cache


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
