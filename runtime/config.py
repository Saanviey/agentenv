import tomllib
from pathlib import Path

# In: path to TOML (relative to where you run). Out: nested dict, e.g. cfg["model"]["name"].
def load_config(path: str = "config.toml") -> dict:
    return tomllib.loads(Path(path).read_text())
