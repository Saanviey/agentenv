import os, tomllib
from pathlib import Path

class ConfigError(Exception): ...  # bad or missing settings; message says what to fix

# In: path to TOML (relative to where you run). Out: nested dict, e.g. cfg["agent"]["max_steps"].
def load_config(path: str = "config.toml") -> dict:
    return tomllib.loads(Path(path).read_text())

# In: full config. Out: the active model profile, with "api_key" filled in from the env var it names.
def model_config(cfg: dict) -> dict:
    name = cfg.get("model")
    profiles = cfg.get("models", {})
    if name not in profiles:
        raise ConfigError(f"model = {name!r} but config has no [models.{name}]. available: {', '.join(profiles)}")
    m = dict(profiles[name])
    if env := m.get("api_key_env"):
        m["api_key"] = os.environ.get(env)  # os.environ: the shell's environment variables
        if not m["api_key"]:
            raise ConfigError(f"model {name!r} needs an API key: run  export {env}=<your key>")
    return m
