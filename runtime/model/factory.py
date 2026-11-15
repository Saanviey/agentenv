from runtime.model.base import ModelClient
from runtime.config import ConfigError

# In: one model profile from model_config(). Out: the matching client.
def build_client(m: dict) -> ModelClient:
    if m["provider"] == "ollama":
        from runtime.model.ollama import OllamaClient
        return OllamaClient(m["name"], m["url"], m["timeout_s"])
    if m["provider"] == "openai_compat":
        from runtime.model.openai_compat import OpenAICompatClient
        return OpenAICompatClient(m["name"], m["url"], m["timeout_s"], m.get("api_key"))
    raise ConfigError(f"unknown provider {m['provider']!r}; use 'ollama' or 'openai_compat'")
