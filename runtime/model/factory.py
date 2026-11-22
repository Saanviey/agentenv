from runtime.model.base import ModelClient
from runtime.model.retry import RetryingClient
from runtime.model.throttle import ThrottledClient
from runtime.config import ConfigError

# In: one model profile from model_config(). Out: the matching client, wrapped:
#   RetryingClient( ThrottledClient( backend ) )  -- every retry also waits for the per-minute budget.
def build_client(m: dict, on_retry=None, on_throttle=None) -> ModelClient:
    client = _backend(m)
    if m.get("tokens_per_minute"):
        client = ThrottledClient(client, m["tokens_per_minute"], on_wait=on_throttle)
    return RetryingClient(client, attempts=m.get("retries", 3), on_retry=on_retry)

def _backend(m: dict) -> ModelClient:
    if m["provider"] == "ollama":
        from runtime.model.ollama import OllamaClient
        return OllamaClient(m["name"], m["url"], m["timeout_s"], m.get("max_context_tokens"))
    if m["provider"] == "openai_compat":
        from runtime.model.openai_compat import OpenAICompatClient
        return OpenAICompatClient(m["name"], m["url"], m["timeout_s"], m.get("api_key"))
    raise ConfigError(f"unknown provider {m['provider']!r}; use 'ollama' or 'openai_compat'")
