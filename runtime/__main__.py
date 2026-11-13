import sys, logging
from runtime.config import load_config
from runtime.types import Message
from runtime.model.ollama import OllamaClient
from runtime.model.base import ModelError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # logs to stderr; also shows httpx's request lines
log = logging.getLogger("runtime")

# In: CLI args as the task. Out: reply on stdout, stats in log; exit 1 on model failure.
def main():
    task = " ".join(sys.argv[1:]) or "Say hi in one line."  # argv[0] is the script itself
    cfg = load_config()["model"]
    client = OllamaClient(cfg["name"], cfg["url"], cfg["timeout_s"])

    messages = [
        Message("system", "You are a concise assistant."),
        Message("user", task),
    ]
    try:
        resp = client.complete(messages)
    except ModelError as e:
        log.error("model call failed: %s", e)
        sys.exit(1)

    print(resp.message.content)
    log.info("model=%s in=%d out=%d stop=%s %.0fms",
             resp.model, resp.input_tokens, resp.output_tokens,
             resp.stop_reason, resp.latency_ms)

if __name__ == "__main__":
    main()
