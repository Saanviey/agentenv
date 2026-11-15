import sys, logging
from runtime.config import load_config, model_config, ConfigError
from runtime.types import Message
from runtime.model.factory import build_client
from runtime.model.base import ModelError
from runtime.tools import default_tools
from runtime.loop import run_agent, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # logs to stderr
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide httpx's per-request lines; the loop logs each step
log = logging.getLogger("runtime")

# In: CLI args as the task. Out: final answer on stdout, run stats in log; exit 1 on model failure.
def main():
    task = " ".join(sys.argv[1:]) or "Say hi in one line."  # argv[0] is the script itself
    cfg = load_config()
    a = cfg.get("agent", {})
    try:
        m = model_config(cfg)
        client = build_client(m)
    except ConfigError as e:
        log.error("config: %s", e)
        sys.exit(2)
    log.info("model: %s via %s", m["name"], m["provider"])
    tools = default_tools(a.get("workspace", "."))

    messages = [
        Message("system", SYSTEM_PROMPT),
        Message("user", task),
    ]
    try:
        result = run_agent(client, tools, messages, a.get("max_steps", 10))
    except ModelError as e:
        log.error("model call failed: %s", e)
        sys.exit(1)

    print(result.answer)
    log.info("done: stopped=%s steps=%d in=%d out=%d",
             result.stopped, result.steps, result.input_tokens, result.output_tokens)

if __name__ == "__main__":
    main()
