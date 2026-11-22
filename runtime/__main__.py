import argparse, sys, logging
from pathlib import Path
from runtime.config import load_config, model_config, ConfigError
from runtime.types import Message
from runtime.model.factory import build_client
from runtime.model.base import ModelError
from runtime.tools import default_tools
from runtime.loop import run_agent, SYSTEM_PROMPT
from runtime.policy import ask_user, approve_all
from runtime.trace import new_trace
from runtime.verify import CodingVerifier
from runtime.context import from_config as context_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # logs to stderr
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide httpx's per-request lines; the loop logs each step
log = logging.getLogger("runtime")

# In: CLI args. Out: final answer on stdout, run stats in log; exit 1 on model failure, 2 on config error.
def main():
    ap = argparse.ArgumentParser(prog="python -m runtime")  # argparse: turns sys.argv into named options + --help
    ap.add_argument("task", nargs="*", help="what the agent should do")
    ap.add_argument("--workspace", help="folder the agent may read/change (default: [agent] workspace)")
    ap.add_argument("--yes", action="store_true", help="approve every action the policy allows, without asking")
    args = ap.parse_args()
    task = " ".join(args.task) or "Say hi in one line."
    cfg = load_config()
    a = cfg.get("agent", {})
    trace = new_trace(a.get("trace_dir", "traces"), task)
    try:
        m = model_config(cfg)
        client = build_client(m, on_retry=lambda attempt, delay, e: trace.event(
            "retry", attempt=attempt, delay_s=delay, error_type=type(e).__name__, error=str(e)),
            on_throttle=lambda wait, used, need: trace.event("throttle", wait_s=round(wait, 1), used=used, need=need))
    except ConfigError as e:
        log.error("config: %s", e)
        sys.exit(2)
    log.info("model: %s via %s", m["name"], m["provider"])
    workspace = args.workspace or a.get("workspace", ".")
    tools = default_tools(workspace, a.get("command_timeout_s", 30))
    approve = approve_all if args.yes or a.get("approval") == "auto" else ask_user
    log.info("workspace: %s, approval: %s", workspace, "auto" if approve is approve_all else "ask")
    trace.event("run_start", task=task, model=m["name"], provider=m["provider"],  # never the API key
                workspace=str(Path(workspace).resolve()),
                approval="auto" if approve is approve_all else "ask", max_steps=a.get("max_steps", 10),
                tools=tools.names(), schemas=tools.schemas())

    messages = [
        Message("system", SYSTEM_PROMPT),
        Message("user", task),
    ]
    try:
        verifier = CodingVerifier(a.get("test_command") or None, a.get("command_timeout_s", 30) * 4)
        result = run_agent(client, tools, messages, a.get("max_steps", 10), approve=approve, trace=trace,
                           verifier=verifier, rollback=a.get("rollback", True), context=context_from_config(m, a, client))
    except ModelError as e:
        log.error("model call failed: %s", e)
        log.info("trace: %s", trace.path)
        sys.exit(1)
    finally:
        trace.close()

    print(result.answer)
    log.info("done: stopped=%s steps=%d in=%d out=%d",
             result.stopped, result.steps, result.input_tokens, result.output_tokens)
    log.info("trace: %s  (view: python -m runtime.trace view)", trace.path)

if __name__ == "__main__":
    main()
