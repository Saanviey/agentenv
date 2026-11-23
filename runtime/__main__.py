import argparse, sys, logging
from pathlib import Path
from runtime.config import load_config, model_config, ConfigError
from runtime.types import Message
from runtime.model.factory import build_client
from runtime.model.base import ModelError
from runtime.tools import default_tools
from runtime.loop import run_agent, SYSTEM_PROMPT
from runtime.policy import ask_user, approve_all
from runtime.trace import Tracer, new_trace
from runtime.verify import CodingVerifier
from runtime.context import from_config as context_from_config
from runtime.store import Store, RESUMABLE
from runtime.checkpoint import checkpointer, decode
from runtime.memory import memory_path, prompt_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # logs to stderr
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide httpx's per-request lines; the loop logs each step
log = logging.getLogger("runtime")

# In: CLI args. Out: final answer on stdout, run stats in log.
# Exit codes: 0 ok, 1 model failed (resumable), 2 config / usage error, 130 interrupted with Ctrl-C (resumable).
def main():
    ap = argparse.ArgumentParser(prog="python -m runtime")  # argparse: turns sys.argv into named options + --help
    ap.add_argument("task", nargs="*", help="what the agent should do (with --continue: the follow-up)")
    ap.add_argument("--workspace", help="folder the agent may read/change (default: [agent] workspace)")
    ap.add_argument("--yes", action="store_true", help="approve every action the policy allows, without asking")
    ap.add_argument("--max-steps", type=int, help="override [agent] max_steps")
    ap.add_argument("--model", help="model profile from config.toml (default: its `model`); a resumed run may switch model")
    ap.add_argument("--resume", metavar="RUN", help="continue an interrupted or crashed run: its id, or 'last'")
    ap.add_argument("--continue", dest="follow_up", metavar="RUN", help="ask a follow-up in an earlier run's conversation: its id, or 'last'")
    ap.add_argument("--runs", action="store_true", help="list recent runs and exit")
    ap.add_argument("--no-memory", action="store_true", help="don't load or save project memory")
    args = ap.parse_args()
    cfg = load_config()
    if args.model:
        cfg["model"] = args.model
    a = cfg.get("agent", {})
    state_dir = Path(a.get("state_dir", "state"))
    store = Store(state_dir / "agent.db")
    if args.runs:
        for r in store.recent():
            print(f"{r['id']:62} {r['status']:12} step {r['step']:<3} {r['updated']}")
        return

    prev = None
    if args.resume or args.follow_up:
        prev = _find_run(store, args.resume or args.follow_up, resuming=bool(args.resume))
        if args.follow_up and not args.task:
            sys.exit("--continue needs the follow-up question, e.g. --continue last \"and what about tax.py?\"")
    task = prev["task"] if args.resume else (" ".join(args.task) or "Say hi in one line.")
    workspace = prev["workspace"] if prev else (args.workspace or a.get("workspace", "."))
    trace = Tracer(Path(prev["trace"])) if prev else new_trace(a.get("trace_dir", "traces"), task)  # a resumed run appends to its trace
    run_id = prev["id"] if prev else trace.path.stem
    try:
        m = model_config(cfg)
        client = build_client(m, on_retry=lambda attempt, delay, e: trace.event(
            "retry", attempt=attempt, delay_s=delay, error_type=type(e).__name__, error=str(e)),
            on_throttle=lambda wait, used, need: trace.event("throttle", wait_s=round(wait, 1), used=used, need=need))
    except ConfigError as e:
        log.error("config: %s", e)
        sys.exit(2)
    mem = None if args.no_memory else memory_path(workspace, state_dir)
    tools = default_tools(workspace, a.get("command_timeout_s", 30), memory_file=mem)
    approve = approve_all if args.yes or a.get("approval") == "auto" else ask_user
    context = context_from_config(m, a, client)
    max_steps = args.max_steps or a.get("max_steps", 10)
    log.info("run %s | model %s via %s | workspace %s | approval %s", run_id, m["name"], m["provider"], workspace,
             "auto" if approve is approve_all else "ask")

    state, messages = None, []
    if prev:
        state = decode(prev["state"], tools, context)  # messages, counters, Session (snapshots!), context budget
        if args.follow_up:  # same conversation, new question: fresh step count and counters
            state.messages.append(Message("user", task))
            state.step = state.reminders = state.rejections = 0
            state.seen, state.pending = {}, None
        trace.event("run_resumed" if args.resume else "run_continued", step=state.step, task=task, model=m["name"])
        log.info("%s at step %d", "resuming" if args.resume else "continuing", state.step)
    else:
        block = prompt_block(mem)
        messages = [Message("system", SYSTEM_PROMPT + (f"\n\n{block}" if block else "")), Message("user", task)]
        store.create(run_id, str(Path(workspace).resolve()), task, m["name"], str(trace.path))
        trace.event("run_start", task=task, model=m["name"], provider=m["provider"],  # never the API key
                    workspace=str(Path(workspace).resolve()), approval="auto" if approve is approve_all else "ask",
                    max_steps=max_steps, tools=tools.names(), schemas=tools.schemas(), memory=bool(block))

    try:
        verifier = CodingVerifier(a.get("test_command") or None, a.get("command_timeout_s", 30) * 4)
        result = run_agent(client, tools, messages, max_steps, approve=approve, trace=trace, verifier=verifier,
                           rollback=a.get("rollback", True), context=context,
                           checkpoint=checkpointer(store, run_id, tools, context), state=state)
    except KeyboardInterrupt:
        log.warning("interrupted. Continue with:  python -m runtime --resume %s", run_id)
        sys.exit(130)
    except ModelError as e:
        log.error("model call failed: %s", e)
        log.info("progress is saved. Continue later with:  python -m runtime --resume %s", run_id)
        sys.exit(1)
    finally:
        trace.close()
        store.close()

    print(result.answer)
    log.info("done: stopped=%s steps=%d in=%d out=%d", result.stopped, result.steps, result.input_tokens, result.output_tokens)
    log.info("trace: %s  (view: python -m runtime.trace view)  follow-up: python -m runtime --continue %s \"...\"",
             trace.path, run_id)

# In: a run id or "last". Out: the run's row. Exits with a clear message if it can't be used.
def _find_run(store: Store, ref: str, resuming: bool) -> dict:
    run_id = store.latest(resumable_only=resuming) if ref == "last" else ref
    run = store.get(run_id) if run_id else None
    if run is None or run["state"] is None:
        sys.exit(f"no {'resumable ' if resuming else ''}run found for {ref!r}. See: python -m runtime --runs")
    if resuming and run["status"] not in RESUMABLE:
        sys.exit(f"run {run_id} already ended ({run['status']}); ask a follow-up with --continue {run_id} \"...\"")
    return run

if __name__ == "__main__":
    main()
