import argparse, fnmatch, json, shlex, shutil, subprocess, sys, tempfile, time, tomllib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from runtime.config import load_config, model_config, ConfigError
from runtime.loop import run_agent, SYSTEM_PROMPT
from runtime.model.base import ModelClient, ModelError
from runtime.policy import approve_all
from runtime.tools import default_tools
from runtime.trace import Tracer, NULL
from runtime.types import Message
from runtime.verify import CodingVerifier
from runtime.context import ContextManager, from_config as context_from_config

TASKS_DIR = Path("evals/tasks")
RESULTS_DIR = Path("evals/results")
CHECK = [sys.executable, "-m", "unittest", "-q"]  # the pass/fail check; runs with the runtime's own Python

# One eval task: a small repo + what to ask the agent.
@dataclass
class Task:
    name: str
    prompt: str
    repo: Path
    protect: list[str]   # glob patterns of files restored before checking (the tests)
    max_steps: int

# Outcome of one task.
@dataclass
class TaskResult:
    task: str
    passed: bool          # check exited 0, with the original tests
    finished: bool        # agent ended by calling finish (didn't crash, run out of steps or give up)
    tampered: bool        # agent changed or deleted a protected file
    stopped: str          # finish / no_finish / max_steps / error
    steps: int
    input_tokens: int
    output_tokens: int
    seconds: float
    error: str = ""
    check_output: str = ""
    trace: str = ""

# In: task names to keep (None = all). Out: tasks sorted by name.
def load_tasks(names: list[str] | None = None, tasks_dir: Path = TASKS_DIR) -> list[Task]:
    tasks = []
    for d in sorted(p for p in tasks_dir.iterdir() if (p / "task.toml").exists()):
        if names and d.name not in names:
            continue
        t = tomllib.loads((d / "task.toml").read_text())
        tasks.append(Task(d.name, t["prompt"], d / "repo", t.get("protect", ["test_*.py"]), t.get("max_steps", 15)))
    return tasks

def _protected(repo: Path, patterns: list[str]) -> list[Path]:
    return [f.relative_to(repo) for f in repo.rglob("*") if f.is_file() and any(fnmatch.fnmatch(f.name, p) for p in patterns)]

def _check(ws: Path) -> subprocess.CompletedProcess:
    return subprocess.run(CHECK, cwd=ws, capture_output=True, text=True, timeout=120)

# In: a task + a model client. Out: TaskResult. Runs the agent on a fresh copy of the repo, then checks it.
def run_task(task: Task, client: ModelClient, trace: Tracer = NULL, command_timeout_s: float = 30,
             context: ContextManager | None = None) -> TaskResult:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "repo"
        shutil.copytree(task.repo, ws)
        tools = default_tools(ws, command_timeout_s)
        trace.event("run_start", task=task.prompt, eval_task=task.name, model="", provider="", workspace=str(ws),
                    approval="auto", max_steps=task.max_steps, tools=tools.names())
        t0 = time.perf_counter()
        stopped, steps, tin, tout, error = "error", 0, 0, 0, ""
        try:
            r = run_agent(client, tools, [Message("system", SYSTEM_PROMPT), Message("user", task.prompt)],
                          task.max_steps, approve=approve_all, trace=trace,
                          verifier=CodingVerifier(shlex.join(CHECK)), rollback=True, context=context)
            stopped, steps, tin, tout = r.stopped, r.steps, r.input_tokens, r.output_tokens
        except ModelError as e:
            error = f"{type(e).__name__}: {e}"
            p = getattr(e, "progress", {})
            steps, tin, tout = p.get("steps", 0), p.get("input_tokens", 0), p.get("output_tokens", 0)
        seconds = time.perf_counter() - t0

        # Undo any change to the tests, so a task only passes if the SOURCE is fixed.
        # Tampering = the agent changed a protected file at any point (session record; rollback may have undone it since)
        # or it differs now (commands like `mv` bypass the session).
        touched = set(tools.session.snapshots) if tools.session else set()
        tampered = any(any(fnmatch.fnmatch(Path(rel).name, p) for p in task.protect) for rel in touched)
        for rel in _protected(task.repo, task.protect):
            orig, now = task.repo / rel, ws / rel
            if not now.exists() or now.read_bytes() != orig.read_bytes():
                tampered = True
                now.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(orig, now)
        check = _check(ws)
        out = (check.stdout + check.stderr).strip()
        trace.event("eval_check", passed=check.returncode == 0, tampered=tampered, output=out[-2000:])
        return TaskResult(task.name, check.returncode == 0, stopped == "finish", tampered, stopped, steps, tin, tout,
                          round(seconds, 1), error, out[-2000:], str(trace.path or ""))

# Sanity check: every task's tests must FAIL before the agent touches it, or the task proves nothing.
def validate(tasks: list[Task]) -> bool:
    ok = True
    for t in tasks:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "repo"
            shutil.copytree(t.repo, ws)
            fails = _check(ws).returncode != 0
        print(f"{t.name:22} {'ok (fails before fix)' if fails else 'BAD: tests already pass'}")
        ok &= fails
    return ok

def summarize(results: list[TaskResult]) -> dict:
    n = len(results)
    passed = [r for r in results if r.passed]
    clean = [r for r in passed if r.finished and not r.tampered]  # fixed it AND ended properly AND didn't touch tests
    return {"tasks": n, "passed": len(passed), "success_rate": round(len(passed) / n, 3) if n else 0,
            "clean": len(clean), "clean_rate": round(len(clean) / n, 3) if n else 0,
            "tampered": sum(r.tampered for r in results),
            "avg_steps_passed": round(sum(r.steps for r in passed) / len(passed), 1) if passed else None,
            "input_tokens": sum(r.input_tokens for r in results), "output_tokens": sum(r.output_tokens for r in results),
            "seconds": round(sum(r.seconds for r in results), 1)}

def print_table(results: list[TaskResult], summary: dict) -> None:
    print(f"\n{'task':22} {'result':7} {'stopped':10} {'steps':>5} {'tokens':>8} {'time':>7}")
    for r in results:
        res = ("PASS" if r.finished else "PASS*") if r.passed else "FAIL"
        print(f"{r.task:22} {res:7} {r.stopped:10} {r.steps:5} {r.input_tokens + r.output_tokens:8} {r.seconds:6.1f}s"
              + ("  TAMPERED" if r.tampered else "") + (f"  {r.error[:60]}" if r.error else ""))
    s = summary
    print(f"\nsuccess {s['passed']}/{s['tasks']} ({s['success_rate']:.0%})  clean {s['clean']}/{s['tasks']} ({s['clean_rate']:.0%})  avg steps (passed) {s['avg_steps_passed']}  "
          f"tokens {s['input_tokens'] + s['output_tokens']}  time {s['seconds']}s  tampered {s['tampered']}")

def main():
    ap = argparse.ArgumentParser(prog="python -m runtime.evals", description="run the agent on eval tasks and score it")
    ap.add_argument("command", choices=["run", "validate", "history"])
    ap.add_argument("--model", help="model profile from config.toml (default: config's `model`)")
    ap.add_argument("--tasks", help="comma-separated task names (default: all)")
    ap.add_argument("--pause", type=float, default=0, help="seconds to wait between tasks (rate limits)")
    ap.add_argument("--max-context", type=int, help="override the model profile's max_context_tokens (experiments)")
    a = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    tasks = load_tasks(a.tasks.split(",") if a.tasks else None)

    if a.command == "validate":
        sys.exit(0 if validate(tasks) else 1)
    if a.command == "history":
        for f in sorted(RESULTS_DIR.glob("*.json")):
            d = json.loads(f.read_text())
            s = d["summary"]
            print(f"{f.stem:28} {d['model']:28} {s['passed']:2}/{s['tasks']:<2} ({s['success_rate']:.0%})  clean {s.get('clean', '?')}  tokens {s['input_tokens'] + s['output_tokens']}")
        return

    from runtime.model.factory import build_client
    cfg = load_config()
    if a.model:
        cfg["model"] = a.model
    try:
        m = model_config(cfg)
    except ConfigError as e:
        sys.exit(f"config: {e}")
    if a.max_context:
        m["max_context_tokens"] = a.max_context
    current = {"trace": NULL}  # retries are written to whichever task's trace is active
    client = build_client(m, on_retry=lambda attempt, delay, e: current["trace"].event(
        "retry", attempt=attempt, delay_s=delay, error_type=type(e).__name__, error=str(e)[:300]),
        on_throttle=lambda wait, used, need: current["trace"].event("throttle", wait_s=round(wait, 1), used=used, need=need))
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['model']}"
    trace_dir = Path(cfg.get("agent", {}).get("trace_dir", "traces")) / "evals" / run_id
    print(f"eval {run_id}: {len(tasks)} tasks, model {m['name']} via {m['provider']}")

    results, aborted = [], ""
    for i, t in enumerate(tasks):
        if i and a.pause:
            time.sleep(a.pause)
        tr = current["trace"] = Tracer(trace_dir / f"{t.name}.jsonl")
        agent_cfg = cfg.get("agent", {})
        r = run_task(t, client, tr, agent_cfg.get("command_timeout_s", 30), context_from_config(m, agent_cfg, client))  # fresh per task
        tr.close()
        results.append(r)
        if r.error.startswith("ModelQuotaExceeded"):  # B29: the rest would fail the same way; stop and keep what we have
            print(f"  STOPPED: the provider's daily limit is used up. {len(tasks) - i - 1} task(s) not run; partial results saved.")
            aborted = f"daily limit reached after {i + 1} of {len(tasks)} tasks"
            break
        print(f"  {t.name:22} {('PASS' if r.finished else 'PASS*') if r.passed else 'FAIL'}  {r.stopped:10} {r.steps:2} steps  {r.seconds:5.1f}s"
              + ("  TAMPERED" if r.tampered else "") + (f"  {r.error[:60]}" if r.error else ""))

    summary = summarize(results)
    print_table(results, summary)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_id}.json"
    out.write_text(json.dumps({"run_id": run_id, "model": m["name"], "provider": m["provider"],
                               "summary": summary, "aborted": aborted, "results": [asdict(r) for r in results]}, indent=2))
    print(f"PASS* = tests pass but the agent didn't finish cleanly (crashed / ran out of steps)\nresults: {out}\ntraces:  {trace_dir}/")

if __name__ == "__main__":
    main()
