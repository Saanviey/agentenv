import argparse, json, re, sys, time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

# Writes one JSON object per line (JSONL) for every event in a run. Flushed after each line, so a crash keeps the trace.
class Tracer:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._t0 = time.perf_counter()
        self._f = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = path.open("a")

    # In: event name + any fields. Out: nothing; appends {"t", "ts", "event", ...fields} to the file.
    def event(self, kind: str, /, **data) -> None:  # `/`: kind is positional-only, so data may contain a "kind" field too
        if not self._f:
            return
        rec = {"t": round(time.perf_counter() - self._t0, 3), "ts": datetime.now().isoformat(timespec="milliseconds"),
               "event": kind, **data}
        self._f.write(json.dumps(rec, default=_jsonable, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None

NULL = Tracer()  # writes nothing; the default, so tracing is optional everywhere

# In: folder + task. Out: a Tracer writing to <folder>/<YYYYmmdd-HHMMSS>-<task-slug>.jsonl
def new_trace(trace_dir: str | Path, task: str) -> Tracer:
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40] or "run"
    return Tracer(Path(trace_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.jsonl")

def _jsonable(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)

def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def latest(trace_dir: str | Path = "traces") -> Path:
    files = sorted(Path(trace_dir).glob("*.jsonl"))
    if not files:
        sys.exit(f"no traces in {trace_dir}/")
    return files[-1]


# ---------- viewer ----------

def _short(s, n):
    s = str(s)
    s = s if n is None or len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"
    return s.replace("\n", "\n             ")  # indent continuation lines under their label

def _args(a):
    if isinstance(a, dict):
        return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in a.items())
    return json.dumps(a, ensure_ascii=False)

# In: trace events. Out: a readable timeline (str). full=False shortens long text.
def render(events: list[dict], full: bool = False) -> str:
    n = None if full else 300
    out = []
    for e in events:
        k = e["event"]
        if k == "run_start":
            out += [f"RUN  {e.get('task')!r}",
                    f"     model={e.get('model')} via {e.get('provider')}  workspace={e.get('workspace')}  "
                    f"approval={e.get('approval')}  max_steps={e.get('max_steps')}",
                    f"     tools: {', '.join(e.get('tools', []))}"]
        elif k == "model_request":
            est = f", ~{e['est_tokens']} tokens" if e.get("est_tokens") else ""
            out.append(f"\n── step {e['step']}  (+{len(e['new_messages'])} new messages, {e['total_messages']} total{est})")
            if full:
                for m in e["new_messages"]:
                    out.append(f"     > [{m['role']}] {_short(m['content'], n)}")
        elif k == "model_response":
            m = e["message"]
            out.append(f"   model  {e['latency_ms']:.0f}ms  in={e['input_tokens']} out={e['output_tokens']}  stop={e['stop_reason']}")
            if m["content"]:
                out.append(f"     text: {_short(m['content'], n)}")
            for c in m["tool_calls"]:
                out.append(f"     call: {c['name']}({_short(_args(c['arguments']), n)})")
        elif k == "model_error":
            out.append(f"   MODEL ERROR {e['type']}: {_short(e['error'], n)}")
            if e.get("generation"):
                out.append(f"     model wrote: {_short(e['generation'], n)}")
        elif k == "compaction":
            out.append(f"   ✂ context {e['before']} → {e['after']} tokens (budget {e['budget']}): {e['method']}; "
                       f"{e['elided']} results elided, {e['summarized']} messages summarized")
            if e.get("summary"):
                out.append(f"     summary: {_short(e['summary'], n)}")
        elif k == "throttle":
            out.append(f"   ⏳ throttle: {e['used']} tokens used in the last minute, need {e['need']}; waited {e['wait_s']}s")
        elif k == "retry":
            out.append(f"   retry {e['attempt']}: {e['error_type']} ({_short(e['error'], 120)}), waiting {e['delay_s']}s")
        elif k == "reminder":
            out.append(f"   → reminder ({e['kind']})")
        elif k == "tool":
            gate = e["decision"] + (f" ({e['reason']})" if e.get("reason") else "")
            if e["decision"] == "ask":
                gate += " → approved" if e["approved"] else " → NOT approved"
            out.append(f"   tool  {e['name']}  policy: {gate}  {e['ms']:.0f}ms" + ("  [repeat]" if e.get("repeat") else ""))
            out.append(f"     result: {_short(e['result'], n)}")
        elif k == "run_end":
            out += [f"\nEND  stopped={e['stopped']}  steps={e['steps']}  in={e['input_tokens']} out={e['output_tokens']}  {e['t']:.1f}s",
                    f"     answer: {_short(e['answer'], n)}"]
        elif k == "run_error":
            out.append(f"\nCRASH {e['type']}: {e['error']}  (after {e['t']:.1f}s)")
    return "\n".join(out)


# ---------- replay ----------

# A model that returns the replies recorded in a trace, in order (and re-raises recorded model errors).
def replay_client(events: list[dict]):
    from runtime.types import Message, Response, ToolCall
    from runtime.model import base
    from runtime.model.scripted import ScriptedClient
    script = []
    for e in events:
        if e["event"] == "model_response":
            m = e["message"]
            msg = Message("assistant", m["content"], tool_calls=[ToolCall(c["name"], c["arguments"], c.get("id")) for c in m["tool_calls"]])
            script.append(Response(msg, e["stop_reason"], e["input_tokens"], e["output_tokens"], e["latency_ms"], e.get("model", "")))
        elif e["event"] == "model_error":
            cls = getattr(base, e["type"], base.ModelError)
            script.append(cls(e["error"], e["generation"]) if e["type"] == "ModelBadOutput" else cls(e["error"]))
    return ScriptedClient(script)

# Re-runs the CURRENT loop on a trace's recorded model replies. Read tools run for real; writes and commands are refused.
def replay(path: Path) -> str:
    from runtime.loop import run_agent
    from runtime.tools import default_tools
    from runtime.types import Message
    events = load(path)
    start = next(e for e in events if e["event"] == "run_start")
    first = next(e for e in events if e["event"] == "model_request")["new_messages"]
    end = next((e for e in events if e["event"] in ("run_end", "run_error")), {})
    messages = [Message(m["role"], m["content"]) for m in first]
    try:
        r = run_agent(replay_client(events), default_tools(start["workspace"]), messages, start["max_steps"])
        now = f"stopped={r.stopped} steps={r.steps}"
    except Exception as e:
        now = f"crash {type(e).__name__}"
    was = f"stopped={end.get('stopped')} steps={end.get('steps')}" if end.get("event") == "run_end" else f"crash {end.get('type')}"
    return f"recorded: {was}\nreplayed: {now}\n" + ("SAME" if was == now else "DIFFERENT: the loop now behaves differently on these model replies")


def main():
    ap = argparse.ArgumentParser(prog="python -m runtime.trace", description="list, view or replay run traces")
    ap.add_argument("command", choices=["list", "view", "replay"])
    ap.add_argument("file", nargs="?", help="trace file (default: latest)")
    ap.add_argument("--full", action="store_true", help="show full text, not previews")
    ap.add_argument("--dir", default="traces")
    a = ap.parse_args()
    if a.command == "list":
        for f in sorted(Path(a.dir).glob("*.jsonl")):
            end = next((e for e in reversed(load(f)) if e["event"] in ("run_end", "run_error")), None)
            status = (end.get("stopped") or "crash") if end else "unfinished"
            print(f"{f.name:60} {status}")
        return
    path = Path(a.file) if a.file else latest(a.dir)
    print(f"[{path}]\n")
    print(render(load(path), a.full) if a.command == "view" else replay(path))

if __name__ == "__main__":
    main()
