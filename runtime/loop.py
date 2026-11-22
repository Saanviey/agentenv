import json, logging
from dataclasses import dataclass, asdict
from runtime.types import Message
from runtime.model.base import ModelClient, ModelBadOutput, ModelContextOverflow
from runtime.context import ContextManager, estimate_tokens
from runtime.tools import ToolRegistry, FINISH, parse_args
from runtime import policy
from runtime.policy import Approver, deny_all
from runtime.trace import Tracer, NULL
from runtime.verify import Verifier
import time

log = logging.getLogger("runtime.loop")

SYSTEM_PROMPT = """You are a coding assistant working inside the user's project folder.
Use the provided tools to act. Paths are relative to the project root.
Files and outputs can be big: use search to find things, read_file with start_line/end_line for parts of a file.
Long tool outputs are cut; read_output pages through them. Old results may be shortened later to save space.
Never describe a file you have not read with read_file.
To change code: read the file, change it with edit_file (exact old text -> new text), then run the tests with run_command to check.
If you changed files, the tests must pass after your last change before you call finish. revert_file undoes your changes to a file.
If an action is blocked or not approved, do not retry it; try another way or call finish and explain.
Every reply must be a tool call. Never write the answer as plain text.
When you have the answer, put it only inside the finish tool's answer argument, short and to the point.
If the task needs no other tool, call finish right away."""

REMINDER = "Your reply was plain text, not a tool call. Call a tool, or put your answer inside finish."
BAD_CALL = ("Your last reply was rejected and nothing was run. Reason: {why}\nYou wrote:\n{gen}\n"
            "Reply with a valid call to one of the available tools: {tools}. To give your final answer, call finish.")
CUT_OFF = "Your reply was cut off because it was too long. Call a tool, or call finish with a shorter answer."
MAX_REJECTIONS = 2  # finish rejected this many times -> accept the answer but mark the run 'unverified'
_decoder = json.JSONDecoder()
REPEAT_NOTE = "\n\nnote: you already made this exact call {n} time(s) before; the result has not changed. Use it and move on."

# Outcome of one agent run.
@dataclass
class RunResult:
    answer: str
    messages: list[Message]   # full history, including tool calls and results
    steps: int                # model calls made
    input_tokens: int         # summed over all steps
    output_tokens: int
    stopped: str              # "finish" | "unverified" | "no_finish" | "max_steps"

# In: client, tools, starting messages, limits. Out: RunResult. Raises: ModelError.
# Repeats: call model -> run the tools it asked for -> append results, until it calls finish.
def run_agent(client: ModelClient, tools: ToolRegistry, messages: list[Message],
              max_steps: int = 10, max_reminders: int = 2, approve: Approver = deny_all,
              trace: Tracer = NULL, verifier: Verifier = Verifier(), rollback: bool = True,
              context: ContextManager | None = None) -> RunResult:
    progress = {"steps": 0, "input_tokens": 0, "output_tokens": 0}  # updated each step, so a crash still reports how far it got
    try:
        result = _run(client, tools, list(messages), max_steps, max_reminders, approve, trace, progress, verifier, context)  # copy; don't mutate the caller's list
    except Exception as e:
        trace.event("run_error", type=type(e).__name__, error=str(e), **progress)
        if rollback:
            _rollback_if_broken(tools, verifier, trace)
        e.progress = progress
        raise
    if rollback and result.stopped != "finish":
        _rollback_if_broken(tools, verifier, trace)
    trace.event("run_end", stopped=result.stopped, steps=result.steps, answer=result.answer,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens)
    return result

def _run(client, tools, messages, max_steps, max_reminders, approve, trace, progress, verifier, context) -> RunResult:
    sent = 0  # how many messages the model has already seen, so the trace stores only new ones
    in_tok = out_tok = reminders = rejections = 0

    # Every way the model can say "done" comes through here. Out: RunResult if accepted, else the reason it was rejected.
    def try_finish(answer: str, step: int, source: str) -> RunResult | str:
        nonlocal rejections
        reason = verifier.check_finish(answer, tools.session)
        if reason is None:
            if source != "finish":
                trace.event("finish_salvaged", step=step, source=source, answer=answer)
            return RunResult(answer, messages, step, in_tok, out_tok, "finish")
        rejections += 1
        trace.event("finish_rejected", step=step, source=source, reason=reason, rejections=rejections)
        log.info("  finish rejected (%d/%d): %s", rejections, MAX_REJECTIONS, _preview(reason, 100))
        if rejections > MAX_REJECTIONS:  # keep the answer, but say it was never verified
            return RunResult(answer, messages, step, in_tok, out_tok, "unverified")
        return f"finish rejected: {reason}"
    seen: dict[str, int] = {}  # call signature -> times called, to spot repeats (B13)

    for step in range(1, max_steps + 1):
        progress.update(steps=step, input_tokens=in_tok, output_tokens=out_tok)
        schemas = tools.schemas()
        sent = _compact(context, messages, schemas, sent, trace, step, facts=_facts(tools))
        trace.event("model_request", step=step, new_messages=messages[sent:], total_messages=len(messages),
                    est_tokens=context.estimate(messages, schemas) if context else estimate_tokens(messages, schemas))
        sent = len(messages)
        try:
            resp, raw_est = _complete(client, messages, schemas, context, trace, step, _facts(tools))
        except ModelBadOutput as e:  # model's mistake, not the server's: show it and let it try again
            trace.event("model_error", step=step, type="ModelBadOutput", error=str(e), generation=e.generation)
            if (answer := _salvage_finish(e.generation, set(tools.names()))) is not None:
                if isinstance(r := try_finish(answer, step, "bad_call"), RunResult):
                    return r
                messages.append(Message("user", r))
                continue
            reminders += 1
            log.info("  malformed tool call (%d/%d): %s", reminders, max_reminders, _preview(e.generation or str(e)))
            if reminders > max_reminders:
                raise
            messages.append(Message("user", BAD_CALL.format(why=_reason(str(e)), gen=e.generation[:500] or "(not shown)",
                                                            tools=", ".join(tools.names()))))
            trace.event("reminder", step=step, kind="bad_call")
            continue
        except Exception as e:
            trace.event("model_error", step=step, type=type(e).__name__, error=str(e), generation="")
            raise
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens
        if context:
            context.observe(raw_est, resp.input_tokens)
        messages.append(resp.message)
        trace.event("model_response", step=step, message=resp.message, stop_reason=resp.stop_reason,
                    input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                    latency_ms=round(resp.latency_ms), model=resp.model, raw=resp.raw)
        log.info("step %d: stop=%s in=%d out=%d %.0fms",
                 step, resp.stop_reason, resp.input_tokens, resp.output_tokens, resp.latency_ms)

        if not resp.message.tool_calls:  # plain text: not an answer, not an action
            if (answer := _salvage_finish(resp.message.content, set(tools.names()))) is not None:
                if isinstance(r := try_finish(answer, step, "plain_text"), RunResult):
                    return r
                messages.append(Message("user", r))
                continue
            reminders += 1
            log.info("  no tool call (%d/%d): %s", reminders, max_reminders, _preview(resp.message.content))
            if reminders > max_reminders:  # model can't follow the format; take its text rather than burn steps
                return RunResult(resp.message.content, messages, step, in_tok, out_tok, "no_finish")
            kind = "cut_off" if resp.stop_reason == "length" else "plain_text"
            messages.append(Message("user", CUT_OFF if kind == "cut_off" else REMINDER))
            trace.event("reminder", step=step, kind=kind)
            continue
        reminders = 0

        for call in resp.message.tool_calls:
            t0 = time.perf_counter()
            decision, reason, approved, repeat = "allow", "", True, False
            known = set(tools.names())
            if call.name == FINISH or (tools.get(call.name) is None and _answer_in(call.arguments, known)):
                answer = _answer_in(call.arguments, known)  # same rule as the check above, so nested answers are found too (B27)
                if answer:
                    r = try_finish(answer, step, "finish" if call.name == FINISH else "alias")
                    if isinstance(r, RunResult):
                        return r
                    result = f"error: {r}"
                else:
                    result = "error: finish needs a non-empty 'answer'"
            else:
                decision, reason, approved, blocked = _gate(call, tools, approve)
                result = blocked or tools.run(call.name, call.arguments)
                key = _signature(call.name, call.arguments)
                if seen.get(key):
                    repeat = True
                    result += REPEAT_NOTE.format(n=seen[key])
                seen[key] = seen.get(key, 0) + 1
            trace.event("tool", step=step, name=call.name, args=call.arguments, decision=decision, reason=reason,
                        approved=approved, result=result, ms=round((time.perf_counter() - t0) * 1000), repeat=repeat)
            log.info("  tool %s(%s) -> %s", call.name, call.arguments, _preview(result))
            messages.append(Message("tool", result, tool_call_id=call.id, tool_name=call.name))

    log.warning("hit max_steps=%d without finish", max_steps)
    return RunResult("(stopped: step limit reached)", messages, max_steps, in_tok, out_tok, "max_steps")

# Before each model call: shrink the conversation if it's over budget. Out: the new `sent` index for the trace.
# Compaction never touches the last turn, which holds every message added since the last call.
def _compact(context, messages, schemas, sent, trace, step, budget=None, facts: str = "") -> int:
    if context is None:
        return sent
    n_new = len(messages) - sent
    c = context.fit(messages, schemas, budget, facts=facts)
    if c is None:
        return sent
    trace.event("compaction", step=step, **asdict(c))
    log.info("  context %d -> %d tokens (budget %d): %s", c.before, c.after, c.budget, c.method)
    return len(messages) - n_new

def _facts(tools: ToolRegistry) -> str:
    return tools.session.facts() if tools.session else ""

# One model call. On "request too large" (HTTP 413), learn a smaller budget, compact, and try once more.
# Out: (response, raw token estimate of what was sent), for calibration.
def _complete(client, messages, schemas, context, trace, step, facts: str = ""):
    try:
        return client.complete(messages, schemas), estimate_tokens(messages, schemas)
    except ModelContextOverflow as e:
        if context is None:
            raise
        trace.event("model_error", step=step, type="ModelContextOverflow", error=str(e), generation="")
        context.budget = max(500, int(context.estimate(messages, schemas) * 0.8))  # the server's limit is lower than we thought
        log.warning("  request too large; budget lowered to %d, compacting", context.budget)
        _compact(context, messages, schemas, len(messages), trace, step, facts=facts)
        return client.complete(messages, schemas), estimate_tokens(messages, schemas)

# A run that didn't finish properly: undo its file changes if the verifier says they're broken.
def _rollback_if_broken(tools: ToolRegistry, verifier: Verifier, trace: Tracer) -> None:
    session = tools.session
    if session is None or not session.changed():
        return
    ok = verifier.changes_ok(session)
    if ok is False:
        restored = session.restore_all()
        log.warning("run did not finish and tests fail: rolled back %s", ", ".join(restored))
        trace.event("rollback", files=restored, reason="run did not finish and the tests fail")
    else:
        trace.event("rollback_skipped", files=session.changed(),
                    reason="tests pass" if ok else "no test command, can't tell if the changes are good")

# Policy + approval before a tool runs.
# Out: (decision, reason, approved, error text if blocked else None). Decision/reason/approved go into the trace.
def _gate(call, tools: ToolRegistry, approve: Approver) -> tuple[str, str, bool, str | None]:
    tool = tools.get(call.name)
    args, err = parse_args(call.arguments)
    if tool is None or err:
        return "allow", "", True, None  # tools.run() reports unknown tools and bad args
    d = policy.check(tool, args)
    if d.action == "deny":
        log.warning("  BLOCKED %s: %s", call.name, d.reason)
        return "deny", d.reason, False, f"error: blocked by policy: {d.reason}"
    if d.action == "ask" and not approve(call.name, args, d.reason):
        log.warning("  NOT APPROVED %s: %s", call.name, d.reason)
        return "ask", d.reason, False, "error: the user did not approve this action. Do not retry it; try another way or call finish."
    return d.action, d.reason, True, None

# In: text the model wrote + names of real tools. Out: the answer if the text is a finish call in disguise, else None.
# B17: gpt-oss wraps finish in made-up tools ("answer", "commentary", "json"...) or writes a bare {"answer": ...}.
def _salvage_finish(text: str, known: set[str]) -> str | None:
    i = text.find("{")
    while i != -1:
        try:
            obj, end = _decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            obj, end = None, i + 1
        if (answer := _answer_in(obj, known)) is not None:
            return answer
        i = text.find("{", end)  # skip past a parsed object, so nested args of another tool are never searched
    return None

# Rule: finish, or any call to a tool that DOESN'T exist, is unwrapped (up to 3 levels) looking for {"answer": "..."}.
# A call to a real tool (read_file...) is never treated as a finish.
def _answer_in(obj, known: set[str], depth: int = 0) -> str | None:
    if not isinstance(obj, dict) or depth > 3:
        return None
    if "name" in obj:
        if obj["name"] != FINISH and obj["name"] in known:
            return None
        return _answer_in(obj.get("arguments"), known, depth + 1)
    answer = obj.get("answer")
    return answer.strip() if isinstance(answer, str) and answer.strip() else None

# Pulls the server's human-readable reason out of an error like '... {"error":{"message":"..."}}'.
def _reason(err: str) -> str:
    try:
        return json.loads(err[err.index("{"):])["error"]["message"][:300]
    except Exception:
        return err[:300]

# Same tool + same args -> same string, regardless of key order.
def _signature(name: str, args) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, default=str)

def _preview(s: str, n: int = 60) -> str:
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."
