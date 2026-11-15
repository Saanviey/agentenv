import json, logging
from dataclasses import dataclass
from runtime.types import Message
from runtime.model.base import ModelClient, ModelBadOutput
from runtime.tools import ToolRegistry, FINISH, parse_args
from runtime import policy
from runtime.policy import Approver, deny_all
from runtime.trace import Tracer, NULL
import time

log = logging.getLogger("runtime.loop")

SYSTEM_PROMPT = """You are a coding assistant working inside the user's project folder.
Use the provided tools to act. Paths are relative to the project root.
Never describe a file you have not read with read_file.
To change code: read the file, change it with edit_file (exact old text -> new text), then run the tests with run_command to check.
If an action is blocked or not approved, do not retry it; try another way or call finish and explain.
Every reply must be a tool call. Never write the answer as plain text.
When you have the answer, put it only inside the finish tool's answer argument, short and to the point.
If the task needs no other tool, call finish right away."""

REMINDER = "Your reply was plain text, not a tool call. Call a tool, or put your answer inside finish."
BAD_CALL = ("Your last tool call was malformed (arguments were not valid JSON), so it was not run. You wrote:\n{gen}\n"
            "Call the tool again with valid JSON arguments.")
CUT_OFF = "Your reply was cut off because it was too long. Call a tool, or call finish with a shorter answer."
REPEAT_NOTE = "\n\nnote: you already made this exact call {n} time(s) before; the result has not changed. Use it and move on."

# Outcome of one agent run.
@dataclass
class RunResult:
    answer: str
    messages: list[Message]   # full history, including tool calls and results
    steps: int                # model calls made
    input_tokens: int         # summed over all steps
    output_tokens: int
    stopped: str              # "finish" | "no_finish" | "max_steps"

# In: client, tools, starting messages, limits. Out: RunResult. Raises: ModelError.
# Repeats: call model -> run the tools it asked for -> append results, until it calls finish.
def run_agent(client: ModelClient, tools: ToolRegistry, messages: list[Message],
              max_steps: int = 10, max_reminders: int = 2, approve: Approver = deny_all,
              trace: Tracer = NULL) -> RunResult:
    try:
        result = _run(client, tools, list(messages), max_steps, max_reminders, approve, trace)  # copy; don't mutate the caller's list
    except Exception as e:
        trace.event("run_error", type=type(e).__name__, error=str(e))
        raise
    trace.event("run_end", stopped=result.stopped, steps=result.steps, answer=result.answer,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens)
    return result

def _run(client, tools, messages, max_steps, max_reminders, approve, trace) -> RunResult:
    sent = 0  # how many messages the model has already seen, so the trace stores only new ones
    in_tok = out_tok = reminders = 0
    seen: dict[str, int] = {}  # call signature -> times called, to spot repeats (B13)

    for step in range(1, max_steps + 1):
        trace.event("model_request", step=step, new_messages=messages[sent:], total_messages=len(messages))
        sent = len(messages)
        try:
            resp = client.complete(messages, tools.schemas())
        except ModelBadOutput as e:  # model's mistake, not the server's: show it and let it try again
            trace.event("model_error", step=step, type="ModelBadOutput", error=str(e), generation=e.generation)
            reminders += 1
            log.info("  malformed tool call (%d/%d): %s", reminders, max_reminders, _preview(e.generation or str(e)))
            if reminders > max_reminders:
                raise
            messages.append(Message("user", BAD_CALL.format(gen=e.generation[:500] or "(not shown)")))
            trace.event("reminder", step=step, kind="bad_call")
            continue
        except Exception as e:
            trace.event("model_error", step=step, type=type(e).__name__, error=str(e), generation="")
            raise
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens
        messages.append(resp.message)
        trace.event("model_response", step=step, message=resp.message, stop_reason=resp.stop_reason,
                    input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                    latency_ms=round(resp.latency_ms), model=resp.model, raw=resp.raw)
        log.info("step %d: stop=%s in=%d out=%d %.0fms",
                 step, resp.stop_reason, resp.input_tokens, resp.output_tokens, resp.latency_ms)

        if not resp.message.tool_calls:  # plain text: not an answer, not an action
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
            if call.name == FINISH:
                answer = call.arguments.get("answer") if isinstance(call.arguments, dict) else None
                if isinstance(answer, str) and answer.strip():
                    return RunResult(answer, messages, step, in_tok, out_tok, "finish")
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

# Same tool + same args -> same string, regardless of key order.
def _signature(name: str, args) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, default=str)

def _preview(s: str, n: int = 60) -> str:
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."
