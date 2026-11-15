import json, logging
from dataclasses import dataclass
from runtime.types import Message
from runtime.model.base import ModelClient
from runtime.tools import ToolRegistry, FINISH

log = logging.getLogger("runtime.loop")

SYSTEM_PROMPT = """You are a coding assistant working inside the user's project folder.
Use the provided tools to act. Paths are relative to the project root.
Never describe a file you have not read with read_file.
Every reply must be a tool call. Never write the answer as plain text.
When you have the answer, put it only inside the finish tool's answer argument, short and to the point.
If the task needs no other tool, call finish right away."""

REMINDER = "Your reply was plain text, not a tool call. Call a tool, or put your answer inside finish."
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
              max_steps: int = 10, max_reminders: int = 2) -> RunResult:
    messages = list(messages)  # copy; don't mutate the caller's list
    in_tok = out_tok = reminders = 0
    seen: dict[str, int] = {}  # call signature -> times called, to spot repeats (B13)

    for step in range(1, max_steps + 1):
        resp = client.complete(messages, tools.schemas())
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens
        messages.append(resp.message)
        log.info("step %d: stop=%s in=%d out=%d %.0fms",
                 step, resp.stop_reason, resp.input_tokens, resp.output_tokens, resp.latency_ms)

        if not resp.message.tool_calls:  # plain text: not an answer, not an action
            reminders += 1
            log.info("  no tool call (%d/%d): %s", reminders, max_reminders, _preview(resp.message.content))
            if reminders > max_reminders:  # model can't follow the format; take its text rather than burn steps
                return RunResult(resp.message.content, messages, step, in_tok, out_tok, "no_finish")
            messages.append(Message("user", CUT_OFF if resp.stop_reason == "length" else REMINDER))
            continue
        reminders = 0

        for call in resp.message.tool_calls:
            if call.name == FINISH:
                answer = call.arguments.get("answer") if isinstance(call.arguments, dict) else None
                if isinstance(answer, str) and answer.strip():
                    return RunResult(answer, messages, step, in_tok, out_tok, "finish")
                result = "error: finish needs a non-empty 'answer'"
            else:
                result = tools.run(call.name, call.arguments)
                key = _signature(call.name, call.arguments)
                if seen.get(key):
                    result += REPEAT_NOTE.format(n=seen[key])
                seen[key] = seen.get(key, 0) + 1
            log.info("  tool %s(%s) -> %s", call.name, call.arguments, _preview(result))
            messages.append(Message("tool", result, tool_call_id=call.id, tool_name=call.name))

    log.warning("hit max_steps=%d without finish", max_steps)
    return RunResult("(stopped: step limit reached)", messages, max_steps, in_tok, out_tok, "max_steps")

# Same tool + same args -> same string, regardless of key order.
def _signature(name: str, args) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, default=str)

def _preview(s: str, n: int = 60) -> str:
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."
