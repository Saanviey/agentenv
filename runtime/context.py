import json, logging
from dataclasses import dataclass
from typing import Callable
from runtime.types import Message
from runtime.model.base import ModelClient, ModelError

log = logging.getLogger("runtime.context")

CHARS_PER_TOKEN = 4   # rough average for English + code; corrected at runtime by ContextManager.scale
PER_MESSAGE = 4       # role markers and separators the chat template adds around each message
PINNED = 2            # messages[0] = system prompt, messages[1] = the task: never removed

class ContextOverflow(ModelError): ...  # even after compaction, the conversation doesn't fit the budget

SUMMARY_PROMPT = """You compress an AI coding agent's work log so it can continue with less context.
Rules (B25: a summary that invents things misleads the agent for the rest of the run):
- Only state what the log actually shows. Never reconstruct, paraphrase or guess code; quote a line only if you copy it exactly from a tool result.
- Don't say where a bug is unless a test or a verified change showed it; otherwise write "suspected".
- An earlier summary may contain mistakes: where the newer log disagrees, trust the log.
Keep: files that matter (with line numbers), facts found, changes made, test results, what is left to do.
At most 200 words, plain text."""
FACTS_HEADER = "Recorded by the runtime (reliable):"

# In: messages (+ tool schemas, which are sent every call too). Out: rough token count, before calibration.
def estimate_tokens(messages: list[Message], tools: list[dict] | None = None) -> int:
    n = 0
    for m in messages:
        chars = len(m.content) + sum(len(c.name) + len(json.dumps(c.arguments, default=str)) for c in m.tool_calls)
        n += PER_MESSAGE + chars // CHARS_PER_TOKEN
    return n + (len(json.dumps(tools)) // CHARS_PER_TOKEN if tools else 0)

# What one compaction did, for the trace.
@dataclass
class Compaction:
    before: int            # estimated tokens before
    after: int             # and after
    budget: int
    elided: int            # tool results replaced by a short stub
    summarized: int        # messages folded into the summary
    summary: str           # the summary text, if one was written
    method: str            # e.g. "elide", "elide+summarize(model)"

# Keeps the conversation under a token budget, compacting it in place when it grows too big:
#   1. elide old tool results  2. summarize old turns  3. elide recent results  4. give up (ContextOverflow).
# The system prompt, the task and the last turn are never touched.
class ContextManager:
    def __init__(self, budget_tokens: int, summarize: Callable[[str], str] | None = None,
                 keep_recent: int = 3, target: float = 0.6):
        self.budget = budget_tokens
        self.summarize = summarize       # None = mechanical summary only (no model call)
        self.keep_recent = keep_recent   # turns kept whole by steps 1-2
        self.target = target             # compact down to target * budget, so it isn't needed again next step
        self.scale = 1.0                 # learned: real tokens / estimated tokens

    def estimate(self, messages: list[Message], tools: list[dict] | None = None) -> int:
        return round(estimate_tokens(messages, tools) * self.scale)

    # After each model call: learn how far off the raw estimate was (code and JSON are usually denser than 4 chars/token).
    def observe(self, raw_estimate: int, actual: int) -> None:
        if raw_estimate <= 0 or actual <= 0 or actual / raw_estimate < 0.5:
            return  # no data, or a cached/partial count (Ollama reuses its cache): don't learn from it
        self.scale = min(2.0, max(0.8, 0.5 * self.scale + 0.5 * actual / raw_estimate))

    # In: the working message list (changed in place). Out: Compaction if anything changed, else None.
    # facts: runtime-recorded facts (Session.facts) appended under any summary; never written by the model.
    def fit(self, messages: list[Message], tools: list[dict] | None = None, budget: int | None = None,
            facts: str = "") -> Compaction | None:
        budget = budget or self.budget
        before = self.estimate(messages, tools)
        if before <= budget:
            return None
        goal = int(budget * self.target)
        fits = lambda limit: self.estimate(messages, tools) <= limit
        elided, summarized, summary, steps = 0, 0, "", []

        old = _turns(messages)[:-self.keep_recent]
        for a, b in old:                                   # 1. oldest tool results first
            for i in range(a, b):
                if fits(goal):
                    break
                elided += _elide(messages[i])
        if elided:
            steps.append("elide")

        if not fits(goal) and old:                         # 2. fold old turns (and any earlier summary) into one summary
            end = old[-1][1]
            summary, how = self._summary(messages[PINNED:end], budget)
            summarized = end - PINNED
            text = f"[Summary of earlier steps, to save space. May be incomplete: re-read files before relying on details.]\n{summary}"
            if facts:
                text += f"\n\n{FACTS_HEADER}\n{facts}"
            messages[PINNED:end] = [Message("user", text, tag="summary")]
            steps.append(f"summarize({how})")

        recent = _turns(messages)[:-1]                     # 3. still too big: elide results in recent turns, except the last
        n = 0
        for a, b in recent:
            for i in range(a, b):
                if fits(budget):
                    break
                n += _elide(messages[i])
        if n:
            elided += n
            steps.append("elide-recent")

        after = self.estimate(messages, tools)
        if after > budget:                                 # 4. nothing left to cut
            raise ContextOverflow(f"conversation is ~{after} tokens after compaction; budget is {budget}. "
                                  "The task, system prompt or the latest step alone are too big.")
        return Compaction(before, after, budget, elided, summarized, summary, "+".join(steps))

    # Out: (summary text, "model" or "mechanical"). Falls back to mechanical if the model call fails or says nothing.
    def _summary(self, msgs: list[Message], budget: int) -> tuple[str, str]:
        if self.summarize:
            try:
                text = self.summarize(_render(msgs, max_chars=int(budget * CHARS_PER_TOKEN * 0.5)))
                if text.strip():
                    return text.strip(), "model"
            except Exception as e:
                log.warning("summary by model failed (%s); using mechanical summary", e)
        return _mechanical(msgs), "mechanical"

# Out: a summarizer that asks the model itself (no tools). Costs one extra model call per compaction.
def model_summarizer(client: ModelClient) -> Callable[[str], str]:
    def summarize(log_text: str) -> str:
        return client.complete([Message("system", SUMMARY_PROMPT), Message("user", log_text)]).message.content
    return summarize

# Out: [start, end) index ranges of turns after the pinned messages. A turn starts at each assistant message;
# an earlier summary counts as its own turn. Tool results and reminders belong to the turn before them.
def _turns(messages: list[Message]) -> list[tuple[int, int]]:
    starts = [i for i in range(PINNED, len(messages)) if i == PINNED or messages[i].role == "assistant"]
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(messages)])]

# Replaces a big tool result with a stub. Out: 1 if it changed something.
def _elide(m: Message) -> int:
    if m.role != "tool" or m.tag == "elided" or len(m.content) <= 300:
        return 0
    m.content = (f"[elided to save context: {m.tool_name} result, {len(m.content)} chars. "
                 f"It started: {m.content[:150]!r}. Call the tool again if you need it.]")
    m.tag = "elided"
    return 1

# Messages as plain text for the summarizer, capped (middle cut) so the summary request itself fits.
def _render(msgs: list[Message], max_chars: int) -> str:
    lines = []
    for m in msgs:
        if m.role == "assistant":
            if m.content:
                lines.append(f"assistant: {m.content[:800]}")
            lines += [f"assistant called {c.name}({json.dumps(c.arguments, default=str)[:300]})" for c in m.tool_calls]
        elif m.role == "tool":
            lines.append(f"result of {m.tool_name}: {m.content[:1500]}")
        elif m.tag == "summary":  # the facts block is rebuilt fresh each time; only the summary part is re-summarized
            lines.append(f"earlier summary (may contain mistakes): {m.content.split(FACTS_HEADER)[0][:1500]}")
        else:
            lines.append(f"{m.role}: {m.content[:1500]}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars // 2] + "\n...[middle of log cut]...\n" + text[-max_chars // 2:]
    return text

# Summary without a model: an earlier summary (if any) + one line per action.
def _mechanical(msgs: list[Message]) -> str:
    lines = []
    for m in msgs:
        if m.tag == "summary":
            lines.append(m.content.split("\n", 1)[-1].split(FACTS_HEADER)[0].strip())
        elif m.role == "assistant":
            lines += [f"- called {c.name}({json.dumps(c.arguments, default=str)[:120]})" for c in m.tool_calls]
        elif m.role == "tool":
            lines.append(f"  -> {m.content.splitlines()[0][:120] if m.content else '(empty)'}")
    return "Earlier actions (results not kept; repeat a call if you need its output):\n" + "\n".join(lines)

# In: model profile + [agent] config. Out: a ContextManager, or None if the profile sets no max_context_tokens.
def from_config(model: dict, agent: dict, client: ModelClient) -> ContextManager | None:
    limit = model.get("max_context_tokens")
    if not limit:
        return None
    summarize = model_summarizer(client) if agent.get("summarize", "model") == "model" else None
    return ContextManager(limit - agent.get("reserve_output_tokens", 1500), summarize, agent.get("keep_recent_turns", 3))
