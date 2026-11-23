from dataclasses import asdict
from runtime.types import Message, ToolCall
from runtime.loop import RunState, Checkpoint
from runtime.store import Store
from runtime.tools import ToolRegistry
from runtime.context import ContextManager

# RunState + the Session + the ContextManager -> one JSON-able dict.
def encode(st: RunState, tools: ToolRegistry, context: ContextManager | None) -> dict:
    d = asdict(st)  # messages and their tool calls become plain dicts too
    d["session"] = tools.session.to_dict() if tools.session else None
    d["context"] = {"budget": context.budget, "scale": context.scale} if context else None
    return d

# The reverse. The Session and ContextManager are restored INTO the existing objects, because the tools already hold them.
def decode(d: dict, tools: ToolRegistry, context: ContextManager | None) -> RunState:
    if d.get("session") and tools.session:
        tools.session.load_dict(d["session"])
    if d.get("context") and context:
        context.budget, context.scale = d["context"]["budget"], d["context"]["scale"]
    messages = [Message(m["role"], m["content"], m["tool_call_id"], m["tool_name"],
                        [ToolCall(**c) for c in m["tool_calls"]], m.get("tag", "")) for m in d["messages"]]
    return RunState(messages, d["step"], d["in_tok"], d["out_tok"], d["reminders"], d["rejections"],
                    dict(d["seen"]), d["sent"], d["pending"])

# Out: a Checkpoint function for run_agent that saves everything to the store after every action.
def checkpointer(store: Store, run_id: str, tools: ToolRegistry, context: ContextManager | None) -> Checkpoint:
    def save(st: RunState, status: str) -> None:
        store.save(run_id, st.step, status, encode(st, tools, context))
    return save
