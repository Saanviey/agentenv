import json
from runtime.types import ToolCall

_decoder = json.JSONDecoder()

# Fallback for models that write the call as plain JSON instead of inside <tool_call> tags (B5/B11).
# In: reply text + names of the tools offered. Out: the first {"name","arguments"} object naming a real tool, else None.
def parse_text_tool_call(text: str, names: set[str]) -> ToolCall | None:
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = _decoder.raw_decode(text, i)  # parse one JSON value starting at i, ignoring what follows
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            name = obj.get("name", obj.get("tool"))
            if name in names:  # only real tools, so an example dict in prose isn't mistaken for a call
                args = obj.get("arguments", obj.get("args"))
                return ToolCall(name, {} if args is None else args)
        i = text.find("{", i + 1)
    return None
