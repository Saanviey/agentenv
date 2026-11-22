from dataclasses import dataclass, field
from typing import Literal, Any

Role = Literal["system", "user", "assistant", "tool"]

# The model asking us to run one tool (structured, from the backend).
@dataclass
class ToolCall:
    name: str
    arguments: Any               # usually a dict; some backends send a JSON string
    id: str | None = None        # Ollama may omit it; then results match calls by order + tool_name

# One conversation entry. A list of these is the history.
@dataclass  # auto-generates __init__ from the fields
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None   # only for role="tool"
    tool_name: str | None = None      # only for role="tool"
    tool_calls: list[ToolCall] = field(default_factory=list)  # only for role="assistant"
    tag: str = ""                     # runtime bookkeeping, never sent: "summary" | "elided"

# Result of one model call: the reply + metadata.
@dataclass
class Response:
    message: Message
    stop_reason: str                  # "end" | "length" | "tool_use"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)  # hidden from print; fresh dict per instance
