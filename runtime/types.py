from dataclasses import dataclass, field
from typing import Literal, Any

Role = Literal["system", "user", "assistant", "tool"]

# One conversation entry. A list of these is the history.
@dataclass  # auto-generates __init__ from the fields
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None   # only for role="tool"

# Result of one model call: the reply + metadata.
@dataclass
class Response:
    message: Message
    stop_reason: str                  # "end" | "length" | ...
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)  # hidden from print; fresh dict per instance
