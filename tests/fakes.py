from runtime.types import Message, Response, ToolCall
from runtime.model.scripted import ScriptedClient as FakeClient

def call(name: str, args=None, stop="tool_use") -> Response:
    return Response(Message("assistant", "", tool_calls=[ToolCall(name, {} if args is None else args)]), stop, 1, 1, 0, "fake")

def text(content: str, stop="end") -> Response:
    return Response(Message("assistant", content), stop, 1, 1, 0, "fake")
