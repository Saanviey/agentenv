from runtime.types import Message, Response, ToolCall
from runtime.model.base import ModelClient

# A model that replies from a script, so the loop can be tested without a network or a real model.
class FakeClient(ModelClient):
    def __init__(self, script: list):  # each item: a Response to return, or an exception to raise
        self.script, self.calls = list(script), []

    def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

def call(name: str, args=None, stop="tool_use") -> Response:
    return Response(Message("assistant", "", tool_calls=[ToolCall(name, {} if args is None else args)]), stop, 1, 1, 0, "fake")

def text(content: str, stop="end") -> Response:
    return Response(Message("assistant", content), stop, 1, 1, 0, "fake")
