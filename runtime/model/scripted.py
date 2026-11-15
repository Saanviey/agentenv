from runtime.model.base import ModelClient

# A model that replies from a script: used by tests and by trace replay. Each item is a Response or an exception to raise.
class ScriptedClient(ModelClient):
    def __init__(self, script: list):
        self.script, self.calls = list(script), []  # calls: every message list it was sent

    def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
