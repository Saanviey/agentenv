import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

MAX_OUTPUT = 8000   # chars; stops one tool result from flooding the context window
FINISH = "finish"   # the loop ends the run when the model calls this tool


class ToolError(Exception): ...  # expected failure; its message is shown to the model


# One tool = what the model sees (name, description, parameters) + what we run (fn).
@dataclass
class Tool:
    name: str
    description: str             # the model reads this to decide when to call the tool
    parameters: dict[str, Any]   # JSON Schema for the arguments
    fn: Callable[..., str]


# Holds the available tools; the loop asks it for schemas and to run calls.
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    # Out: tool schemas in Ollama/OpenAI format, sent in the request's `tools` field.
    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in self._tools.values()]

    # In: tool name + args as the model gave them. Out: result text.
    # Never raises: every failure becomes an "error: ..." string the model can read and retry from.
    def run(self, name: str, args: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'. available: {', '.join(self._tools)}"

        if isinstance(args, str):  # some models send arguments as a JSON string
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return f"error: arguments for {name} are not valid JSON: {args!r}"
        args = args or {}
        if not isinstance(args, dict):
            return f"error: arguments for {name} must be a JSON object, got {type(args).__name__}"

        props = tool.parameters.get("properties", {})
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        unknown = [k for k in args if k not in props]
        if missing or unknown:
            return f"error: bad arguments for {name}: missing={missing} unknown={unknown}"

        try:
            out = tool.fn(**args)
        except ToolError as e:
            return f"error: {e}"
        except Exception as e:  # a bug in the tool shouldn't crash the loop
            return f"error: {name} failed: {type(e).__name__}: {e}"

        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n...[truncated {len(out) - MAX_OUTPUT} chars]"
        return out


# In: workspace folder the file tools may touch. Out: registry with the starter tools.
def default_tools(root: str | Path = ".") -> ToolRegistry:
    root = Path(root).resolve()

    # Resolves a model-given path and refuses anything outside root (../, absolute paths, symlinks out).
    def safe_path(p: str) -> Path:
        full = (root / p).resolve()
        if not full.is_relative_to(root):
            raise ToolError(f"'{p}' is outside the workspace")
        return full

    def list_dir(path: str = ".") -> str:
        d = safe_path(path)
        if not d.is_dir():
            raise ToolError(f"'{path}' is not a directory")
        entries = sorted((e for e in d.iterdir() if not e.name.startswith(".") and e.name != "__pycache__"),
                         key=lambda e: (e.is_file(), e.name))
        # full paths from the root, so the model can pass them straight to read_file
        return "\n".join(str(e.relative_to(root)) + ("/" if e.is_dir() else "") for e in entries) or "(empty)"

    def read_file(path: str) -> str:
        f = safe_path(path)
        if not f.is_file():
            raise ToolError(f"'{path}' is not a file")
        try:
            return f.read_text()
        except UnicodeDecodeError:
            raise ToolError(f"'{path}' is not a text file")

    reg = ToolRegistry()
    reg.add(Tool(
        "list_dir", "List a directory. Returns paths relative to the project root; folders end with '/'.",
        {"type": "object",
         "properties": {"path": {"type": "string", "description": "Directory relative to the project root. Default '.'"}}},
        list_dir))
    reg.add(Tool(
        "read_file", "Read a text file and return its contents.",
        {"type": "object",
         "properties": {"path": {"type": "string", "description": "File path relative to the project root."}},
         "required": ["path"]},
        read_file))
    reg.add(Tool(
        "get_time", "Get the current local date and time.",
        {"type": "object", "properties": {}},
        lambda: datetime.now().isoformat(timespec="seconds")))
    reg.add(Tool(
        FINISH, "End the task and give the final answer to the user.",
        {"type": "object",
         "properties": {"answer": {"type": "string", "description": "The complete final answer."}},
         "required": ["answer"]},
        lambda answer: answer))  # never run: the loop handles finish itself
    return reg
