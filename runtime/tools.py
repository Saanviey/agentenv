import json, os, subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

MAX_OUTPUT = 8000   # chars; stops one tool result from flooding the context window
FINISH = "finish"   # the loop ends the run when the model calls this tool
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")  # env vars with these in the name are hidden from commands

Risk = Literal["read", "write", "exec"]  # read = no side effects; write = changes files; exec = runs a program


class ToolError(Exception): ...  # expected failure; its message is shown to the model


# One tool = what the model sees (name, description, parameters) + what we run (fn) + how risky it is.
@dataclass
class Tool:
    name: str
    description: str             # the model reads this to decide when to call the tool
    parameters: dict[str, Any]   # JSON Schema for the arguments
    fn: Callable[..., str]
    risk: Risk = "read"          # the policy uses this to decide allow / ask


# In: arguments as the model sent them. Out: (dict, None) or (None, error text).
# Some models send arguments as a JSON string; parse it once so policy and run() see the same dict.
def parse_args(args: Any) -> tuple[dict | None, str | None]:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None, f"arguments are not valid JSON: {args!r}"
    args = args or {}
    if not isinstance(args, dict):
        return None, f"arguments must be a JSON object, got {type(args).__name__}"
    return args, None


# Holds the available tools; the loop asks it for schemas and to run calls.
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

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

        args, err = parse_args(args)
        if err:
            return f"error: {name}: {err}"

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


# In: workspace folder the tools may touch, max seconds per command. Out: registry with all tools.
def default_tools(root: str | Path = ".", command_timeout_s: float = 30) -> ToolRegistry:
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

    def write_file(path: str, content: str) -> str:
        f = safe_path(path)
        if f.is_dir():
            raise ToolError(f"'{path}' is a directory")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def edit_file(path: str, old: str, new: str) -> str:
        f = safe_path(path)
        if not f.is_file():
            raise ToolError(f"'{path}' is not a file")
        text = f.read_text()
        n = text.count(old)
        if n == 0:
            raise ToolError(f"old text not found in {path}. read_file it and copy the exact text, including spaces")
        if n > 1:
            raise ToolError(f"old text appears {n} times in {path}. include more surrounding lines so it matches once")
        f.write_text(text.replace(old, new, 1))
        return f"edited {path}: replaced 1 occurrence"

    def run_command(command: str) -> str:
        env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in SECRET_WORDS)}
        try:
            p = subprocess.run(command, shell=True, cwd=root, env=env, capture_output=True, text=True,
                               timeout=command_timeout_s, stdin=subprocess.DEVNULL)
            # shell=True: run through /bin/sh so pipes work; cwd=root: commands start in the workspace;
            # capture_output: collect stdout/stderr instead of printing; stdin=DEVNULL: a command waiting for input fails instead of hanging
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {command_timeout_s}s")
        out = f"exit code: {p.returncode}"
        if p.stdout:
            out += f"\n--- stdout ---\n{p.stdout}"
        if p.stderr:
            out += f"\n--- stderr ---\n{p.stderr}"
        return out

    path_prop = {"type": "string", "description": "Path relative to the project root."}
    reg = ToolRegistry()
    reg.add(Tool(
        "list_dir", "List a directory. Returns paths relative to the project root; folders end with '/'.",
        {"type": "object",
         "properties": {"path": {"type": "string", "description": "Directory relative to the project root. Default '.'"}}},
        list_dir))
    reg.add(Tool(
        "read_file", "Read a text file and return its contents.",
        {"type": "object", "properties": {"path": path_prop}, "required": ["path"]},
        read_file))
    reg.add(Tool(
        "write_file", "Create or overwrite a whole file. Prefer edit_file for small changes to existing files.",
        {"type": "object", "properties": {"path": path_prop, "content": {"type": "string", "description": "Full new file contents."}},
         "required": ["path", "content"]},
        write_file, risk="write"))
    reg.add(Tool(
        "edit_file", "Replace one exact piece of text in a file. `old` must appear exactly once; copy it exactly from read_file.",
        {"type": "object", "properties": {"path": path_prop,
                                          "old": {"type": "string", "description": "Exact text to replace."},
                                          "new": {"type": "string", "description": "Replacement text."}},
         "required": ["path", "old", "new"]},
        edit_file, risk="write"))
    reg.add(Tool(
        "run_command", f"Run a shell command in the project root (e.g. tests). Returns exit code, stdout and stderr. Times out after {command_timeout_s:g}s.",
        {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to run."}},
         "required": ["command"]},
        run_command, risk="exec"))
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
