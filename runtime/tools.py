import json, os, re, subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal
from runtime.session import Session

OFFLOAD_CHARS = 4000  # results longer than this are stored outside the conversation; the model sees a preview
PREVIEW_HEAD, PREVIEW_TAIL = 2000, 800
PAGE_CHARS = 3500     # max chars read_output returns at once (below OFFLOAD_CHARS, so pages are never offloaded again)
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
FINISH = "finish"   # the loop ends the run when the model calls this tool
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")  # env vars with these in the name are hidden from commands

# Out: a copy of the environment without secrets (API keys etc.), for anything we run.
def safe_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in SECRET_WORDS)}
    # No .pyc cache: Python checks it by mtime (1 s resolution) + size, so a same-size edit within a second
    # (a * b -> a + b) would run the OLD compiled code (B19).
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env

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
    aliases: dict[str, str] | None = None  # other names models use for an argument, e.g. line_start -> start_line (B26)


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
    def __init__(self, session: Session | None = None):
        self._tools: dict[str, Tool] = {}
        self.session = session  # what the tools did this run (evidence + snapshots); None = not tracked

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
        for alias, real in (tool.aliases or {}).items():
            if alias in args and real not in args:
                args[real] = args.pop(alias)

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

        return self._offload(out) if len(out) > OFFLOAD_CHARS else out

    # Big result: keep the full text in the session, give the model the start and end + how to get the rest.
    def _offload(self, out: str) -> str:
        if self.session is None:  # nowhere to store it: plain cut
            return out[:OFFLOAD_CHARS] + f"\n...[cut: {len(out) - OFFLOAD_CHARS} more chars]"
        oid = self.session.store_output(out)
        mid_end = len(out) - PREVIEW_TAIL
        return (f"{out[:PREVIEW_HEAD]}\n\n...[output #{oid} is {len(out)} chars; chars {PREVIEW_HEAD}-{mid_end} not shown. "
                f"Use read_output(id={oid}, offset={PREVIEW_HEAD}) to page through it, or search / read_file with line ranges.]...\n\n"
                f"{out[mid_end:]}")


# Out: every file under base, sorted, skipping hidden folders and SKIP_DIRS (.venv etc.).
def _walk_files(base: Path) -> list[Path]:
    files = []
    for d, dirs, names in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))  # editing dirs in place prunes the walk
        files += [Path(d) / n for n in sorted(names)]
    return files

# In: workspace folder the tools may touch, max seconds per command. Out: registry with all tools.
def default_tools(root: str | Path = ".", command_timeout_s: float = 30) -> ToolRegistry:
    root = Path(root).resolve()
    session = Session(root)

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

    def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        f = safe_path(path)
        if not f.is_file():
            raise ToolError(f"'{path}' is not a file")
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            raise ToolError(f"'{path}' is not a text file")
        session.note_read(str(f.relative_to(root)))
        if start_line is None and end_line is None:
            return text
        lines = text.splitlines(keepends=True)
        a, b = max(1, int(start_line or 1)), min(len(lines), int(end_line or len(lines)))
        if a > b:
            raise ToolError(f"{path} has {len(lines)} lines; asked for {a}-{b}")
        return f"[{path} lines {a}-{b} of {len(lines)}]\n" + "".join(lines[a - 1:b])

    def search(pattern: str, path: str = ".") -> str:
        base = safe_path(path)
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))  # not a valid regex: search for the literal text
        hits = []
        for f in ([base] if base.is_file() else _walk_files(base)):
            try:
                text = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(root)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 50:
                        break
            if len(hits) >= 50:
                hits.append("...[stopped at 50 matches; narrow the pattern or path]")
                break
        out = "\n".join(hits) or f"no matches for {pattern!r}"
        session.note_seen(out)
        return out

    def read_output(id: int, offset: int = 0) -> str:
        if not 1 <= int(id) <= len(session.outputs):
            raise ToolError(f"no output #{id}; ids go from 1 to {len(session.outputs)}")
        text, offset = session.outputs[int(id) - 1], max(0, int(offset))
        part = text[offset:offset + PAGE_CHARS]
        end = offset + len(part)
        more = f" Next: read_output(id={id}, offset={end})." if end < len(text) else " (end)"
        session.note_seen(part)
        return f"[output #{id}, chars {offset}-{end} of {len(text)}.{more}]\n{part}"

    # Every file change goes through here: snapshot the original, write, and undo at once if Python no longer parses.
    def change(f: Path, new_text: str) -> None:
        rel = str(f.relative_to(root))
        before = f.read_text() if f.is_file() else None
        session.before_change(rel, before)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(new_text)
        if f.suffix == ".py":
            try:
                compile(new_text, rel, "exec")  # parse only, nothing runs
            except SyntaxError as e:
                if before is None:
                    f.unlink()
                else:
                    f.write_text(before)
                raise ToolError(f"that change would make {rel} invalid Python ({e.msg}, line {e.lineno}), so it was undone. "
                                "Fix the code and try again.")

    def write_file(path: str, content: str) -> str:
        f = safe_path(path)
        if f.is_dir():
            raise ToolError(f"'{path}' is a directory")
        change(f, content)
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
        change(f, text.replace(old, new, 1))
        return f"edited {path}: replaced 1 occurrence"

    def revert_file(path: str) -> str:
        f = safe_path(path)
        rel = str(f.relative_to(root))
        if rel not in session.snapshots:
            raise ToolError(f"you haven't changed {path} in this run, so there is nothing to revert")
        session.restore(rel)
        return f"restored {path} to how it was at the start of the run"

    def run_command(command: str) -> str:
        try:
            p = subprocess.run(command, shell=True, cwd=root, env=safe_env(), capture_output=True, text=True,
                               timeout=command_timeout_s, stdin=subprocess.DEVNULL)
            # shell=True: run through /bin/sh so pipes work; cwd=root: commands start in the workspace;
            # capture_output: collect stdout/stderr instead of printing; stdin=DEVNULL: a command waiting for input fails instead of hanging
        except subprocess.TimeoutExpired:
            session.note_command(command, None, "")
            raise ToolError(f"command timed out after {command_timeout_s}s")
        out = f"exit code: {p.returncode}"
        if p.stdout:
            out += f"\n--- stdout ---\n{p.stdout}"
        if p.stderr:
            out += f"\n--- stderr ---\n{p.stderr}"
        session.note_command(command, p.returncode, out)
        return out

    path_prop = {"type": "string", "description": "Path relative to the project root."}
    reg = ToolRegistry(session)
    reg.add(Tool(
        "list_dir", "List a directory. Returns paths relative to the project root; folders end with '/'.",
        {"type": "object",
         "properties": {"path": {"type": "string", "description": "Directory relative to the project root. Default '.'"}}},
        list_dir))
    reg.add(Tool(
        "read_file", "Read a text file. For big files, pass start_line/end_line (1-based, inclusive) to read only part of it.",
        {"type": "object", "properties": {"path": path_prop,
                                          "start_line": {"type": "integer", "description": "First line to read (optional)."},
                                          "end_line": {"type": "integer", "description": "Last line to read (optional)."}},
         "required": ["path"]},
        read_file, aliases={"line_start": "start_line", "line_end": "end_line"}))
    reg.add(Tool(
        "search", "Search file contents with a regular expression. Returns matching lines as path:line: text (max 50).",
        {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regex (or plain text) to look for."},
                                          "path": {"type": "string", "description": "File or folder to search, relative to the project root. Default '.'"}},
         "required": ["pattern"]},
        search))
    reg.add(Tool(
        "read_output", "Page through a long tool output that was cut (the message says its id and the next offset).",
        {"type": "object", "properties": {"id": {"type": "integer", "description": "Output id from the cut message."},
                                          "offset": {"type": "integer", "description": "Character to start from. Default 0."}},
         "required": ["id"]},
        read_output))
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
        "revert_file", "Undo all your changes to one file in this run, restoring it to how it was at the start.",
        {"type": "object", "properties": {"path": path_prop}, "required": ["path"]},
        revert_file, risk="write"))
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
