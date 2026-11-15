import re, sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Literal
from runtime.tools import Tool

# What the policy says about one tool call.
@dataclass
class Decision:
    action: Literal["allow", "deny", "ask"]
    reason: str = ""

# Commands refused outright, whatever the user would say. (pattern, why)
DENY_COMMANDS = [
    (r"\brm\s+-\w*[rRf]", "recursive or forced delete"),
    (r"\bsudo\b|(^|[;&|]\s*)su\b", "privilege escalation"),
    (r"\b(curl|wget)\b.*\|\s*\w*sh\b", "piping a download into a shell"),
    (r"\bmkfs\b|\bdd\s+if=", "disk-level write"),
    (r"\b(shutdown|reboot|halt)\b", "power control"),
    (r":\(\)\s*\{", "fork bomb"),
    (r"\bchmod\s+-R\b|\bchown\b", "bulk permission change"),
    (r"\bgit\s+(push|reset\s+--hard|clean)\b", "destructive or remote git command"),
    (r">\s*/(?!dev/null)", "redirecting output to an absolute path"),
    (r"(^|\s)~|\$HOME|\.\./", "touching paths outside the workspace"),
]
PROTECTED = (".git", ".venv")  # folders write tools may never touch

# In: the tool and its parsed args. Out: allow (just run it), deny (refuse), or ask (needs approval).
def check(tool: Tool, args: dict) -> Decision:
    if tool.risk == "read":
        return Decision("allow")
    if tool.risk == "write":
        parts = PurePosixPath(str(args.get("path", ""))).parts
        if any(p in PROTECTED for p in parts):
            return Decision("deny", f"writes to {'/'.join(parts)} are not allowed (protected folder)")
        return Decision("ask", f"{tool.name} {args.get('path')}")
    if tool.risk == "exec":
        cmd = str(args.get("command", ""))
        for pattern, why in DENY_COMMANDS:
            if re.search(pattern, cmd):
                return Decision("deny", f"{why}: {cmd!r}")
        return Decision("ask", f"run: {cmd}")
    return Decision("deny", f"unknown risk level {tool.risk!r}")

# An approver: In: tool name, args, reason. Out: True to allow.
Approver = Callable[[str, dict, str], bool]

def ask_user(name: str, args: dict, reason: str) -> bool:
    if not sys.stdin.isatty():  # no human to ask (piped / scripted run): refuse
        return False
    print(f"\n  agent wants to: {reason}", file=sys.stderr)
    if name in ("write_file", "edit_file"):
        preview = args.get("content") or f"- {args.get('old')}\n+ {args.get('new')}"
        print("  " + str(preview)[:800].replace("\n", "\n  "), file=sys.stderr)
    return input("  allow? [y/N] ").strip().lower() in ("y", "yes")

def approve_all(name: str, args: dict, reason: str) -> bool:
    return True

def deny_all(name: str, args: dict, reason: str) -> bool:
    return False
