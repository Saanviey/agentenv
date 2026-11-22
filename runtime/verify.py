import os, re, subprocess
from pathlib import Path
from runtime.session import Session
from runtime.tools import safe_env

# Decides whether a run is really done. Base version: accepts anything, can't judge (used by tests and replay).
class Verifier:
    # In: the proposed final answer + what happened. Out: None to accept, or the reason to reject.
    def check_finish(self, answer: str, session: Session | None) -> str | None:
        return None

    # At the end of a run that did NOT finish properly. Out: True (changes are good), False (broken), None (can't tell).
    def changes_ok(self, session: Session) -> bool | None:
        return None

FILE_RE = re.compile(r"[\w./-]+\.(?:py|toml|md|txt|json|cfg|ini|ya?ml|js|ts)\b")
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}

# Evidence rules for a coding agent.
class CodingVerifier(Verifier):
    def __init__(self, test_command: str | None = None, timeout_s: float = 120):
        self.test_command, self.timeout_s = test_command, timeout_s

    def check_finish(self, answer, session):
        if session is None:
            return None
        changed = session.changed()
        if changed:
            files = ", ".join(changed)
            if self.test_command:
                code, out = self.run_tests(session)
                if code != 0:
                    return f"you changed {files} but the tests still fail:\n{out[-1500:]}\nFix the code before finishing."
            elif not session.checked_since_last_change():
                return (f"you changed {files} but haven't run a successful check (e.g. the tests) since your last change. "
                        "Run the tests with run_command, then call finish.")
        unread = unread_mentions(answer, session)
        if unread:
            return f"your answer talks about {', '.join(unread)} but you never read it. Read it first, then finish."
        return None

    def changes_ok(self, session):
        if not self.test_command:
            return None
        return self.run_tests(session)[0] == 0

    # Out: (exit code or None if timed out, output). Runs the project's tests itself; doesn't trust the model's word.
    def run_tests(self, session: Session) -> tuple[int | None, str]:
        try:
            p = subprocess.run(self.test_command, shell=True, cwd=session.root, env=safe_env(), capture_output=True,
                               text=True, timeout=self.timeout_s, stdin=subprocess.DEVNULL)
            return p.returncode, (p.stdout + p.stderr).strip()
        except subprocess.TimeoutExpired:
            return None, f"tests timed out after {self.timeout_s}s"

# Out: workspace files named in the answer that the agent never read, changed, or saw in a command's output (B9).
def unread_mentions(answer: str, session: Session) -> list[str]:
    seen_text = "\n".join([c.output for c in session.commands] + session.seen)
    touched = session.read | set(session.snapshots)
    out = []
    for name in sorted(set(FILE_RE.findall(answer))):
        name = name.strip("./")
        if not _exists(session.root, name) or name in seen_text:
            continue
        if not any(t == name or t.endswith("/" + name) or name.endswith("/" + t) for t in touched):
            out.append(name)
    return out

def _exists(root: Path, name: str) -> bool:
    if (root / name).is_file():
        return True
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]  # don't walk .venv etc.
        if Path(name).name in files and (Path(d) / Path(name).name).as_posix().endswith(name):
            return True
    return False
