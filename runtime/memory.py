import argparse, hashlib, re
from datetime import date
from pathlib import Path

MAX_FACTS = 40      # the file stops growing here; delete old lines to make room
MAX_FACT_CHARS = 300
MAX_PROMPT_CHARS = 2000
HEADER = ("Project memory: notes saved by earlier runs in this folder. They may be outdated or wrong; "
          "check them before relying on them.")

# One plain markdown file per workspace, kept OUTSIDE the workspace (state/memory/<folder>-<hash>.md), so you can read and edit it.
def memory_path(workspace: str | Path, state_dir: str | Path = "state") -> Path:
    ws = Path(workspace).resolve()
    h = hashlib.sha1(str(ws).encode()).hexdigest()[:8]  # same folder -> same file; different folders with the same name don't clash
    return Path(state_dir) / "memory" / f"{ws.name or 'root'}-{h}.md"

# Out: the facts, one per "- " line (anything else in the file is ignored, so you can add your own headings).
def load_facts(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line[2:].strip() for line in path.read_text().splitlines() if line.startswith("- ")]

# Out: text to add to the system prompt at the start of a run, or "" if there's nothing.
def prompt_block(path: Path | None) -> str:
    facts = load_facts(path) if path else []
    if not facts:
        return ""
    text = "\n".join(f"- {f}" for f in facts)
    if len(text) > MAX_PROMPT_CHARS:
        text = text[:MAX_PROMPT_CHARS] + "\n- ...(more in the memory file)"
    return f"{HEADER}\n{text}"

# In: the memory file, a fact, who saved it. Out: a message for the agent. Raises ValueError for an unusable fact.
def add_fact(path: Path, fact: str, source: str = "agent") -> str:
    fact = re.sub(r"\s+", " ", fact).strip()  # one line, no extra spaces
    if not fact:
        raise ValueError("the fact is empty")
    if len(fact) > MAX_FACT_CHARS:
        raise ValueError(f"keep facts under {MAX_FACT_CHARS} characters; this one is {len(fact)}")
    facts = load_facts(path)
    if any(f.split(" (")[0].lower() == fact.lower() for f in facts):
        return "already in project memory"
    if len(facts) >= MAX_FACTS:
        raise ValueError(f"project memory is full ({MAX_FACTS} facts); the user can delete old lines in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"- {fact} ({source}, {date.today().isoformat()})\n")
    return f"saved to project memory ({len(facts) + 1}/{MAX_FACTS} facts)"

def main():
    ap = argparse.ArgumentParser(prog="python -m runtime.memory", description="show or manage a workspace's project memory")
    ap.add_argument("command", choices=["show", "path", "add", "clear"])
    ap.add_argument("fact", nargs="*", help="for add: the fact")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--state-dir", default="state")
    a = ap.parse_args()
    path = memory_path(a.workspace, a.state_dir)
    if a.command == "path":
        print(path)
    elif a.command == "show":
        print(path.read_text() if path.is_file() else f"(no memory yet for {Path(a.workspace).resolve()})")
    elif a.command == "add":
        print(add_fact(path, " ".join(a.fact), source="user"))
    elif a.command == "clear":
        path.unlink(missing_ok=True)
        print(f"cleared {path}")

if __name__ == "__main__":
    main()
