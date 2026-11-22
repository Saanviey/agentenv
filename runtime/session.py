from dataclasses import dataclass, field
from pathlib import Path

# One command the agent ran, and its outcome.
@dataclass
class CommandRun:
    command: str
    exit_code: int | None    # None = timed out
    output: str
    order: int               # when it happened, relative to other actions in this run

# What the tools did during one run: files read, files changed (with the original saved), commands run.
# The verifier uses it as evidence; rollback uses the snapshots. Filled in by the tools, never by the model.
@dataclass
class Session:
    root: Path
    read: set[str] = field(default_factory=set)                     # relative paths read
    snapshots: dict[str, str | None] = field(default_factory=dict)  # path -> content at FIRST change (None = file didn't exist)
    commands: list[CommandRun] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)                # full text of big tool results (offloaded); id = index + 1
    seen: list[str] = field(default_factory=list)                   # other tool output the agent saw (search results)
    last_change: int = 0     # order of the most recent file change
    _clock: int = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def note_read(self, rel: str) -> None:
        self.read.add(rel)

    # Call before writing a file. Saves the original only the first time, so restore goes back to the start of the run.
    def before_change(self, rel: str, current: str | None) -> None:
        self.snapshots.setdefault(rel, current)
        self.last_change = self._tick()

    def note_command(self, command: str, exit_code: int | None, output: str) -> None:
        self.commands.append(CommandRun(command, exit_code, output, self._tick()))

    # Keeps a big tool result out of the conversation. Out: its id, for read_output.
    def store_output(self, text: str) -> int:
        self.outputs.append(text)
        return len(self.outputs)

    def note_seen(self, text: str) -> None:
        self.seen.append(text)

    # Out: what the tools recorded, as plain text. Written by the runtime, so it can't be hallucinated (used under summaries).
    def facts(self) -> str:
        lines = []
        if self.read:
            lines.append("- files read: " + ", ".join(sorted(self.read)))
        if self.snapshots:
            lines.append("- files changed: " + (", ".join(self.changed()) or "none (changes were undone)"))
        if self.commands:
            c = self.commands[-1]
            result = "timed out" if c.exit_code is None else f"exit code {c.exit_code}"
            lines.append(f"- last command: `{c.command}` -> {result} ({len(self.commands)} commands so far)")
        return "\n".join(lines)

    # Out: files whose content now differs from the start of the run.
    def changed(self) -> list[str]:
        return [rel for rel, orig in self.snapshots.items() if self._current(rel) != orig]

    # Out: True if some command succeeded (exit 0) after the last file change.
    def checked_since_last_change(self) -> bool:
        return any(c.exit_code == 0 and c.order > self.last_change for c in self.commands)

    def restore(self, rel: str) -> None:
        f, orig = self.root / rel, self.snapshots[rel]
        if orig is None:
            f.unlink(missing_ok=True)
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(orig)

    # Out: the files that were put back.
    def restore_all(self) -> list[str]:
        changed = self.changed()
        for rel in changed:
            self.restore(rel)
        return changed

    def _current(self, rel: str) -> str | None:
        f = self.root / rel
        return f.read_text() if f.is_file() else None
