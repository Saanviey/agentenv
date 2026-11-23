import json, sqlite3
from datetime import datetime
from pathlib import Path

# One row per run. `state` holds the latest checkpoint (JSON), overwritten after every action.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id        TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    task      TEXT NOT NULL,
    model     TEXT,
    status    TEXT NOT NULL,          -- running | interrupted | crashed | finish | unverified | no_finish | max_steps
    step      INTEGER NOT NULL DEFAULT 0,
    trace     TEXT,
    created   TEXT NOT NULL,
    updated   TEXT NOT NULL,
    state     TEXT                    -- JSON: messages, counters, session, context (see checkpoint.py)
);
CREATE INDEX IF NOT EXISTS runs_by_updated ON runs (workspace, updated);
"""
RESUMABLE = ("running", "interrupted", "crashed")  # "running" in the database = the process died without saying goodbye

# The session store: a small SQLite database file. sqlite3 is in Python's standard library; no server needed.
class Store:
    def __init__(self, path: str | Path = "state/agent.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row  # rows behave like dicts: row["status"]
        self.db.executescript(SCHEMA)

    def create(self, run_id: str, workspace: str, task: str, model: str, trace: str) -> None:
        now = _now()
        with self.db:  # `with` = one transaction: committed if the block succeeds, undone if it fails
            self.db.execute("INSERT INTO runs (id, workspace, task, model, status, trace, created, updated) "
                            "VALUES (?, ?, ?, ?, 'running', ?, ?, ?)", (run_id, workspace, task, model, trace, now, now))

    # Overwrites the run's checkpoint. Atomic: after a crash the database holds the old checkpoint or the new one, never half.
    def save(self, run_id: str, step: int, status: str, state: dict) -> None:
        with self.db:
            self.db.execute("UPDATE runs SET state = ?, step = ?, status = ?, updated = ? WHERE id = ?",
                            (json.dumps(state), step, status, _now(), run_id))

    # Out: the run as a dict (state parsed), or None.
    def get(self, run_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["state"] = json.loads(d["state"]) if d["state"] else None
        return d

    # Out: id of the most recently updated run (optionally only resumable ones), or None.
    def latest(self, resumable_only: bool = False) -> str | None:
        where = f"WHERE status IN ({', '.join('?' * len(RESUMABLE))})" if resumable_only else ""
        row = self.db.execute(f"SELECT id FROM runs {where} ORDER BY updated DESC, rowid DESC LIMIT 1",
                              RESUMABLE if resumable_only else ()).fetchone()
        return row["id"] if row else None

    def recent(self, limit: int = 15) -> list[dict]:
        rows = self.db.execute("SELECT id, status, step, task, updated FROM runs ORDER BY updated DESC, rowid DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.db.close()

def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")  # seconds alone tie when runs start in the same second
