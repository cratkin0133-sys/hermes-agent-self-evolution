"""SQLite store for usage signals and evolution runs (Phase 5).

Default location: ``~/.hermes/self-evolution/metrics.db`` (override with
``HERMES_EVOLUTION_DB``). Local only; nothing is sent anywhere.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, phase TEXT NOT NULL, target TEXT NOT NULL,
    uses INTEGER NOT NULL, corrections INTEGER NOT NULL, source TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, phase TEXT NOT NULL, target TEXT NOT NULL,
    accepted INTEGER NOT NULL, baseline REAL, evolved REAL, improvement REAL,
    output_dir TEXT, details TEXT
);
CREATE INDEX IF NOT EXISTS runs_target ON runs(phase, target, ts);
CREATE INDEX IF NOT EXISTS signals_target ON signals(phase, target, ts);
"""


def default_db_path() -> Path:
    env = os.getenv("HERMES_EVOLUTION_DB")
    return Path(env).expanduser() if env else Path.home() / ".hermes" / "self-evolution" / "metrics.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MetricsStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or default_db_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def record_signals(self, signals: list, source: str = "") -> None:
        ts = now()
        self.db.executemany(
            "INSERT INTO signals (ts, phase, target, uses, corrections, source) VALUES (?,?,?,?,?,?)",
            [(ts, s.phase, s.target, s.uses, s.corrections, source) for s in signals],
        )
        self.db.commit()

    def latest_signals(self) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT s.* FROM signals s JOIN (SELECT phase, target, MAX(ts) m FROM signals GROUP BY phase, target) l
               ON s.phase = l.phase AND s.target = l.target AND s.ts = l.m ORDER BY s.phase, s.target"""
        ).fetchall()

    def record_run(self, phase: str, target: str, metrics: dict) -> None:
        def f(*keys):
            for k in keys:
                if isinstance(metrics.get(k), (int, float)):
                    return float(metrics[k])
            return None
        self.db.execute(
            "INSERT INTO runs (ts, phase, target, accepted, baseline, evolved, improvement, output_dir, details) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (now(), phase, target, int(bool(metrics.get("accepted"))),
             f("baseline_score", "baseline_accuracy"), f("evolved_score", "evolved_accuracy", "best_score"),
             f("improvement"), metrics.get("output_dir"), json.dumps(metrics, default=str)[:20000]),
        )
        self.db.commit()

    def last_run(self, phase: str, target: str) -> Optional[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM runs WHERE phase=? AND target=? ORDER BY ts DESC, id DESC LIMIT 1", (phase, target)
        ).fetchone()

    def previous_scored_run(self, phase: str, target: str) -> Optional[sqlite3.Row]:
        """Latest accepted run for the target that recorded a numeric score (the baseline)."""
        return self.db.execute(
            "SELECT * FROM runs WHERE phase=? AND target=? AND accepted=1 AND evolved IS NOT NULL "
            "ORDER BY ts DESC, id DESC LIMIT 1",
            (phase, target),
        ).fetchone()

    def runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM runs ORDER BY ts DESC, id DESC LIMIT ?", (limit,)).fetchall()
