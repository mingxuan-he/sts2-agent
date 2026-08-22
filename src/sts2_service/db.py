"""SQLite persistence for the game service.

Single-writer, owned by the service process. Schema mirrors the spirebird
scenes.json shape (state, action per decision) so a replay UI is nearly free.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS seeds (
    seed      TEXT PRIMARY KEY,
    pool      TEXT NOT NULL CHECK (pool IN ('train', 'eval'))
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uuid    TEXT UNIQUE NOT NULL,
    seed        TEXT NOT NULL,
    character   TEXT NOT NULL,
    ascension   INTEGER NOT NULL DEFAULT 0,
    pool        TEXT NOT NULL,
    snapshot_id TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'victory', 'defeat', 'abandoned', 'error')),
    act         INTEGER,
    floor       INTEGER,
    started_at  REAL NOT NULL,
    ended_at    REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    idx         INTEGER NOT NULL,          -- 0-based sequence within the run
    decision    TEXT NOT NULL,             -- combat_play, map_select, ...
    raw_state   TEXT NOT NULL,             -- full CLI JSON (ground truth)
    serialized  TEXT NOT NULL,             -- exact text served to the agent
    action      TEXT,                      -- chosen action JSON (filled on POST /action)
    action_ok   INTEGER,                   -- 1 = accepted, 0 = CLI returned error
    created_at  REAL NOT NULL,
    UNIQUE (run_id, idx)
);
CREATE TABLE IF NOT EXISTS action_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    decision_id INTEGER REFERENCES decisions(id),
    action      TEXT NOT NULL,
    ok          INTEGER NOT NULL,          -- 0 = rejected by engine (illegal/invalid)
    error       TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS detail_queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    decision_id INTEGER REFERENCES decisions(id),
    endpoint    TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_seed ON runs(seed);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ── seeds ──
    def seed_pools_empty(self) -> bool:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM seeds").fetchone()
        return n == 0

    def insert_seeds(self, seeds: list[tuple[str, str]]) -> None:
        self._conn.executemany("INSERT OR IGNORE INTO seeds (seed, pool) VALUES (?, ?)", seeds)
        self._conn.commit()

    def pick_seed(self, pool: str, character: str) -> str | None:
        """Least-played seed in the pool for this character; ties break lexically."""
        row = self._conn.execute(
            """
            SELECT s.seed FROM seeds s
            LEFT JOIN runs r ON r.seed = s.seed AND r.character = ?
            WHERE s.pool = ?
            GROUP BY s.seed
            ORDER BY COUNT(r.id) ASC, s.seed ASC
            LIMIT 1
            """,
            (character, pool),
        ).fetchone()
        return row[0] if row else None

    def sweep_orphaned_runs(self) -> int:
        """Mark 'active' runs as abandoned. Call at service startup: engine
        subprocesses die with the service, so any active row is stale."""
        cur = self._conn.execute(
            "UPDATE runs SET status = 'abandoned', ended_at = ? WHERE status = 'active'",
            (time.time(),),
        )
        self._conn.commit()
        return cur.rowcount

    # ── runs ──
    def create_run(self, run_uuid: str, seed: str, character: str,
                   ascension: int, pool: str, snapshot_id: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (run_uuid, seed, character, ascension, pool, snapshot_id, started_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_uuid, seed, character, ascension, pool, snapshot_id, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, act: int | None, floor: int | None) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, act = ?, floor = ?, ended_at = ? WHERE id = ?",
            (status, act, floor, time.time(), run_id),
        )
        self._conn.commit()

    # ── decisions ──
    def add_decision(self, run_id: int, idx: int, decision: str,
                     raw_state: dict[str, Any], serialized: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO decisions (run_id, idx, decision, raw_state, serialized, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, idx, decision, json.dumps(raw_state, ensure_ascii=False), serialized, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_action(self, run_id: int, decision_id: int, action: dict[str, Any],
                      ok: bool, error: str | None = None) -> None:
        """Log every attempt; the decision row keeps only the final accepted action."""
        blob = json.dumps(action, ensure_ascii=False)
        self._conn.execute(
            "INSERT INTO action_attempts (run_id, decision_id, action, ok, error, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, decision_id, blob, int(ok), error, time.time()),
        )
        if ok:
            self._conn.execute(
                "UPDATE decisions SET action = ?, action_ok = 1 WHERE id = ?",
                (blob, decision_id),
            )
        self._conn.commit()

    def log_detail_query(self, run_id: int, decision_id: int | None, endpoint: str) -> None:
        self._conn.execute(
            "INSERT INTO detail_queries (run_id, decision_id, endpoint, created_at) VALUES (?, ?, ?, ?)",
            (run_id, decision_id, endpoint, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
