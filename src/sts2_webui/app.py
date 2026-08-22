"""Observability webui for the Track 2 pod experiment.

Read-only over the game service's SQLite (mounted volume) and the pod's
session JSONL logs. Host network only — this service must NEVER be reachable
from the pod (eval leakage), which compose guarantees by network membership.

Run: uvicorn sts2_webui.app:app --port 8310
Env: STS2_DB, POD_HOME, PRICE_IN / PRICE_OUT ($/M tokens, for cost estimates).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

DB_PATH = os.environ.get("STS2_DB", "/data/sts2.sqlite3")
POD_HOME = Path(os.environ.get("POD_HOME", "/podhome"))
PRICE_IN = float(os.environ.get("PRICE_IN", "0.14"))    # $/M input tokens
PRICE_OUT = float(os.environ.get("PRICE_OUT", "1.00"))  # $/M output tokens
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="sts2 observability")


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        by_status = dict(conn.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall())
        finished = conn.execute(
            "SELECT status FROM runs WHERE status IN ('victory','defeat') ORDER BY id DESC LIMIT 30"
        ).fetchall()
        wins30 = sum(1 for r in finished if r["status"] == "victory")
        decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        attempts = conn.execute("SELECT COUNT(*) FROM action_attempts").fetchone()[0]
        illegal = conn.execute("SELECT COUNT(*) FROM action_attempts WHERE ok=0").fetchone()[0]
    sess = _sessions()
    return {
        "runs": total,
        "by_status": by_status,
        "winrate30": round(wins30 / len(finished), 3) if finished else None,
        "decisions": decisions,
        "attempts": attempts,
        "illegal": illegal,
        "sessions": len(sess),
        "tokens_in": sum(s["tokens_in"] for s in sess),
        "tokens_out": sum(s["tokens_out"] for s in sess),
        "cost_usd": round(sum(s["cost_usd"] for s in sess), 3),
    }


@app.get("/api/runs")
async def runs() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT r.*,
                   (SELECT COUNT(*) FROM decisions d WHERE d.run_id = r.id) AS n_decisions,
                   (SELECT COUNT(*) FROM action_attempts a WHERE a.run_id = r.id AND a.ok = 0) AS n_illegal,
                   (SELECT COUNT(*) FROM detail_queries q WHERE q.run_id = r.id) AS n_detail
            FROM runs r ORDER BY r.id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/runs/{run_id}/decisions")
async def run_decisions(run_id: int, start: int = 0, limit: int = 2000) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.idx, d.decision, d.serialized, d.action, d.action_ok, d.created_at,
                   (SELECT json_group_array(json_object('action', a.action, 'error', a.error))
                    FROM action_attempts a WHERE a.decision_id = d.id AND a.ok = 0) AS rejected
            FROM decisions d WHERE d.run_id = ? AND d.idx >= ? ORDER BY d.idx LIMIT ?
            """,
            (run_id, start, limit),
        ).fetchall()
    if not rows and start == 0:
        raise HTTPException(404, "no decisions for run")
    return [dict(r) for r in rows]


# ── pod session logs ───────────────────────────────────────────────────────

_session_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _parse_session(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "id": path.stem, "reason": None, "tokens_in": 0, "tokens_out": 0,
        "n_tools": 0, "n_turns": 0, "start": None, "end": None, "model": None,
    }
    with path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("type")
            info["end"] = e.get("t")
            if t == "session_start":
                info["start"] = e.get("t")
                info["model"] = e.get("model")
            elif t == "assistant":
                info["n_turns"] += 1
            elif t == "tool":
                info["n_tools"] += 1
            elif t == "session_end":
                info["reason"] = e.get("reason")
                tok = e.get("tokens") or {}
                info["tokens_in"] = tok.get("input", 0)
                info["tokens_out"] = tok.get("output", 0)
    info["cost_usd"] = round(
        info["tokens_in"] / 1e6 * PRICE_IN + info["tokens_out"] / 1e6 * PRICE_OUT, 4
    )
    if info["reason"] is None:
        info["reason"] = "running" if time.time() * 1000 - (info["end"] or 0) < 120_000 else "crash"
    return info


def _sessions() -> list[dict[str, Any]]:
    out = []
    sess_dir = POD_HOME / "sessions"
    if not sess_dir.is_dir():
        return out
    for path in sorted(sess_dir.glob("*.jsonl"), reverse=True):
        key = str(path)
        mtime = path.stat().st_mtime
        cached = _session_cache.get(key)
        if cached and cached[0] == mtime:
            out.append(cached[1])
            continue
        info = _parse_session(path)
        _session_cache[key] = (mtime, info)
        out.append(info)
    return out


@app.get("/api/sessions")
async def sessions() -> list[dict[str, Any]]:
    return _sessions()


@app.get("/api/pod/files")
async def pod_files() -> dict[str, Any]:
    """The agent's notes/skills/prompt — read-only window into its memory."""
    out = {}
    for name in ("PROMPT.md", "HANDOFF.md"):
        p = POD_HOME / name
        if p.is_file():
            out[name] = p.read_text(errors="replace")[:20000]
    for sub in ("notes", "skills", "bin"):
        d = POD_HOME / sub
        if d.is_dir():
            out[sub + "/"] = {
                f.name: f.read_text(errors="replace")[:20000]
                for f in sorted(d.iterdir()) if f.is_file() and f.name != ".keep"
            }
    return out
