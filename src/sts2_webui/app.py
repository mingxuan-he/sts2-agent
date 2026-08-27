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


def _classify_tool(e: dict[str, Any]) -> str:
    """Bucket a tool call for the self-learning metrics."""
    n = e.get("name", "")
    args = str(e.get("args", {}))
    if n.startswith("game_"):
        return "game"
    if n == "bash":
        cmd = str((e.get("args") or {}).get("command", ""))
        if "8300" in cmd or ("fetch" in cmd and "game" in cmd.lower()):
            return "game_fetch"      # plays via raw fetch (the calcified workaround)
        if any(w in cmd for w in ("sessions", "notes", "skills")):
            return "self_read"       # inspects its own memory/logs
        return "misc"
    if n == "write_file" and any(w in args for w in ("notes", "PROMPT", "HANDOFF", "skills", "bin")):
        return "doc"
    if n == "finish_session":
        return "doc"
    if n == "read_file" and any(w in args for w in ("notes", "sessions", "skills")):
        return "self_read"
    return "misc"


def _parse_session(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "id": path.stem, "reason": None, "tokens_in": 0, "tokens_out": 0,
        "n_tools": 0, "n_turns": 0, "start": None, "end": None, "model": None,
        "game": 0, "game_fetch": 0, "doc": 0, "self_read": 0, "misc": 0, "doc_chars": 0,
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
                bucket = _classify_tool(e)
                info[bucket] += 1
                if bucket == "doc":
                    a = e.get("args") or {}
                    info["doc_chars"] += len(str(a.get("content", a.get("handoff", ""))))
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


# Per-act progression, cached once a run is finished (rows are immutable then).
# floors reset to 0/1 at each act boundary, so a single MAX(floor) undercounts
# any run that beats a boss — acts must be aggregated separately.
_run_acts_cache: dict[int, list[dict[str, Any]]] = {}


def _run_acts(conn: sqlite3.Connection, run_id: int, finished: bool) -> list[dict[str, Any]]:
    if run_id in _run_acts_cache:
        return _run_acts_cache[run_id]
    rows = conn.execute(
        """
        SELECT json_extract(raw_state, '$.context.act')      AS act,
               json_extract(raw_state, '$.context.act_name') AS act_name,
               MAX(json_extract(raw_state, '$.context.floor')) AS floor
        FROM decisions WHERE run_id = ?
        GROUP BY act, act_name ORDER BY act
        """,
        (run_id,),
    ).fetchall()
    acts = [dict(r) for r in rows if r["act"] is not None]
    if finished:
        _run_acts_cache[run_id] = acts
    return acts


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    sess = list(reversed(_sessions()))  # chronological
    total = {k: sum(s[k] for s in sess) for k in ("game", "game_fetch", "doc", "self_read", "misc", "doc_chars")}
    play = total["game"] + total["game_fetch"]

    with db() as conn:
        runs = []
        for r in conn.execute(
            """
            SELECT r.id, r.character, r.status, r.floor, r.started_at,
                   (SELECT COUNT(*) FROM decisions d WHERE d.run_id = r.id) AS n_decisions,
                   (SELECT COUNT(*) FROM action_attempts a WHERE a.run_id = r.id) AS n_attempts,
                   (SELECT COUNT(*) FROM action_attempts a WHERE a.run_id = r.id AND a.ok = 0) AS n_illegal,
                   (SELECT COUNT(*) FROM detail_queries q WHERE q.run_id = r.id) AS n_detail,
                   (SELECT MAX(json_extract(d.raw_state, '$.context.floor'))
                    FROM decisions d WHERE d.run_id = r.id) AS max_floor
            FROM runs r ORDER BY r.id
            """
        ).fetchall():
            d = dict(r)
            acts = _run_acts(conn, r["id"], d["status"] in ("victory", "defeat"))
            d["acts"] = acts
            d["act"] = acts[0]["act_name"] if acts else None           # act-1 variant
            d["total_floor"] = sum(a["floor"] or 0 for a in acts)      # cumulative
            d["top_act"] = acts[-1]["act_name"] if acts else None
            runs.append(d)

    # novelty tax: act-1 progress by act-1 variant (finished real runs only)
    act_floor: dict[str, list[int]] = {}
    for r in runs:
        if r["status"] in ("victory", "defeat") and r["acts"]:
            a1 = r["acts"][0]
            if a1["floor"]:
                act_floor.setdefault(a1["act_name"], []).append(a1["floor"])

    return {
        "tool_split": {
            "play": play,
            "play_via_fetch": total["game_fetch"],
            "doc": total["doc"],
            "self_read": total["self_read"],
            "misc": total["misc"],
        },
        "doc_chars": total["doc_chars"],
        "sessions": [
            {k: s[k] for k in ("id", "reason", "game", "game_fetch", "doc", "self_read", "doc_chars", "tokens_out")}
            for s in sess
        ],
        "runs": runs,
        "act_avg_floor": {
            a: round(sum(v) / len(v), 1) for a, v in act_floor.items()
        },
    }


# ── spire-codex art map (name -> image url), cached server-side ───────────

_codex_cache: tuple[float, dict[str, Any]] | None = None


def _fetch_codex() -> dict[str, Any]:
    import urllib.request

    out: dict[str, dict[str, str]] = {}
    for kind in ("relics", "monsters", "potions"):
        try:
            req = urllib.request.Request(
                f"https://spire-codex.com/api/{kind}?limit=1000",
                headers={"User-Agent": "sts2-agent-webui/0.1 (personal project)"},
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            items = data if isinstance(data, list) else data.get("items", [])
            out[kind] = {
                e["name"].lower(): "https://spire-codex.com" + e["image_url"]
                for e in items
                if e.get("name") and e.get("image_url")
            }
        except Exception:
            out[kind] = {}
    return out


@app.get("/api/codex/images")
async def codex_images() -> dict[str, Any]:
    global _codex_cache
    import asyncio

    if _codex_cache is None or time.time() - _codex_cache[0] > 86400:
        _codex_cache = (time.time(), await asyncio.to_thread(_fetch_codex))
    return _codex_cache[1]


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
