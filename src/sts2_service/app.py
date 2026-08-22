"""Game service: FastAPI over headless sts2-cli.

The pod agent's ONLY interface to the game. Engine subprocesses live on the
host side; the pod reaches this HTTP API and nothing else, so state-editing
cheats are impossible by construction and the SQLite log is tamper-proof
ground truth.

API (STS2MCP-shaped, see docs/track2-pod-design.md):

    POST /runs                        {character, ascension?, pool?} → new run, server-assigned seed
    GET  /runs/{run}/state?format=    text (canonical compact obs) | json (raw CLI state)
    POST /runs/{run}/action           {"action": "...", "args": {...}} → new state
    GET  /runs/{run}/deck             full deck w/ upgrade status + descriptions
    GET  /runs/{run}/piles            draw (SORTED — order hidden) / discard / exhaust
    GET  /runs/{run}/relics           relics w/ descriptions + counters
    GET  /runs/{run}/potions          potions w/ descriptions
    GET  /runs/{run}/map              full act map incl. boss
    DELETE /runs/{run}                abandon run
    GET  /runs                        list runs (host-side observability)

Run:  uvicorn sts2_service.app:app --host 127.0.0.1 --port 8300
Env:  STS2_DB (default data/sts2.sqlite3), STS2_ALLOW_EVAL=1 to permit pool=eval.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from headless_env import HeadlessEnv  # noqa: E402

from . import serializer  # noqa: E402
from .db import Database  # noqa: E402
from .seeds import generate_pools  # noqa: E402

CHARACTERS = {"Ironclad", "Silent", "Defect", "Necrobinder", "Regent"}
TERMINAL = {"game_over"}

db = Database(os.environ.get("STS2_DB", REPO_ROOT / "data" / "sts2.sqlite3"))
if db.seed_pools_empty():
    db.insert_seeds(generate_pools())
_swept = db.sweep_orphaned_runs()
if _swept:
    print(f"[startup] swept {_swept} orphaned active run(s) -> abandoned")

app = FastAPI(title="sts2 game service")


class RunSession:
    def __init__(self, run_uuid: str, db_id: int, env: HeadlessEnv, seed: str, pool: str):
        self.run_uuid = run_uuid
        self.db_id = db_id
        self.env = env
        self.seed = seed
        self.pool = pool
        self.idx = -1                       # decision sequence counter
        self.decision_id: int | None = None  # DB id of the currently-served decision
        self.raw_state: dict[str, Any] = {}
        self.serialized: str = ""
        self.done = False
        self.lock = asyncio.Lock()

    def record_state(self, raw: dict[str, Any]) -> None:
        self.raw_state = raw
        self.serialized = serializer.serialize(raw)
        self.idx += 1
        self.decision_id = db.add_decision(
            self.db_id, self.idx, raw.get("decision", raw.get("type", "?")), raw, self.serialized
        )


_runs: dict[str, RunSession] = {}


def _get_run(run_uuid: str) -> RunSession:
    session = _runs.get(run_uuid)
    if session is None:
        raise HTTPException(404, f"unknown run {run_uuid}")
    return session


class NewRunRequest(BaseModel):
    character: str = "Ironclad"
    ascension: int = Field(0, ge=0, le=20)
    pool: str = Field("train", pattern="^(train|eval)$")


class ActionRequest(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)


def _state_payload(session: RunSession, fmt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run": session.run_uuid,
        "seed": session.seed,
        "decision": session.raw_state.get("decision"),
        "done": session.done,
    }
    if fmt == "json":
        payload["state"] = session.raw_state
    else:
        payload["state"] = session.serialized
    return payload


async def _finish_if_over(session: RunSession) -> None:
    raw = session.raw_state
    if raw.get("decision") in TERMINAL:
        session.done = True
        status = "victory" if raw.get("victory") else "defeat"
        db.finish_run(session.db_id, status, raw.get("act"), raw.get("floor"))
        await session.env.close()


@app.post("/runs")
async def create_run(req: NewRunRequest) -> dict[str, Any]:
    if req.character not in CHARACTERS:
        raise HTTPException(422, f"character must be one of {sorted(CHARACTERS)}")
    if req.pool == "eval" and os.environ.get("STS2_ALLOW_EVAL") != "1":
        raise HTTPException(403, "eval pool disabled (service not started in eval mode)")

    seed = db.pick_seed(req.pool, req.character)
    if seed is None:
        raise HTTPException(500, "seed pool empty")

    run_uuid = uuid.uuid4().hex[:12]
    env = HeadlessEnv(character=req.character, ascension=req.ascension, seed=seed)
    try:
        state = await asyncio.wait_for(env.start(), timeout=180)
    except Exception as exc:
        await env.close()
        raise HTTPException(500, f"engine start failed: {exc}") from exc

    db_id = db.create_run(run_uuid, seed, req.character, req.ascension, req.pool,
                          snapshot_id=os.environ.get("STS2_SNAPSHOT_ID"))
    session = RunSession(run_uuid, db_id, env, seed, req.pool)
    session.record_state(state)
    _runs[run_uuid] = session
    await _finish_if_over(session)
    return _state_payload(session, "text")


@app.get("/runs/{run_uuid}/state")
async def get_state(run_uuid: str, format: str = "text") -> dict[str, Any]:
    session = _get_run(run_uuid)
    return _state_payload(session, format)


@app.post("/runs/{run_uuid}/action")
async def post_action(run_uuid: str, req: ActionRequest) -> dict[str, Any]:
    session = _get_run(run_uuid)
    async with session.lock:
        if session.done:
            raise HTTPException(409, "run is over")
        try:
            result = await asyncio.wait_for(
                session.env.action(req.action, req.args or None), timeout=120
            )
        except Exception as exc:
            db.record_action(session.db_id, session.decision_id, req.model_dump(), ok=False,
                             error=str(exc))
            db.finish_run(session.db_id, "error", None, None)
            session.done = True
            await session.env.close()
            raise HTTPException(500, f"engine error: {exc}") from exc

        if result.get("type") == "error":
            # Illegal/invalid action: run continues, current state unchanged.
            db.record_action(session.db_id, session.decision_id, req.model_dump(), ok=False,
                             error=result.get("message"))
            payload = _state_payload(session, "text")
            # Echo back exactly what was received so key/type mismatches are
            # unmissable (e.g. sending args={"option": "1"} to choose_option).
            payload["error"] = (
                f"{result.get('message', 'invalid action')} "
                f"(received: action={req.action!r}, args={json.dumps(req.args)})"
            )
            return payload

        db.record_action(session.db_id, session.decision_id, req.model_dump(), ok=True)
        session.record_state(result)
        await _finish_if_over(session)
        return _state_payload(session, "text")


@app.delete("/runs/{run_uuid}")
async def abandon_run(run_uuid: str) -> dict[str, Any]:
    session = _get_run(run_uuid)
    async with session.lock:
        if not session.done:
            db.finish_run(session.db_id, "abandoned",
                          (session.raw_state.get("context") or {}).get("act"),
                          (session.raw_state.get("context") or {}).get("floor"))
            session.done = True
            await session.env.close()
    del _runs[run_uuid]
    return {"run": run_uuid, "status": "abandoned"}


@app.get("/runs")
async def list_runs() -> list[dict[str, Any]]:
    return [
        {
            "run": s.run_uuid, "seed": s.seed, "pool": s.pool,
            "decision": s.raw_state.get("decision"), "done": s.done,
            "decisions": s.idx + 1,
        }
        for s in _runs.values()
    ]


# ── detail endpoints (full information set, on demand, logged) ────────────


def _detail(session: RunSession, endpoint: str) -> None:
    db.log_detail_query(session.db_id, session.decision_id, endpoint)


@app.get("/runs/{run_uuid}/deck")
async def get_deck(run_uuid: str) -> dict[str, Any]:
    session = _get_run(run_uuid)
    _detail(session, "deck")
    player = session.raw_state.get("player") or {}
    return {"run": run_uuid, "deck": serializer.render_deck(player)}


@app.get("/runs/{run_uuid}/relics")
async def get_relics(run_uuid: str) -> dict[str, Any]:
    session = _get_run(run_uuid)
    _detail(session, "relics")
    player = session.raw_state.get("player") or {}
    return {"run": run_uuid, "relics": serializer.render_relics(player)}


@app.get("/runs/{run_uuid}/potions")
async def get_potions(run_uuid: str) -> dict[str, Any]:
    session = _get_run(run_uuid)
    _detail(session, "potions")
    player = session.raw_state.get("player") or {}
    return {"run": run_uuid, "potions": serializer.render_potions(player)}


@app.get("/runs/{run_uuid}/piles")
async def get_piles(run_uuid: str) -> dict[str, Any]:
    session = _get_run(run_uuid)
    if session.done:
        raise HTTPException(409, "run is over")
    async with session.lock:
        _detail(session, "piles")
        result = await session.env._send({"cmd": "get_piles"})
    if result.get("type") != "piles":
        raise HTTPException(409, result.get("message", "not in combat"))
    return {"run": run_uuid, "piles": serializer.render_piles(result)}


@app.get("/runs/{run_uuid}/map")
async def get_map(run_uuid: str) -> dict[str, Any]:
    session = _get_run(run_uuid)
    if session.done:
        raise HTTPException(409, "run is over")
    async with session.lock:
        _detail(session, "map")
        result = await session.env.get_map()
    if result.get("type") != "map":
        raise HTTPException(409, result.get("message", "no map available"))
    return {"run": run_uuid, "map": serializer.render_map(result)}
