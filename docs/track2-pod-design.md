# Track 2 design: self-improving harness pod

Status: agreed design, pre-implementation (2026-08-22). Complements
`docs/model-selection-2026-08.md` (runtime model + costs) and the README's Track 2
overview.

## Architecture

```
┌─ VPS host ───────────────────────────────────────────────────────────┐
│                                                                      │
│  game service (host side)              pod (the sandbox)             │
│  ┌──────────────────────────────┐     ┌──────────────────────────┐   │
│  │ FastAPI wrapping             │◄────│ agent harness            │   │
│  │ HeadlessEnv / sts2-cli       │ http│ (pi-agent-core, node)    │   │
│  │ • serializer (server-side)   │     │ tools: game, bash,       │   │
│  │ • seed pools (train / eval)  │     │ notes, detail queries    │   │
│  │ • SQLite logging             │     │ /pod volume = its home ──┼───┼── snapshots
│  └──────────────────────────────┘     └──────────────────────────┘   │
│            ▲                                    │                    │
│  webapp (later, host network only,              ▼                    │
│  never reachable from pod)            egress proxy (allowlist:       │
│                                       LLM API domain only)           │
└──────────────────────────────────────────────────────────────────────┘
```

Principles:
- **Game engine outside the sandbox.** The pod can only reach the JSON API; it can
  never touch DLLs, savefiles, or logs. State-editing cheats impossible by
  construction; server-side logs are tamper-proof ground truth.
- **Egress allowlist is load-bearing.** STS2 is post-cutoff content; open internet
  would let the agent read wikis and destroy attribution of what it learned.
  `internal: true` Docker network + tiny forward proxy (tinyproxy/squid) whose
  allowlist is only the LLM API domain (OpenRouter initially).
- **Contamination policy.** Pod filesystem must not contain decompiled game source,
  spire-codex dumps, or expert replays unless a phase deliberately grants them.
- **Plain Docker is enough.** Threat model is "agent cheats at a card game," not
  hostile escape. Non-root user, --memory/--cpus/--pids-limit caps.

## Game service API (STS2MCP-shaped)

STS2MCP (github.com/Gennadiyev/STS2MCP) precedents we adopt: single GET-state /
POST-action pair, `format=` param for server-rendered LLM views (proven in Yui's
co-op runs), same action vocabulary as `sts2_client.py`, and state-as-response on
every POST (one round-trip per decision). Matching its shape means one client works
against both backends: our headless service and the live-game mod.

Extensions:

```
POST /runs                    → new run; SERVER assigns seed (train or eval pool)
GET  /run/state?format=text   → compact serialized observation (canonical)
POST /run/action              → {"action": "play_card", ...} → new compact state
GET  /run/deck                → full deck, upgrade status, descriptions
GET  /run/piles               → draw (SORTED — order hidden), discard, exhaust
GET  /run/relics              → relics with descriptions + counters
GET  /run/potions             → potions with descriptions
GET  /run/map                 → full act map: node types, paths, boss
```

- Draw pile is returned sorted, never in draw order — the real game shows contents
  but not order; leaking order is Frozen-Eye-grade info reserved for the seed-flag
  experiment.
- Seeds are issued server-side. Train pool for the loop; held-out eval pool only in
  eval mode. The agent never picks seeds.

## Serializer (the observation contract)

- Lives in the game service (Python), shared verbatim with Track 1 — pod
  trajectories stay valid SFT data and a trained LoRA drops into the pod with no
  distribution shift. The TS pod receives finished text, never raw JSON.
- **Compact by default, everything on demand.** The per-turn observation stays lean
  (~400 tokens); the full information set (deck, piles, relics+desc, potions+desc,
  map, buffs/debuffs) is one detail-endpoint call away. The serializer must never
  SHRINK the agent's information set relative to what a human player can see —
  only defer it behind a query.
- Detail queries are logged: information-seeking becomes measurable behavior
  (e.g. "checked map before pathing" vs winrate).
- Deterministic: same state → byte-identical text, stable ordering (prompt caching
  + comparable evals).
- Known gaps in Yui's serializer to fix when porting: relics absent from combat
  view; pile contents reduced to counts.

## Logging: SQLite

Single-writer SQLite owned by the game service (systemd supervises the process;
data lives in the DB, not journald).

- `runs`: id, seed, character, ascension, pool (train/eval), snapshot_id of pod,
  outcome, floors, timestamps.
- `decisions`: run_id, floor, turn, state JSON, serialized text, legal actions,
  chosen action, latency, tokens, cost.
- `detail_queries`: run_id, decision_id, endpoint, timestamp.

This is ~the spirebird scenes.json schema → replay UI later is nearly free.
Snapshot the DB alongside pod snapshots. Raw sts2-cli JSONL kept as backup.

## Pod harness: pi-agent-core

`@earendil-works/pi-agent-core` (badlogic/pi-mono, MIT, TypeScript) — vendored
into /pod, not node_modules, so "the agent edits its own harness" means editing
files in its home.

Fit (verified 2026-08-22):
- Tools = {name, description, TypeBox schema, execute} → game_action, detail
  queries, bash, notes.
- pi-ai supports OpenRouter + any OpenAI-compatible endpoint → commodity inference
  now, Tinker/PI-served LoRA later. Token/cost tracking built in.
- Extension points map onto the experiment: `transformContext` = skill-file
  injection + history pruning (the context-evolution knob), `before/afterToolCall`
  = guardrails + mirror actions to DB, `shouldStopAfterTurn` = Ralph-loop pacing,
  SQLite session backend package available.

Pod image: node + vendored harness; entrypoint = restart-on-exit loop (Ralph
loop); /pod volume = notes, skills, harness code — everything the agent may evolve.

## Snapshots & eval

- Snapshot = tar the /pod volume (+ DB copy). Rollback = restore.
- Eval = fresh container from a snapshot, pointed at the eval seed pool, fixed
  seed battery. Winrate vs accumulated-context-size is the headline curve.
- Swarm phase: skill-library edits merge only after winning an A/B on a fixed
  seed battery (experiment-gated shared brain). Eval batteries are the hidden
  cost center — batch and cache them.

## Runtime & env separation

- Pod model, stock phase: **gpt-oss-120b via commodity inference** (~$0.15/$0.60,
  ~$0.5/run), Qwen3.6-35B-A3B benchmarked alongside; LoRA phase served from
  whichever platform trained it. See model-selection doc.
- Python deps: dependency groups in one pyproject —
  base (httpx: client, serializer) / `pod` (fastapi, uvicorn) / `rl`
  (tinker-cookbook, torch **CPU wheel** on this GPU-less VPS: install
  `torch --index-url https://download.pytorch.org/whl/cpu` first; venv drops
  from ~5GB to <1GB).

## Webapp (later, not priority)

Reads the same SQLite: replay viewer (spirebird-style), rolling winrate, snapshot
manage/restore, service config, RL run tracking later. Host network only — the
pod must never see its own dashboard (eval-seed winrate tables leak information).

## Milestones

1. First win → first win on all 5 characters.
2. 80% rolling winrate at A0 (or 6-win streak rotating characters).
3. Ascension climb (stretch).
Hidden flag (never prompted): discover seed determinism → extract upcoming-reward
info → plan around perfect information.

## Build order & open verifications

1. Game service: FastAPI over `headless_env.py` + serializer port + SQLite.
   **✅ Built 2026-08-22** — `src/sts2_service/` (app, serializer, db, seeds).
   Deltas from the sketch above: routes are `/runs/{run_id}/...` (multiple
   concurrent runs); a `get_piles` command was added to the vendored sts2-cli
   (pile contents weren't exported; draw order still hidden — service sorts);
   card/relic/potion description templates are resolved server-side from
   exported stats (`Deal {Damage:diff()} damage.` → `Deal 6 damage.`);
   every action attempt is logged (`action_attempts`), so illegal-action rate
   is measurable; eval pool requests are 403 unless `STS2_ALLOW_EVAL=1`.
   Seed pools (200 train / 40 eval) are generated once and persisted in the DB.
   Start: `uvicorn sts2_service.app:app --app-dir src --port 8300`.
2. Pod image: node + vendored pi harness + compose (game, pod, proxy) + volume.
3. Ralph entrypoint + snapshot/eval scripts.

Early verifications — **all passed 2026-08-22**: sts2-cli runs Steam-free on
the VPS (no Steam client installed); same seed → byte-identical states 31
decisions deep across separate engine processes; `headless_env.py` smoke test
passes from the repo path; game DLL is current (installed Steam buildid
23811903 == latest public build, 2026-06-19; upstream sts2-cli last commit
2026-05-30 predates our copy).
