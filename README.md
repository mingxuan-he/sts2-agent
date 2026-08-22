# sts2-agent

LLM agents that learn to play **Slay the Spire 2**, two ways:

1. **Weights track (`sts2_rl`)** — train a small reasoning LLM with GRPO so it plays well zero-shot.
2. **Agent track (Idea 2, planned)** — freeze the weights and let a persistent agent improve its own *context*: notes, skill files, and the harness itself.

Both tracks run on the same substrate: [sts2-cli](https://github.com/wuhao21/sts2-cli), the real game engine (IL-patched `sts2.dll`) running headless with a JSON stdin/stdout protocol. Deterministic, seedable, all five characters, cheap to run in parallel — see `headless_env.py` for the async wrapper and `HeadlessEnvPool` for parallel rollouts.

## Track 1: RL on Tinker (weights)

Train **Qwen3-30B-A3B** (MoE, 3B active params) via GRPO on [Tinker](https://thinkingmachines.ai/tinker/), starting with per-combat episodes.

Status: pipeline largely built —

- `src/sts2_rl/envs/` — HTTP/subprocess clients, state serializer, reward shaping, Tinker `MessageEnv`
- `src/sts2_rl/envs/scenarios.py` — seedable Act 1 combat scenarios in difficulty tiers
- `src/sts2_rl/training/dataset.py` — GRPO group builder + dataset (verified end-to-end with the real Qwen3 tokenizer; `tests/` passing)

Remaining: swap the approximate Python sim (`sts2_sim.py`, now obsolete) for real `HeadlessEnv` stepping, write the train config, add Tinker billing, run the first loop.

### Reward shaping (per-combat, dense)

| Signal | Value |
|--------|-------|
| Invalid action format | -0.5 |
| Illegal action | -0.3 |
| Damage dealt | +0.01/HP |
| HP lost | -0.02/HP |
| Enemy killed | +0.2 |
| Combat won / lost | +1.0 / -1.0 |

Per-run shaping (floor progress +0.05, act boss +2.0, final win +5.0) reserved for full-run episodes later.

## Track 2: Self-improving harness (context)

Pure inference — no gradients. A persistent agent lives in a pod with the game exposed **only** as a JSON endpoint (engine process outside the agent's sandbox, so state-editing cheats are impossible by construction). It plays runs in a continuous loop and may write notes, build skill files, and edit its own harness. Weights never change; all learning is in the accumulated context.

- **Eval:** snapshot the pod, play held-out seeds never seen in training. Winrate vs. accumulated-context-size is the headline curve.
- **Milestones:** first win → first win on every character → 80% rolling winrate (or a 6-win streak rotating characters) at A0, then climb ascensions.
- **Swarm variant:** the main agent spawns parallel subagent runs and synthesizes their reports — but any edit to the shared skill library must win an A/B test on a fixed seed battery before merging (no superstition in the shared brain).
- **Hidden flag:** seeds are visible and deterministic. An agent that independently discovers seed determinism, extracts upcoming-reward information, and *plans around perfect information* has done something genuinely hard. Not prompted, not required.
- **Contamination policy:** the pod filesystem must not contain decompiled game source, spire-codex dumps, or expert replays unless a phase deliberately grants them, so that discoveries are attributable.

Why the same repo: both tracks share the engine wrapper, state serializer, seed batteries, and eval harness; and the agent track's winning trajectories become SFT cold-start data for the weights track.

Full architecture and API design: [docs/track2-pod-design.md](docs/track2-pod-design.md).
Model choice + cost analysis: [docs/model-selection-2026-08.md](docs/model-selection-2026-08.md).

## Repo layout

```
headless_env.py         — async subprocess wrapper around sts2-cli (+ HeadlessEnvPool)
sts2-cli/               — headless game engine (vendored build; lib/ DLLs not committed)
src/sts2_service/       — Track 2 game service: FastAPI + serializer + SQLite logging
src/sts2_rl/            — Track 1: envs, rewards, scenarios, GRPO dataset
tests/                  — serializer + sim + dataset tests
scripts/                — gameplay recording utilities
benchmarks/             — model latency notes
```

Run the game service (venv: `uv venv && uv pip install -e ".[service,dev]"`):

```
uvicorn sts2_service.app:app --app-dir src --host 127.0.0.1 --port 8300
```

## References

- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [sts2-cli](https://github.com/wuhao21/sts2-cli) — headless STS2 engine
- [STS2MCP](https://github.com/Gennadiyev/STS2MCP) — mod for driving the live game (co-op play)
- [Spire Codex](https://github.com/ptrlrd/spire-codex) — card/relic/monster data
- [spirebird replays](https://spirebird.com/replays.html) — expert `.spgn` action tapes (small corpus for now; revisit as it grows)
