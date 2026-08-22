# Model selection & cost estimates (2026-08-22)

Prices verified 2026-08-22 from tinker-docs.thinkingmachines.ai/tinker/models/ and the
Anthropic price sheet. All $/1M tokens. Tinker gives an 80% discount on cached prefill;
Anthropic cache reads are ~0.1× input price. Tinker checkpoint storage $0.10/GB-mo.

## Candidate prices

| Model | Where | Ctx | Prefill | Cached | Sample | Train |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (MoE) | Tinker | 64K | 0.54 | 0.108 | 1.335 | 1.177 |
| GPT-OSS-120B (MoE, ~5B act) | Tinker | 32K | 0.33 | 0.066 | 0.84 | 0.737 |
| Nemotron 3.5-Lightning-30B-A3B | Tinker | 64K | 0.39 | 0.039 | 0.99 | 0.88 |
| Qwen3.5-9B (dense) | Tinker | 64K | 0.66 | 0.132 | 1.995 | 1.463 |
| Claude Haiku 4.5 | Anthropic | 200K | 1.00 | ~0.10 | 5.00 | — |
| Claude Sonnet 5 (intro→2026-08-31) | Anthropic | 1M | 2.00→3.00 | 0.2→0.3 | 10→15 | — |
| Claude Opus 5 | Anthropic | 1M | 5.00 | 0.50 | 25.00 | — |

Notes: **Qwen3-30B-A3B (the original Track-1 target) is gone from Tinker** —
Qwen3.6-35B-A3B is the successor. Qwen3.6-27B retires Sept 2; Tinker rotates models,
so re-verify before long campaigns. Anthropic Batch API is 50% off but too slow for
an interactive game loop (fine for offline eval scoring).

## Workload model (Idea 2, full run)

From jorbs spirebird tapes: ~1,200 scenes/run → ~800 agent decisions/run (569 card
plays + end turns + map/reward/event/shop). Old serializer measured a combat obs at
~377 tokens; with skill files and preamble assume **~3K input/decision at ~70%
cache-hit, ~600 output/decision** (brief reasoning). Per run: ~2.4M in / ~0.5M out.

| Model | $/run (cached) | $/run (no cache) | 1,000-run campaign |
|---|---|---|---|
| GPT-OSS-120B | ~$0.75 | ~$1.2 | ~$750 |
| Nemotron Lightning 30B | ~$0.85 | ~$1.4 | ~$850 |
| **Qwen3.6-35B-A3B** | **~$1.2** | ~$1.9 | **~$1,200** |
| Haiku 4.5 | ~$3.3 | ~$4.8 | ~$3,300 |
| Sonnet 5 (intro) | ~$6.6 | ~$9.6 | ~$6,600 |
| Opus 5 | ~$16 | ~$24 | ~$16,000 |

Sensitivity: long per-decision CoT (2-3K output) multiplies the output half by 3-5×
(Qwen worst case ~$3-4/run). Cache discipline (stable system/skill prefix, state
last) is worth ~40% — design the serializer for it from day one.

## Decision: Idea 2 runtime = **Qwen3.6-35B-A3B via Tinker sampling**

Rationale, in order:

1. **Track-1 synergy is the whole ballgame.** Same model family as the GRPO target →
   the pod's trajectories are near-on-policy SFT/RL data, the serializer/tokenizer
   work is shared, and Tinker sampling can serve **our own LoRA checkpoints** — so a
   Track-1 fine-tune can be hot-swapped into the running pod. That closes the
   Idea-2 → Idea-1 → back-into-the-pod continual-learning loop, which no Anthropic
   model can do (no weight access).
2. Cost is comfortably in personal-project range (~$1-2/full run).
3. 64K context fits decision-scoped prompting with a fat skill file; per-run state
   never needs to live in one context.

**Rival to benchmark, not skip:** GPT-OSS-120B — likely smarter per dollar (~40%
cheaper, bigger total params), also trainable on Tinker ($0.737/M). Run the same
20-seed battery on both before committing the pod; if it wins decisively on winrate,
it becomes the target for both tracks. Downside: 32K ctx (128K tier is 2.4× the
price) and Harmony formatting quirks.

**Architect/synthesizer tier:** the swarm's main agent (skill synthesis, harness
edits, A/B verdicts) runs rarely — use Claude via the Max plan interactively, or
Haiku 4.5/Sonnet 5 API for automated calls. At ~1 call per merged skill this is
noise (<$50 over the project). Don't spend frontier tokens on card plays.

**Ruled out:** Qwen3.5-9B/4B (too weak for harness self-editing to be meaningful),
Opus/Sonnet as the runtime (5-13× Qwen for gameplay decisions; Sonnet's intro price
also expires 2026-08-31), Inkling ($9.36 sample — priced like a frontier model).

## Idea 1 (GRPO) revised estimate at real prices

Per-combat episodes: ~20 decisions × (2K prefill + 300 sample) ≈ 40K prefill + 6K
sample per rollout. Group size 8, ~2,000 steps → 16K rollouts:

- Prefill: ~640M × $0.54 (less with caching) ≈ $200-350
- Sample: ~96M × $1.335 ≈ $130
- Train (fwd+bwd over full sequences): ~740M × $1.177 ≈ $870

**≈ $1.0-1.4K for a serious combat-only GRPO campaign; a 200-step pilot ≈ $100-150.**
Full-run GRPO (800-decision episodes) is ~40× that — stays off the table; combat-only
plus SFT-from-pod-trajectories remains the plan.

## Addendum (2026-08-22, later): Tinker inference verified + Prime Intellect

**Tinker inference: real, but overpriced for stock checkpoints.** Tinker has a
dedicated `SamplingClient`, supports sampling-only usage, exposes an OpenAI-compatible
endpoint, and can export checkpoints to HuggingFace (no lock-in). But its sampling
prices are frontier-adjacent for what are commodity open models:

| Serving Qwen3.6-35B-A3B / similar | Input $/M | Output $/M |
|---|---|---|
| Tinker sampling | 0.54 | 1.335 |
| Prime Intellect hosted-training rollout rates (same model) | 0.25 | 0.75 |
| Open market, gpt-oss-120b (OpenRouter, typical) | ~0.15 | ~0.60 |
| Open market, gpt-oss-120b (cheapest, CoreWeave) | 0.03 | 0.17 |

Conclusion: **Tinker's sampling premium is only worth paying when it's serving OUR
LoRA.** For the stock-checkpoint phase of the Idea 2 pod, use commodity inference
(OpenRouter or a direct provider) — pod cost drops to roughly **$0.30–0.80/run**.

**Prime Intellect (docs.primeintellect.ai):** GPU marketplace + sandboxes + an
OpenAI-compatible inference router (per-token pricing "coming soon"; catalog includes
third-party models) + **hosted RL training** built on their open-source `verifiers`
(environment/eval library) and `prime-rl` stack, with a community Environments Hub.
Hosted training prices (verify live via `prime train models --output json`):

| Model | Input | Output | Train |
|---|---|---|---|
| Qwen3.6-35B-A3B | 0.25 | 0.75 | 1.00 |
| Qwen3.5-35B-A3B | 0.25 | 0.75 | 1.00 |
| Nemotron-3.5-Lightning-30B-A3B | 0.15 | 0.45 | 0.60 |
| gpt-oss-20b | 0.10 | 0.30 | 0.40 |
| Qwen3.5-9B | 0.20 | 0.60 | 0.60 |

Same target model as Tinker at roughly half the token rates; GRPO campaign estimate
re-done at PI prices ≈ **$950–1,000** (train tokens dominate, so the end-to-end saving
vs Tinker is ~25–30%, not half). gpt-oss-120b is NOT in PI's hosted-training list
(Tinker has it). Open due diligence on PI: LoRA vs full FT, checkpoint
download/serving story, hosted-RL maturity (Tinker's API + cookbook are more proven,
and our existing Track-1 code targets Tinker's MessageEnv).

**Hedge that costs almost nothing:** write the STS2 combat environment as a
`verifiers` environment. verifiers is platform-agnostic (evals run against any
OpenAI-compatible endpoint), PI hosted training consumes it natively, porting to it
from our MessageEnv is mostly moving the serializer/reward calls, and the
Environments Hub is a natural place to later publish the STS2 env + seed benchmark.

**Revised runtime decision for Idea 2:**
- Stock phase: **gpt-oss-120b via commodity inference** (~$0.15/$0.60 typical) as the
  default player, with Qwen3.6-35B-A3B benchmarked alongside wherever it's served
  cheaply. Same 20-seed battery decides.
- LoRA phase (weights trained): serve via Tinker sampling or PI depending on which
  platform trained it; HF export keeps self-hosting open.
- Training platform: decide after a ~$100 pilot on each — PI wins on price +
  ecosystem, Tinker on maturity + gpt-oss-120b availability + existing code.

## Immediate implications

- Update Track-1 code/docs: Qwen3-30B-A3B → Qwen3.6-35B-A3B (check `check_models.py`
  against the live model list; renderer name may have changed).
- Budget envelope for Idea 2 through the 80%-winrate chase: **~$1.5-4K** total
  (milestone 1 "first win, all characters" ≈ 100-200 runs ≈ $150-400). The eval
  batteries for swarm skill-gating are the hidden cost center — 20-seed batteries
  per candidate merge add up faster than the main loop; batch and cache them.
