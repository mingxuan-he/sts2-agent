# TPS Benchmark: Fable 5 vs Opus 4.8

**Date:** 2026-06-09
**Method:** `claude` CLI v2.1.167 via Max plan (cliproxyapi)
**Prompt:** B-tree technical explanation (~2000 words requested)
**Runs:** 2 per model

## Results

| Metric | Fable 5 | Opus 4.8 |
|--------|---------|----------|
| Avg wall time | 102.7s | 84.3s |
| Avg output tokens (est) | ~5,306 | ~4,683 |
| Effective TPS (tok/wall) | 51.7 | 55.6 |
| Wall time ratio vs Opus | 1.22x slower | baseline |

### Raw runs

| Model | Run | Wall (s) | Est tokens | Eff TPS |
|-------|-----|----------|------------|---------|
| Fable 5 | 1 | 101.6 | 5,331 | 52.5 |
| Fable 5 | 2 | 103.8 | 5,281 | 50.9 |
| Opus 4.8 | 1 | 83.7 | 4,649 | 55.6 |
| Opus 4.8 | 2 | 84.9 | 4,716 | 55.6 |

## Notes

- CLI buffers output (no true streaming), so TTFB vs decode TPS can't be separated
- Fable 5 produced ~13% more output for the same prompt (more thorough)
- Wall time dominated by thinking/reasoning, not decode
- Both models working through cliproxyapi on Max plan
- **Fable 5 burns 2x token quota** on Max plan vs Opus

## STS2 Feasibility Concern

~100s wall time per action is **too slow for live STS2 play**:
- Typical combat: 10-15 card plays + end turns ≈ 15-20 actions
- At 100s/action → **25-33 min per combat floor**
- Full run (~50 floors, not all combat) → easily **10+ hours**
- This is for a long-form prose prompt though; STS2 game actions should be shorter

### Mitigations to test
1. **Shorter prompts = faster response.** STS2 game state + "pick a card" is much shorter than "write 2000 words about B-trees". Expect 10-30s per action, not 100s.
2. **Use Sonnet 4.5 or Haiku** for baseline speed comparison — might be 5-10x faster.
3. **API direct** (not CLI) avoids CLI overhead and gives real streaming + token counts.
4. **Batch decisions** — ask model to plan full turn (all card plays) in one call instead of one card at a time.
5. **The RL target model (Qwen3-30B-A3B) is local/Tinker** — inference will be much faster than frontier API models. Fable 5 is just for generating training data / baselines, not the runtime agent.
