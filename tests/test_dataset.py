"""Tests for the GRPO dataset / env-group wiring (no network / no Tinker API)."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sts2_rl.envs import scenarios
from sts2_rl.training.dataset import STS2Dataset, STS2GroupBuilder


def test_scenario_seed_determinism():
    s = scenarios.SCENARIOS_BY_NAME["jaw_worm"]
    assert s.make(42) == s.make(42)
    assert s.make(42) != s.make(7)


def test_curriculum_tiers():
    assert {sc.name for sc in scenarios.curriculum(0)} == {"louse", "jaw_worm"}
    assert len(scenarios.curriculum(2)) == len(scenarios.SCENARIOS)


def _fake_renderer():
    """A minimal renderer stub so we can test group/dataset structure without
    loading a real tokenizer."""

    class _R:
        def get_stop_sequences(self):
            return [0]

    return _R()


def test_dataset_batches_and_group_size():
    ds = STS2Dataset(batch_size=2, group_size=4, renderer=_fake_renderer(), max_tier=1)
    assert len(ds) == 2  # 3 tier<=1 scenarios, batch_size 2 -> 2 batches
    batch = ds.get_batch(0)
    assert len(batch) == 2
    assert all(isinstance(b, STS2GroupBuilder) for b in batch)
    assert all(b.num_envs == 4 for b in batch)
    assert batch[0].scenario_name == "louse"


def test_group_builder_make_envs_same_initial_state():
    """All envs in a GRPO group must start from the identical state."""
    gb = STS2GroupBuilder(
        scenario_name="jaw_worm", seed=1, num_envs=4, renderer=_fake_renderer()
    )
    envs = asyncio.run(gb.make_envs())
    assert len(envs) == 4
    states = [json.dumps(e.message_env.initial_state, sort_keys=True) for e in envs]
    assert len(set(states)) == 1  # identical starts, variance comes from sampling


def test_logging_tags():
    gb = STS2GroupBuilder(
        scenario_name="gremlin_nob", seed=0, num_envs=2, renderer=_fake_renderer()
    )
    assert gb.logging_tags() == ["sts2", "sts2_tier2", "sts2_gremlin_nob"]


if __name__ == "__main__":
    import inspect

    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"All {len(fns)} passed")
