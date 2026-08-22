"""GRPO dataset + env-group builder wiring STS2CombatEnv into Tinker RL.

The training loop consumes an ``RLDataset`` that yields ``EnvGroupBuilder``
batches. For GRPO each *group* is N rollouts of the **same** combat scenario
(same initial state) so rewards can be centered within the group; variance
comes from model sampling, not from the environment.

Layering:
    Scenario (scenarios.py)            -- seedable initial-state factory
      → STS2CombatEnv (MessageEnv)     -- multi-turn combat, JSON actions
        → EnvFromMessageEnv (Env)      -- token-level adapter Tinker runs
          → STS2GroupBuilder           -- N envs from one scenario (GRPO group)
            → STS2Dataset              -- batches of group builders
              → STS2DatasetBuilder     -- chz-configurable entry point
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from sts2_rl.envs import scenarios
from sts2_rl.envs.sts2_combat_env import STS2CombatEnv


@dataclass(frozen=True)
class STS2GroupBuilder(EnvGroupBuilder):
    """Builds a GRPO group: ``num_envs`` rollouts of one combat scenario.

    Picklable by design — stores only config strings/ints and the (pickle-safe)
    renderer, and constructs the heavy env objects lazily in ``make_envs``.
    """

    scenario_name: str
    seed: int
    num_envs: int
    renderer: renderers.Renderer
    max_turns: int = 50
    max_trajectory_tokens: int | None = None
    max_generation_tokens: int | None = None

    async def make_envs(self) -> Sequence[Env]:
        scenario = scenarios.SCENARIOS_BY_NAME[self.scenario_name]
        envs: list[Env] = []
        for _ in range(self.num_envs):
            # One fresh state per env, all identical for a given (scenario, seed).
            initial_state = scenario.make(self.seed)
            message_env = STS2CombatEnv(initial_state=initial_state, max_turns=self.max_turns)
            envs.append(
                EnvFromMessageEnv(
                    renderer=self.renderer,
                    message_env=message_env,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                )
            )
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        # All reward is per-step (combat reward shaping); no group-level reward.
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        tier = scenarios.SCENARIOS_BY_NAME[self.scenario_name].tier
        return ["sts2", f"sts2_tier{tier}", f"sts2_{self.scenario_name}"]


class STS2Dataset(RLDataset):
    """Batches of :class:`STS2GroupBuilder`, one builder per combat scenario.

    Each batch index maps to a fixed set of (scenario, seed) pairs so training
    is reproducible. With ``group_size`` rollouts per scenario, a batch of
    ``batch_size`` scenarios produces ``batch_size * group_size`` trajectories.
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        max_tier: int = 2,
        max_turns: int = 50,
        max_trajectory_tokens: int | None = None,
        max_generation_tokens: int | None = None,
        seed: int = 0,
    ):
        self.scenarios = scenarios.curriculum(max_tier)
        if not self.scenarios:
            raise ValueError(f"No scenarios at or below tier {max_tier}")
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.seed = seed

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        # Cycle through scenarios; the batch index seeds the scenario state so
        # each batch trains on fresh-but-reproducible starts.
        builders: list[EnvGroupBuilder] = []
        for i in range(self.batch_size):
            scenario = self.scenarios[(index * self.batch_size + i) % len(self.scenarios)]
            builders.append(
                STS2GroupBuilder(
                    scenario_name=scenario.name,
                    seed=self.seed + index * self.batch_size + i,
                    num_envs=self.group_size,
                    renderer=self.renderer,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                )
            )
        return builders

    def __len__(self) -> int:
        # One "epoch" = each scenario seen once as a group.
        return math.ceil(len(self.scenarios) / self.batch_size)


@chz.chz
class STS2DatasetBuilder(RLDatasetBuilder):
    """chz-configurable entry point for the STS2 GRPO dataset."""

    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    max_tier: int = 2
    max_turns: int = 50
    max_trajectory_tokens: int | None = None
    max_generation_tokens: int | None = None
    seed: int = 0

    async def __call__(self) -> tuple[STS2Dataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        dataset = STS2Dataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            max_tier=self.max_tier,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
            seed=self.seed,
        )
        return dataset, None
