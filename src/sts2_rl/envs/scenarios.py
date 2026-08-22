"""Combat scenario factory for RL training.

A scenario is a named, seedable factory that produces an initial combat state
for the simulator. Scenarios form a curriculum (easy Act 1 normals → elites).

Each scenario factory takes a ``seed`` and returns a fresh state dict. The seed
controls the opening-hand shuffle so a given (scenario, seed) pair is fully
reproducible — required so every env in a GRPO group starts from the same state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import sts2_sim

StateFactory = Callable[[int], dict[str, Any]]


@dataclass(frozen=True)
class Scenario:
    """A named combat scenario with a difficulty tier and a state factory."""

    name: str
    tier: int  # 0 = easy normal, 1 = harder normal, 2 = elite
    factory: StateFactory

    def make(self, seed: int) -> dict[str, Any]:
        return self.factory(seed)


# ---------------------------------------------------------------------------
# Scenario definitions (Ironclad, Act 1)
# ---------------------------------------------------------------------------

def _jaw_worm(seed: int) -> dict[str, Any]:
    return sts2_sim.make_ironclad_starter_state(
        enemy_hp=40, enemy_name="Jaw Worm", enemy_id="jaw_worm_0",
        enemy_intent_damage=11, player_hp=80, seed=seed,
    )


def _cultist(seed: int) -> dict[str, Any]:
    # Cultist: low HP but ramps Strength via Ritual — punishes slow kills.
    state = sts2_sim.make_ironclad_starter_state(
        enemy_hp=48, enemy_name="Cultist", enemy_id="cultist_0",
        enemy_intent_damage=6, player_hp=80, seed=seed,
    )
    state["enemies"][0]["powers"] = [{"name": "Ritual", "amount": 3}]
    return state


def _louse(seed: int) -> dict[str, Any]:
    return sts2_sim.make_ironclad_starter_state(
        enemy_hp=12, enemy_name="Red Louse", enemy_id="louse_0",
        enemy_intent_damage=6, player_hp=80, seed=seed,
    )


def _gremlin_nob(seed: int) -> dict[str, Any]:
    # Elite: high HP, hard-hitting. Punishes skill spam (Enrage in real game;
    # here just modeled as a high-damage intent + bulk).
    state = sts2_sim.make_ironclad_starter_state(
        enemy_hp=82, enemy_name="Gremlin Nob", enemy_id="gremlin_nob_0",
        enemy_intent_damage=14, player_hp=80, seed=seed,
    )
    return state


def _lagavulin(seed: int) -> dict[str, Any]:
    # Elite: starts asleep with heavy block; big payoff once awake.
    state = sts2_sim.make_ironclad_starter_state(
        enemy_hp=109, enemy_name="Lagavulin", enemy_id="lagavulin_0",
        enemy_intent_damage=18, player_hp=80, seed=seed,
    )
    state["enemies"][0]["block"] = 8
    return state


SCENARIOS: list[Scenario] = [
    Scenario("louse", tier=0, factory=_louse),
    Scenario("jaw_worm", tier=0, factory=_jaw_worm),
    Scenario("cultist", tier=1, factory=_cultist),
    Scenario("gremlin_nob", tier=2, factory=_gremlin_nob),
    Scenario("lagavulin", tier=2, factory=_lagavulin),
]

SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}


def curriculum(max_tier: int = 2) -> list[Scenario]:
    """Return scenarios up to and including ``max_tier`` (curriculum subset)."""
    return [s for s in SCENARIOS if s.tier <= max_tier]
