"""Reward shaping for STS2 RL training.

Rewards are computed from game state transitions. We use a combination of
sparse terminal rewards and dense intermediate signals.
"""

from __future__ import annotations

from typing import Any


def compute_combat_reward(
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    action_valid: bool,
) -> float:
    """Compute shaped reward for a single combat step.

    Components:
    - Invalid action penalty: -0.5 for illegal moves
    - Damage dealt: +0.01 per HP of damage dealt to enemies
    - HP preserved: -0.02 per HP lost by player
    - Enemy killed: +0.2 per enemy killed
    - Combat won: +1.0
    - Combat lost (player died): -1.0
    """
    if not action_valid:
        return -0.5

    reward = 0.0

    # Player HP change
    player_before = state_before.get("player", {})
    player_after = state_after.get("player", {})
    hp_before = player_before.get("hp", 0)
    hp_after = player_after.get("hp", 0)
    hp_delta = hp_after - hp_before
    if hp_delta < 0:
        reward += hp_delta * 0.02  # penalty for HP loss

    # Enemy damage dealt
    enemies_before = {e["entity_id"]: e for e in state_before.get("enemies", [])}
    enemies_after = {e["entity_id"]: e for e in state_after.get("enemies", [])}

    for eid, eb in enemies_before.items():
        ea = enemies_after.get(eid)
        if ea is None:
            # Enemy killed (no longer present)
            reward += 0.2
        else:
            dmg = eb.get("hp", 0) - ea.get("hp", 0)
            if dmg > 0:
                reward += dmg * 0.01

    # Combat terminal conditions
    state_type_after = state_after.get("state_type", "")
    if state_type_after == "combat_rewards":
        # Won the combat
        reward += 1.0
    elif hp_after <= 0:
        # Player died
        reward -= 1.0

    return reward


def compute_run_reward(final_state: dict[str, Any]) -> float:
    """Compute terminal reward for a full run.

    Components:
    - Floor reached: +0.05 per floor
    - Boss killed: +2.0 per act boss
    - Run won (heart kill): +5.0
    - Run lost: -0.5
    """
    floor = final_state.get("floor", 0)
    reward = floor * 0.05

    # TODO: track boss kills and act progression once we have
    # a full run trajectory tracker

    # Check if run ended in victory
    state_type = final_state.get("state_type", "")
    if state_type == "menu":
        # Run ended — check if it was a win
        # (need to parse from final state or track externally)
        pass

    return reward


def compute_card_reward_reward(
    chosen_card: dict[str, Any] | None,
    skipped: bool,
    deck_size: int,
) -> float:
    """Reward for card reward decisions.

    Very light shaping — we don't want to hardcode card tier lists.
    - Slight penalty for bloating deck (>20 cards)
    - Neutral for reasonable deck sizes
    """
    if skipped:
        return 0.0  # Skipping is always valid

    if deck_size > 25:
        return -0.05  # Mild penalty for deck bloat
    return 0.0


def compute_rest_site_reward(
    option_chosen: str,
    player_hp: int,
    player_max_hp: int,
) -> float:
    """Reward for rest site decisions.

    Light shaping:
    - Resting at high HP is slightly penalized (waste)
    - Smithing/upgrading at low HP is slightly penalized (risky)
    """
    hp_ratio = player_hp / max(player_max_hp, 1)

    if "rest" in option_chosen.lower() or "heal" in option_chosen.lower():
        if hp_ratio > 0.8:
            return -0.05  # Resting when nearly full
        return 0.0
    else:
        # Upgrade/smith/other
        if hp_ratio < 0.3:
            return -0.05  # Risky to not heal
        return 0.05  # Mild reward for upgrading when healthy
