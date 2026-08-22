"""STS2 Combat Environment for Tinker RL training.

This implements the MessageEnv interface from tinker_cookbook.rl.message_env.
Each environment instance represents a single combat encounter.
The model sees the game state as a system+user message, outputs
a JSON action as its assistant message, and we step the game.

Design: multi-turn per-combat. Each turn the model sees the current
combat state and outputs one action (play_card, end_turn, use_potion).
The episode ends when combat ends (win/loss).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from tinker_cookbook.renderers import Message
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult

from .reward import compute_combat_reward
from .state_serializer import serialize_combat_state
from .sts2_client import STS2Client
from . import sts2_sim

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert Slay the Spire 2 player. You are in combat.
Analyze the game state and choose the best action.

Output a single JSON action. Examples:
- {"action": "play_card", "card_index": 0, "target": "jaw_worm_0"}
- {"action": "play_card", "card_index": 2}  (no target needed for self-targeting cards)
- {"action": "end_turn"}
- {"action": "use_potion", "slot": 0, "target": "jaw_worm_0"}

Think step by step about:
1. What enemies are doing (their intents)
2. How much damage you can deal vs block you need
3. Card synergies and energy efficiency
4. When to use potions (boss fights, emergencies)

Output ONLY the JSON action, nothing else."""


@dataclass
class STS2CombatEnv(MessageEnv):
    """A single STS2 combat encounter as a Tinker MessageEnv.

    For training, we need a way to create reproducible combat scenarios.
    Options:
    1. Live game via STS2MCP (slow, non-reproducible, requires game running)
    2. Recorded state snapshots + simulator (fast, reproducible, no game needed)

    This class supports both modes via the `initial_state` parameter.
    If provided, we use the simulator (mode 2). Otherwise, we connect to a
    live game (mode 1, not yet wired for async).
    """

    client: STS2Client | None = None
    initial_state: dict[str, Any] | None = None
    max_turns: int = 50
    _current_state: dict[str, Any] = field(default_factory=dict, init=False)
    _prev_state: dict[str, Any] = field(default_factory=dict, init=False)
    _turn: int = field(default=0, init=False)
    _cumulative_reward: float = field(default=0.0, init=False)

    async def initial_observation(self) -> list[Message]:
        """Return the initial combat state as messages."""
        if self.initial_state is not None:
            self._current_state = self.initial_state
        elif self.client is not None:
            # Live mode: fetch state from game
            self._current_state = await self.client.get_state()
        else:
            raise ValueError("Must provide either client or initial_state")

        state_text = serialize_combat_state(self._current_state)
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"Current combat state:\n\n{state_text}"),
        ]

    async def step(self, message: Message) -> MessageStepResult:
        """Process the model's action and return the result."""
        self._turn += 1
        self._prev_state = self._current_state

        # Parse the model's output as a JSON action
        content = _extract_text(message)
        action, parse_error = _parse_action(content)

        if parse_error:
            reward = -0.5
            self._cumulative_reward += reward
            return MessageStepResult(
                reward=reward,
                episode_done=self._turn >= self.max_turns,
                next_messages=[
                    Message(
                        role="user",
                        content=f"Invalid action format: {parse_error}\nTry again.\n\n"
                        + serialize_combat_state(self._current_state),
                    )
                ],
                metrics={"turn": float(self._turn)},
                logs={"error": parse_error},
            )

        # Validate the action against current state
        valid, validation_error = _validate_action(action, self._current_state)
        if not valid:
            reward = -0.3
            self._cumulative_reward += reward
            return MessageStepResult(
                reward=reward,
                episode_done=self._turn >= self.max_turns,
                next_messages=[
                    Message(
                        role="user",
                        content=f"Invalid action: {validation_error}\nTry again.\n\n"
                        + serialize_combat_state(self._current_state),
                    )
                ],
                metrics={"turn": float(self._turn)},
                logs={"error": validation_error},
            )

        # Step the environment
        if self.client is not None:
            # Live mode
            self._current_state = await self.client.send_action(action)
        else:
            # Simulator mode
            self._current_state = sts2_sim.step(self._current_state, action)

        done_terminal, victory = sts2_sim.is_terminal(self._current_state)
        done = done_terminal or self._turn >= self.max_turns

        reward = compute_combat_reward(self._prev_state, self._current_state, action_valid=True)
        self._cumulative_reward += reward

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=[
                Message(
                    role="user",
                    content=serialize_combat_state(self._current_state),
                )
            ]
            if not done
            else [],
            metrics={
                "turn": float(self._turn),
                "cumulative_reward": self._cumulative_reward,
                "victory": 1.0 if (done and victory) else 0.0,
            },
            logs={
                "action": json.dumps(action),
                "victory": str(victory) if done else "ongoing",
            },
        )


# ---------------------------------------------------------------------------
# Parsing helpers (module-level so they're testable)
# ---------------------------------------------------------------------------

def _extract_text(message: Message) -> str:
    """Extract text content from a Message (TypedDict), stripping thinking blocks."""
    content = message["content"]
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if hasattr(part, "text"):
                parts.append(part.text)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        text = " ".join(parts)
    else:
        text = str(content)

    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Strip thinking tags
    if "<think>" in text:
        parts = text.split("</think>")
        text = parts[-1].strip() if len(parts) > 1 else text

    return text


def _parse_action(content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the model's output as a JSON action."""
    try:
        action = json.loads(content)
        if not isinstance(action, dict):
            return None, "Output must be a JSON object"
        if "action" not in action:
            return None, "JSON must contain 'action' key"
        return action, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def _validate_action(action: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate an action against the current game state."""
    action_type = action.get("action")

    if action_type == "play_card":
        card_index = action.get("card_index")
        if card_index is None:
            return False, "play_card requires card_index"
        hand = state.get("hand", [])
        if not isinstance(card_index, int) or card_index < 0 or card_index >= len(hand):
            return False, f"card_index {card_index} out of range (hand has {len(hand)} cards)"

        card = hand[card_index]
        cost = card.get("cost", 0)
        energy = state.get("player", {}).get("energy", 0)
        if isinstance(cost, int) and cost > energy:
            return False, f"Not enough energy ({energy}) to play {card.get('name')} (cost {cost})"

        target_type = card.get("target_type", "")
        if target_type and target_type not in ("none", "self", "all"):
            if "target" not in action:
                return False, f"Card {card.get('name')} requires a target"
            target = action["target"]
            enemy_ids = [e.get("entity_id") for e in state.get("enemies", [])]
            if target not in enemy_ids:
                return False, f"Target '{target}' not found. Valid: {enemy_ids}"

        return True, None

    elif action_type == "end_turn":
        return True, None

    elif action_type == "use_potion":
        slot = action.get("slot")
        if slot is None:
            return False, "use_potion requires slot"
        potions = state.get("potions", [])
        if not isinstance(slot, int) or slot < 0 or slot >= len(potions):
            return False, f"Potion slot {slot} out of range"
        potion = potions[slot] if slot < len(potions) else None
        if not potion or not potion.get("name"):
            return False, f"No potion in slot {slot}"
        return True, None

    else:
        return False, f"Unknown action type: {action_type}"
