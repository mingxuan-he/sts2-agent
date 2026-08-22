"""Serialize STS2 game state JSON into a compact text prompt for the LLM.

The model needs to understand the game state to make decisions. We convert
the raw JSON from STS2MCP into a structured but token-efficient text format.
"""

from __future__ import annotations

from typing import Any


def serialize_combat_state(state: dict[str, Any]) -> str:
    """Convert a combat game state into a compact text prompt.

    Includes: player HP/block/energy, hand, draw/discard counts,
    enemies with HP/intent/buffs, player buffs/debuffs, potions.
    """
    parts = []

    # Player info
    player = state.get("player", {})
    parts.append(
        f"[Player] HP: {player.get('hp', '?')}/{player.get('max_hp', '?')} | "
        f"Block: {player.get('block', 0)} | "
        f"Energy: {player.get('energy', '?')}/{player.get('max_energy', '?')}"
    )

    # Player buffs/debuffs
    powers = player.get("powers", [])
    if powers:
        power_strs = [f"{p.get('name', '?')}({p.get('amount', '')})" for p in powers]
        parts.append(f"[Buffs/Debuffs] {', '.join(power_strs)}")

    # Hand
    hand = state.get("hand", [])
    if hand:
        card_strs = []
        for i, card in enumerate(hand):
            cost = card.get("cost", "?")
            name = card.get("name", "?")
            card_type = card.get("type", "")
            desc = card.get("description", "")
            # Include target requirement
            target = card.get("target_type", "")
            target_hint = " [needs target]" if target and target != "none" else ""
            card_strs.append(f"  {i}: {name} (cost:{cost}, {card_type}){target_hint} — {desc}")
        parts.append("[Hand]\n" + "\n".join(card_strs))

    # Draw/discard piles
    draw_count = len(state.get("draw_pile", []))
    discard_count = len(state.get("discard_pile", []))
    exhaust_count = len(state.get("exhaust_pile", []))
    parts.append(f"[Piles] Draw: {draw_count} | Discard: {discard_count} | Exhaust: {exhaust_count}")

    # Enemies
    enemies = state.get("enemies", [])
    if enemies:
        enemy_strs = []
        for e in enemies:
            eid = e.get("entity_id", "?")
            name = e.get("name", "?")
            hp = e.get("hp", "?")
            max_hp = e.get("max_hp", "?")
            block = e.get("block", 0)
            intent = e.get("intent", {})
            intent_str = _format_intent(intent)
            e_powers = e.get("powers", [])
            power_str = ""
            if e_powers:
                power_str = " | " + ", ".join(
                    f"{p.get('name', '?')}({p.get('amount', '')})" for p in e_powers
                )
            enemy_strs.append(
                f"  {eid}: {name} HP:{hp}/{max_hp} Block:{block} Intent:{intent_str}{power_str}"
            )
        parts.append("[Enemies]\n" + "\n".join(enemy_strs))

    # Potions
    potions = state.get("potions", [])
    if potions:
        pot_strs = []
        for i, p in enumerate(potions):
            if p and p.get("name"):
                pot_strs.append(f"  Slot {i}: {p['name']} — {p.get('description', '')}")
        if pot_strs:
            parts.append("[Potions]\n" + "\n".join(pot_strs))

    # Turn number
    turn = state.get("turn", "?")
    parts.append(f"[Turn] {turn}")

    return "\n".join(parts)


def serialize_map_state(state: dict[str, Any]) -> str:
    """Convert a map navigation state into a prompt."""
    parts = []

    player = state.get("player", {})
    parts.append(
        f"[Player] HP: {player.get('hp', '?')}/{player.get('max_hp', '?')} | "
        f"Gold: {player.get('gold', 0)}"
    )

    floor = state.get("floor", "?")
    parts.append(f"[Floor] {floor}")

    # Available nodes
    nodes = state.get("available_nodes", [])
    if nodes:
        node_strs = []
        for i, n in enumerate(nodes):
            node_type = n.get("type", "?")
            node_strs.append(f"  {i}: {node_type}")
        parts.append("[Available Paths]\n" + "\n".join(node_strs))

    return "\n".join(parts)


def serialize_reward_state(state: dict[str, Any]) -> str:
    """Convert a post-combat reward state into a prompt."""
    parts = []
    rewards = state.get("rewards", [])
    if rewards:
        for i, r in enumerate(rewards):
            parts.append(f"  {i}: {r.get('type', '?')} — {r.get('description', str(r))}")
    return "[Rewards]\n" + "\n".join(parts) if parts else "[Rewards] None"


def serialize_card_reward_state(state: dict[str, Any]) -> str:
    """Convert a card reward selection state into a prompt."""
    cards = state.get("cards", [])
    card_strs = []
    for i, c in enumerate(cards):
        card_strs.append(f"  {i}: {c.get('name', '?')} ({c.get('type', '')}, {c.get('rarity', '')}) — {c.get('description', '')}")
    return "[Card Reward — pick one or skip]\n" + "\n".join(card_strs)


def serialize_rest_site_state(state: dict[str, Any]) -> str:
    """Convert a rest site state into a prompt."""
    options = state.get("options", [])
    opt_strs = [f"  {i}: {o.get('name', '?')} — {o.get('description', '')}" for i, o in enumerate(options)]
    return "[Rest Site — choose one]\n" + "\n".join(opt_strs)


def serialize_event_state(state: dict[str, Any]) -> str:
    """Convert an event state into a prompt."""
    parts = []
    parts.append(f"[Event] {state.get('event_name', '?')}")
    body = state.get("body", "")
    if body:
        parts.append(body)
    options = state.get("options", [])
    if options:
        opt_strs = [f"  {i}: {o.get('text', '?')}" for i, o in enumerate(options)]
        parts.append("[Options]\n" + "\n".join(opt_strs))
    return "\n".join(parts)


def serialize_state(state: dict[str, Any]) -> str:
    """Dispatch to the right serializer based on state_type."""
    state_type = state.get("state_type", "")

    if state_type in ("monster", "elite", "boss"):
        return serialize_combat_state(state)
    elif state_type == "map":
        return serialize_map_state(state)
    elif state_type == "combat_rewards":
        return serialize_reward_state(state)
    elif state_type == "card_reward":
        return serialize_card_reward_state(state)
    elif state_type == "rest_site":
        return serialize_rest_site_state(state)
    elif state_type in ("event",):
        return serialize_event_state(state)
    else:
        # Fallback: dump key info
        return f"[State: {state_type}]\n{state}"


def _format_intent(intent: dict[str, Any]) -> str:
    """Format enemy intent into a readable string."""
    if not intent:
        return "Unknown"
    intent_type = intent.get("type", "unknown")
    damage = intent.get("damage")
    hits = intent.get("hits", 1)
    block = intent.get("block")

    parts = [intent_type]
    if damage is not None:
        if hits > 1:
            parts.append(f"{damage}x{hits}")
        else:
            parts.append(str(damage))
    if block is not None:
        parts.append(f"block:{block}")
    return " ".join(parts)
