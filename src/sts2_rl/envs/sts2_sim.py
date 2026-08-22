"""Lightweight STS2 combat simulator.

Simulates the STS2 combat loop deterministically from a game state dict.
Used for offline RL training — no real game instance required.

State format matches STS2MCP JSON: player, hand, draw_pile, discard_pile,
exhaust_pile, enemies, potions, turn, state_type.

Cards are data-driven via CARD_DB. We model: Strike, Defend, Bash, Body Slam,
Clash, Cleave, Clothesline, Flex, Havoc, Headbutt, Heavy Blade, Iron Wave,
Perfected Strike, Shrug It Off, Sword Boomerang, Thunderclap, Wild Strike,
Wound, Dazed, Slimed, plus generic "attack/block N" cards via description parse.

Limitations (intentional — training simulator, not full game):
- No RNG seeding / random card draw is implemented (draw from top of draw pile)
- Relics not modeled
- Only core status effects: Strength, Dexterity, Weak, Vulnerable, Frail, Ritual,
  Metallicize, Evolve, Flame Barrier, Combust, Dark Embrace, Rupture, Barricade,
  Juggernaut, Brutality, Fire Breathing, Berserk, Inflame
- Enemy AI uses their stated intent directly (taken from game state)
- No multi-hit tracking through block mid-chain
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

# ---------------------------------------------------------------------------
# Card database
# ---------------------------------------------------------------------------
# Each entry: name -> dict with effect function key and params
# effect: one of "attack", "block", "attack+block", "power", "special"
# For attacks: damage (before Strength/Weak/Vuln), hits, aoe
# For blocks: amount (before Dexterity/Frail)
# For powers: applied to player

CARD_DB: dict[str, dict[str, Any]] = {
    # Ironclad starters
    "Strike": {"type": "attack", "damage": 6, "hits": 1, "aoe": False},
    "Defend": {"type": "block", "amount": 5},
    "Bash": {"type": "attack+debuff", "damage": 8, "hits": 1, "aoe": False,
              "debuffs": [("Vulnerable", 2)]},
    # Common Ironclad
    "Body Slam": {"type": "attack_from_block", "hits": 1, "aoe": False},
    "Clash": {"type": "attack", "damage": 14, "hits": 1, "aoe": False},
    "Cleave": {"type": "attack", "damage": 8, "hits": 1, "aoe": True},
    "Clothesline": {"type": "attack+debuff", "damage": 12, "hits": 1, "aoe": False,
                    "debuffs": [("Weak", 2)]},
    "Flex": {"type": "power_temp", "buff": "Strength", "amount": 2},
    "Havoc": {"type": "special_havoc"},
    "Headbutt": {"type": "attack+retrieve", "damage": 9, "hits": 1, "aoe": False},
    "Heavy Blade": {"type": "attack_strength_mult", "damage": 14, "strength_mult": 3,
                    "hits": 1, "aoe": False},
    "Iron Wave": {"type": "attack+block", "damage": 5, "amount": 5, "hits": 1, "aoe": False},
    "Perfected Strike": {"type": "attack_per_strike", "base_damage": 6, "bonus_per_strike": 2},
    "Shrug It Off": {"type": "block+draw", "amount": 8, "draw": 1},
    "Sword Boomerang": {"type": "attack", "damage": 3, "hits": 3, "aoe": False,
                        "random_target": True},
    "Thunderclap": {"type": "attack+debuff", "damage": 4, "hits": 1, "aoe": True,
                    "debuffs": [("Vulnerable", 1)]},
    "Wild Strike": {"type": "attack+shuffle_wound", "damage": 12, "hits": 1, "aoe": False},
    "Twin Strike": {"type": "attack", "damage": 5, "hits": 2, "aoe": False},
    "Pommel Strike": {"type": "attack+draw", "damage": 9, "hits": 1, "aoe": False, "draw": 1},
    "Armaments": {"type": "block+upgrade", "amount": 5},
    "Warcry": {"type": "draw+topdeck", "draw": 2},
    "True Grit": {"type": "block+exhaust", "amount": 7, "exhaust_count": 1},
    "Hemokinesis": {"type": "attack+self_dmg", "damage": 14, "self_damage": 2, "hits": 1},
    "Blood for Blood": {"type": "attack", "damage": 18, "hits": 1, "aoe": False},
    "Carnage": {"type": "attack", "damage": 20, "hits": 1, "aoe": False},
    "Inflame": {"type": "power", "buff": "Strength", "amount": 2},
    "Combust": {"type": "power_combust", "hp_loss": 1, "damage": 5},
    "Corruption": {"type": "power_corruption"},
    "Demon Form": {"type": "power_demon_form", "amount": 2},
    "Feel No Pain": {"type": "power", "buff": "Feel No Pain", "amount": 3},
    "Fire Breathing": {"type": "power", "buff": "Fire Breathing", "amount": 6},
    "Rupture": {"type": "power", "buff": "Rupture", "amount": 1},
    "Metallicize": {"type": "power", "buff": "Metallicize", "amount": 3},
    "Rage": {"type": "power", "buff": "Rage", "amount": 3},
    "Evolve": {"type": "power", "buff": "Evolve", "amount": 1},
    "Dark Embrace": {"type": "power", "buff": "Dark Embrace", "amount": 1},
    "Juggernaut": {"type": "power", "buff": "Juggernaut", "amount": 5},
    "Barricade": {"type": "power", "buff": "Barricade", "amount": 0},
    "Berserk": {"type": "power+debuff", "self_debuffs": [("Vulnerable", 2)],
                "buff": "Berserk", "amount": 0},
    "Brutality": {"type": "power", "buff": "Brutality", "amount": 0},
    "Limit Break": {"type": "special_limit_break"},
    "Offering": {"type": "special_offering", "hp_loss": 6, "draw": 3},
    "Feed": {"type": "attack+permanent_hp", "damage": 10, "hp_gain": 3},
    # Status cards (can't normally be played)
    "Wound": {"type": "unplayable"},
    "Dazed": {"type": "unplayable"},
    "Slimed": {"type": "block", "amount": 0},  # costs 0, exhaust — treat as skip
    "Burn": {"type": "unplayable"},
    "Void": {"type": "unplayable"},
    "Curse of the Bell": {"type": "unplayable"},
    "Regret": {"type": "unplayable"},
    "Shame": {"type": "unplayable"},
    "Doubt": {"type": "unplayable"},
    "Injury": {"type": "unplayable"},
    "Normality": {"type": "unplayable"},
    "Pain": {"type": "unplayable"},
    "Parasite": {"type": "unplayable"},
    "Pride": {"type": "unplayable"},
    "Writhe": {"type": "unplayable"},
    "Clumsy": {"type": "unplayable"},
    "Decay": {"type": "unplayable"},
}


# ---------------------------------------------------------------------------
# Power / status effect helpers
# ---------------------------------------------------------------------------

def get_power(entity: dict, name: str) -> int:
    """Get the amount of a power/buff/debuff on an entity."""
    for p in entity.get("powers", []):
        if p.get("name") == name:
            return p.get("amount", 1)
    return 0


def set_power(entity: dict, name: str, amount: int) -> None:
    """Set/update a power on an entity. Removes if amount <= 0."""
    powers = entity.setdefault("powers", [])
    for p in powers:
        if p.get("name") == name:
            if amount <= 0:
                powers.remove(p)
            else:
                p["amount"] = amount
            return
    if amount > 0:
        powers.append({"name": name, "amount": amount})


def add_power(entity: dict, name: str, delta: int) -> None:
    """Add delta to a power (create if absent)."""
    current = get_power(entity, name)
    set_power(entity, name, current + delta)


def tick_power(entity: dict, name: str, delta: int = 1) -> None:
    """Decrement a duration-based power (Weak, Vuln, etc.) at end of turn."""
    current = get_power(entity, name)
    if current > 0:
        set_power(entity, name, max(0, current - delta))


# ---------------------------------------------------------------------------
# Damage calculation
# ---------------------------------------------------------------------------

def calc_attack_damage(base: int, attacker: dict, defender: dict, is_attack_card: bool = True) -> int:
    """Apply Strength, Weak, Vulnerable modifiers to base damage."""
    dmg = base

    # Attacker Strength adds flat
    strength = get_power(attacker, "Strength")
    dmg += strength

    # Attacker Weak reduces by 25%
    if get_power(attacker, "Weak") > 0:
        dmg = math.floor(dmg * 0.75)

    # Defender Vulnerable takes 50% more (only from attack cards)
    if is_attack_card and get_power(defender, "Vulnerable") > 0:
        dmg = math.floor(dmg * 1.5)

    return max(0, dmg)


def apply_damage_to_entity(entity: dict, damage: int) -> int:
    """Apply damage to an entity, respecting block. Returns actual HP damage taken."""
    block = entity.get("block", 0)
    absorbed = min(block, damage)
    entity["block"] = block - absorbed
    remaining = damage - absorbed
    entity["hp"] = max(0, entity.get("hp", 0) - remaining)
    return remaining


# ---------------------------------------------------------------------------
# Card application
# ---------------------------------------------------------------------------

def apply_card(state: dict, card_index: int, target_id: str | None) -> dict:
    """Apply a card from the player's hand. Returns new state (deep copy).

    Does NOT handle drawing — that happens in end_turn or via draw effects.
    """
    state = copy.deepcopy(state)
    player = state["player"]
    hand = state.get("hand", [])
    enemies = state.get("enemies", [])

    if card_index >= len(hand):
        return state  # shouldn't happen after validation

    card = hand[card_index]
    card_name = card.get("name", "")
    cost = card.get("cost", 0)
    if isinstance(cost, int):
        player["energy"] = player.get("energy", 0) - cost

    # Move card from hand to discard (or exhaust if card says so)
    hand.pop(card_index)
    exhaust_card = card.get("exhaust", False)

    def discard_or_exhaust(c: dict) -> None:
        if exhaust_card or c.get("exhaust"):
            state.setdefault("exhaust_pile", []).append(c)
        else:
            state.setdefault("discard_pile", []).append(c)

    # Find target enemy
    target_enemy = None
    if target_id:
        for e in enemies:
            if e.get("entity_id") == target_id:
                target_enemy = e
                break
    elif enemies:
        target_enemy = enemies[0]  # default to first

    # Look up card effect
    db_entry = CARD_DB.get(card_name)
    if db_entry is None:
        # Try to parse from description generically
        db_entry = _infer_card_from_description(card)

    effect_type = db_entry.get("type", "") if db_entry else ""

    # ---- Attack ----
    if effect_type == "attack":
        hits = db_entry.get("hits", 1)
        base_dmg = db_entry.get("damage", 0)
        aoe = db_entry.get("aoe", False)
        targets = enemies if aoe else ([target_enemy] if target_enemy else [])
        for _ in range(hits):
            for t in targets:
                dmg = calc_attack_damage(base_dmg, player, t)
                apply_damage_to_entity(t, dmg)

    elif effect_type == "block":
        amt = db_entry.get("amount", 0)
        amt = _apply_block_modifiers(amt, player)
        player["block"] = player.get("block", 0) + amt

    elif effect_type == "attack+block":
        base_dmg = db_entry.get("damage", 0)
        hits = db_entry.get("hits", 1)
        if target_enemy:
            for _ in range(hits):
                dmg = calc_attack_damage(base_dmg, player, target_enemy)
                apply_damage_to_entity(target_enemy, dmg)
        amt = db_entry.get("amount", 0)
        player["block"] = player.get("block", 0) + _apply_block_modifiers(amt, player)

    elif effect_type == "attack+debuff":
        hits = db_entry.get("hits", 1)
        base_dmg = db_entry.get("damage", 0)
        aoe = db_entry.get("aoe", False)
        targets = enemies if aoe else ([target_enemy] if target_enemy else [])
        for _ in range(hits):
            for t in targets:
                dmg = calc_attack_damage(base_dmg, player, t)
                apply_damage_to_entity(t, dmg)
        debuffs = db_entry.get("debuffs", [])
        for t in targets:
            for dname, damt in debuffs:
                add_power(t, dname, damt)

    elif effect_type == "attack_from_block":
        block_val = player.get("block", 0)
        if target_enemy:
            dmg = calc_attack_damage(block_val, player, target_enemy)
            apply_damage_to_entity(target_enemy, dmg)

    elif effect_type == "attack_strength_mult":
        base_dmg = db_entry.get("damage", 0)
        s_mult = db_entry.get("strength_mult", 1)
        strength = get_power(player, "Strength")
        total = base_dmg + strength * s_mult
        # Weak/Vuln still apply but strength already counted in total
        # re-do without strength for weak calc
        dmg_no_str = base_dmg
        if get_power(player, "Weak") > 0:
            dmg_no_str = math.floor(dmg_no_str * 0.75)
            total = math.floor(total * 0.75)
        if target_enemy and get_power(target_enemy, "Vulnerable") > 0:
            total = math.floor(total * 1.5)
        if target_enemy:
            apply_damage_to_entity(target_enemy, max(0, total))

    elif effect_type == "attack_per_strike":
        strike_count = sum(
            1 for c in (
                state.get("hand", []) + state.get("draw_pile", []) +
                state.get("discard_pile", []) + [card]
            )
            if "strike" in c.get("name", "").lower()
        )
        dmg_base = db_entry.get("base_damage", 6) + db_entry.get("bonus_per_strike", 2) * strike_count
        if target_enemy:
            dmg = calc_attack_damage(dmg_base, player, target_enemy)
            apply_damage_to_entity(target_enemy, dmg)

    elif effect_type == "block+draw":
        amt = db_entry.get("amount", 0)
        player["block"] = player.get("block", 0) + _apply_block_modifiers(amt, player)
        _draw_cards(state, db_entry.get("draw", 1))

    elif effect_type == "attack+draw":
        base_dmg = db_entry.get("damage", 0)
        if target_enemy:
            dmg = calc_attack_damage(base_dmg, player, target_enemy)
            apply_damage_to_entity(target_enemy, dmg)
        _draw_cards(state, db_entry.get("draw", 1))

    elif effect_type == "power":
        buff = db_entry.get("buff", "")
        amt = db_entry.get("amount", 1)
        add_power(player, buff, amt)
        exhaust_card = True  # powers exhaust by convention

    elif effect_type == "power_temp":
        buff = db_entry.get("buff", "")
        amt = db_entry.get("amount", 2)
        add_power(player, buff, amt)
        # Add temporary marker so it wears off at end of turn
        add_power(player, f"_temp_{buff}", amt)
        exhaust_card = True

    elif effect_type == "attack+self_dmg":
        base_dmg = db_entry.get("damage", 0)
        self_dmg = db_entry.get("self_damage", 0)
        if target_enemy:
            dmg = calc_attack_damage(base_dmg, player, target_enemy)
            apply_damage_to_entity(target_enemy, dmg)
        apply_damage_to_entity(player, self_dmg)

    elif effect_type == "special_limit_break":
        strength = get_power(player, "Strength")
        add_power(player, "Strength", strength)
        exhaust_card = True

    elif effect_type == "special_offering":
        hp_loss = db_entry.get("hp_loss", 6)
        apply_damage_to_entity(player, hp_loss)
        player["energy"] = player.get("energy", 0) + 2
        _draw_cards(state, db_entry.get("draw", 3))
        exhaust_card = True

    elif effect_type == "attack+permanent_hp":
        base_dmg = db_entry.get("damage", 0)
        hp_gain = db_entry.get("hp_gain", 3)
        if target_enemy and target_enemy.get("hp", 1) <= base_dmg:
            # Would kill — gain HP
            player["max_hp"] = player.get("max_hp", 80) + hp_gain
            player["hp"] = min(player.get("hp", 80) + hp_gain, player["max_hp"])
        if target_enemy:
            dmg = calc_attack_damage(base_dmg, player, target_enemy)
            apply_damage_to_entity(target_enemy, dmg)
        exhaust_card = True

    elif effect_type == "wild_strike_variant" or effect_type == "attack+shuffle_wound":
        base_dmg = db_entry.get("damage", 12)
        if target_enemy:
            dmg = calc_attack_damage(base_dmg, player, target_enemy)
            apply_damage_to_entity(target_enemy, dmg)
        # Shuffle a Wound into draw pile
        wound = {"name": "Wound", "cost": 1, "type": "Status", "exhausts": True}
        state.setdefault("draw_pile", []).append(wound)

    # Remove dead enemies
    state["enemies"] = [e for e in enemies if e.get("hp", 0) > 0]

    discard_or_exhaust(card)
    return state


# ---------------------------------------------------------------------------
# Block modifiers
# ---------------------------------------------------------------------------

def _apply_block_modifiers(amount: int, player: dict) -> int:
    """Apply Dexterity and Frail to a block amount."""
    dex = get_power(player, "Dexterity")
    amount += dex
    if get_power(player, "Frail") > 0:
        amount = math.floor(amount * 0.75)
    return max(0, amount)


# ---------------------------------------------------------------------------
# Draw cards
# ---------------------------------------------------------------------------

def _draw_cards(state: dict, count: int) -> None:
    """Draw count cards from draw pile into hand. Reshuffles if needed."""
    for _ in range(count):
        draw_pile = state.get("draw_pile", [])
        hand = state.get("hand", [])
        if not draw_pile:
            # Reshuffle discard into draw
            discard = state.get("discard_pile", [])
            if discard:
                random.shuffle(discard)
                state["draw_pile"] = discard
                state["discard_pile"] = []
                draw_pile = state["draw_pile"]
            else:
                break  # Nothing to draw
        card = draw_pile.pop()
        hand.append(card)
    state["hand"] = hand


# ---------------------------------------------------------------------------
# End turn
# ---------------------------------------------------------------------------

def end_turn(state: dict) -> dict:
    """Process end of player turn and enemy actions. Returns new state."""
    state = copy.deepcopy(state)
    player = state["player"]
    enemies = state.get("enemies", [])

    # --- Player end of turn effects ---
    # Note: player block clears at start of next turn, NOT here.
    # It remains active to absorb enemy attacks this end-of-turn phase.
    # (Barricade prevents even that clearance on the next turn start.)

    # Remove temp buffs (Flex etc.)
    powers = player.get("powers", [])
    temp_powers = [p for p in powers if p.get("name", "").startswith("_temp_")]
    for tp in temp_powers:
        real_name = tp["name"][len("_temp_"):]
        add_power(player, real_name, -tp.get("amount", 0))
        powers.remove(tp)

    # Tick player debuffs
    for debuff in ("Weak", "Frail", "Vulnerable"):
        tick_power(player, debuff)

    # Combust
    combust = get_power(player, "Combust")
    if combust:
        player["hp"] = max(0, player.get("hp", 0) - 1)
        for e in enemies:
            apply_damage_to_entity(e, combust)

    # Discard hand
    discard = state.setdefault("discard_pile", [])
    hand = state.get("hand", [])
    for c in hand:
        discard.append(c)
    state["hand"] = []

    # --- Enemy turn ---
    for enemy in enemies:
        if enemy.get("hp", 0) <= 0:
            continue

        intent = enemy.get("intent", {})
        intent_type = intent.get("type", "")
        enemy["block"] = 0  # Reset enemy block (simplified — most enemies don't keep it)

        if "attack" in intent_type.lower() or intent_type == "attack":
            base_dmg = intent.get("damage", 0)
            hits = intent.get("hits", 1)
            for _ in range(hits):
                dmg = calc_attack_damage(base_dmg, enemy, player, is_attack_card=True)
                apply_damage_to_entity(player, dmg)

        elif "block" in intent_type.lower() or "defend" in intent_type.lower():
            block_amt = intent.get("block", 0)
            enemy["block"] = enemy.get("block", 0) + block_amt

        elif "buff" in intent_type.lower() or "strength" in intent_type.lower():
            # Generic buff intent — apply Strength
            buff_amt = intent.get("amount", 3)
            add_power(enemy, "Strength", buff_amt)

        # Tick enemy debuffs
        for debuff in ("Weak", "Frail", "Vulnerable"):
            tick_power(enemy, debuff)
        for debuff in ("Strength",):
            pass  # Strength doesn't tick

        # Ritual power
        ritual = get_power(enemy, "Ritual")
        if ritual:
            add_power(enemy, "Strength", ritual)

    # Remove dead enemies
    state["enemies"] = [e for e in enemies if e.get("hp", 0) > 0]

    # Check combat end
    if not state["enemies"]:
        state["state_type"] = "combat_rewards"
        return state

    if player.get("hp", 0) <= 0:
        state["state_type"] = "game_over"
        return state

    # --- Start of player turn ---
    # Clear block (unless Barricade)
    if get_power(player, "Barricade") <= 0:
        player["block"] = 0

    player["energy"] = player.get("max_energy", 3)

    # Metallicize
    met = get_power(player, "Metallicize")
    if met:
        player["block"] = player.get("block", 0) + met

    # Brutality: draw 1, lose 1 hp
    if get_power(player, "Brutality") > 0:
        player["hp"] = max(0, player.get("hp", 0) - 1)
        _draw_cards(state, 1)

    # Berserk: extra energy when vulnerable
    if get_power(player, "Berserk") > 0 and get_power(player, "Vulnerable") > 0:
        player["energy"] += 1

    # Draw 5 cards
    _draw_cards(state, 5)

    state["turn"] = state.get("turn", 1) + 1
    return state


# ---------------------------------------------------------------------------
# Top-level step function
# ---------------------------------------------------------------------------

def step(state: dict, action: dict) -> dict:
    """Apply an action to a state, return new state.

    action: {"action": "play_card", "card_index": int, "target": str?}
         or {"action": "end_turn"}
         or {"action": "use_potion", "slot": int, "target": str?}
    """
    action_type = action.get("action")
    if action_type == "play_card":
        return apply_card(state, action.get("card_index", 0), action.get("target"))
    elif action_type == "end_turn":
        return end_turn(state)
    elif action_type == "use_potion":
        return _use_potion(state, action.get("slot", 0), action.get("target"))
    return state


def is_terminal(state: dict) -> tuple[bool, bool]:
    """Returns (is_done, is_victory). Victory = combat_rewards state."""
    state_type = state.get("state_type", "")
    if state_type == "combat_rewards":
        return True, True
    if state_type == "game_over":
        return True, False
    if state.get("player", {}).get("hp", 1) <= 0:
        return True, False
    if not state.get("enemies"):
        return True, True
    return False, False


# ---------------------------------------------------------------------------
# Potion handling
# ---------------------------------------------------------------------------

def _use_potion(state: dict, slot: int, target_id: str | None) -> dict:
    state = copy.deepcopy(state)
    potions = state.get("potions", [])
    if slot >= len(potions):
        return state
    potion = potions[slot]
    if not potion or not potion.get("name"):
        return state

    name = potion.get("name", "")
    player = state["player"]
    enemies = state.get("enemies", [])
    target = next((e for e in enemies if e.get("entity_id") == target_id), None)
    if not target and enemies:
        target = enemies[0]

    # Simple potion effects
    if "Fire" in name and target:
        apply_damage_to_entity(target, 20)
    elif "Strength" in name:
        add_power(player, "Strength", 5)
    elif "Dexterity" in name:
        add_power(player, "Dexterity", 5)
    elif "Block" in name or "Iron" in name:
        player["block"] = player.get("block", 0) + 12
    elif "Heal" in name or "Health" in name or "Fairy" in name:
        heal = max(10, player.get("max_hp", 80) // 3)
        player["hp"] = min(player.get("hp", 0) + heal, player.get("max_hp", 80))
    elif "Energy" in name or "Colorless" in name:
        player["energy"] = player.get("energy", 0) + 2
    elif "Explosive" in name:
        for e in enemies:
            apply_damage_to_entity(e, 10)
    elif "Fear" in name or "Weak" in name.lower():
        for e in enemies:
            add_power(e, "Weak", 3)
    elif "Vulnerable" in name.lower() or "Flex" in name:
        for e in enemies:
            add_power(e, "Vulnerable", 3)

    # Remove potion from slot
    potions[slot] = {}
    state["potions"] = potions
    state["enemies"] = [e for e in enemies if e.get("hp", 0) > 0]
    return state


# ---------------------------------------------------------------------------
# Description-based card inference (fallback for unknown cards)
# ---------------------------------------------------------------------------

def _infer_card_from_description(card: dict) -> dict:
    """Try to infer card effect from its description text. Very rough."""
    desc = card.get("description", "").lower()
    name = card.get("name", "")

    # Look for "deal X damage" pattern
    import re
    attack_match = re.search(r"deal\s+(\d+)\s+damage", desc)
    block_match = re.search(r"gain\s+(\d+)\s+block", desc)

    if attack_match and block_match:
        return {
            "type": "attack+block",
            "damage": int(attack_match.group(1)),
            "amount": int(block_match.group(1)),
            "hits": 1,
            "aoe": False,
        }
    elif attack_match:
        hits = 1
        multi = re.search(r"(\d+)\s+times", desc)
        if multi:
            hits = int(multi.group(1))
        aoe = "all" in desc or "every enemy" in desc
        return {
            "type": "attack",
            "damage": int(attack_match.group(1)),
            "hits": hits,
            "aoe": aoe,
        }
    elif block_match:
        return {"type": "block", "amount": int(block_match.group(1))}

    # Default: skip (treat as 0-damage attack so training doesn't crash)
    return {"type": "attack", "damage": 0, "hits": 1, "aoe": False}


# ---------------------------------------------------------------------------
# State factory helpers (for tests / offline scenario generation)
# ---------------------------------------------------------------------------

def make_ironclad_starter_state(
    enemy_hp: int = 40,
    enemy_name: str = "Jaw Worm",
    enemy_id: str = "jaw_worm_0",
    enemy_intent_damage: int = 11,
    player_hp: int = 80,
    seed: int | None = None,
) -> dict:
    """Create a standard Act 1 starting combat state for testing.

    If ``seed`` is given, the opening hand is shuffled deterministically so the
    same seed always yields the same starting state (needed for reproducible
    RL group rollouts).
    """
    rng = random.Random(seed) if seed is not None else random
    starter_deck = [
        {"name": "Strike", "cost": 1, "type": "Attack",
         "target_type": "enemy", "description": "Deal 6 damage."},
        {"name": "Strike", "cost": 1, "type": "Attack",
         "target_type": "enemy", "description": "Deal 6 damage."},
        {"name": "Strike", "cost": 1, "type": "Attack",
         "target_type": "enemy", "description": "Deal 6 damage."},
        {"name": "Strike", "cost": 1, "type": "Attack",
         "target_type": "enemy", "description": "Deal 6 damage."},
        {"name": "Strike", "cost": 1, "type": "Attack",
         "target_type": "enemy", "description": "Deal 6 damage."},
        {"name": "Defend", "cost": 1, "type": "Skill",
         "target_type": "none", "description": "Gain 5 block."},
        {"name": "Defend", "cost": 1, "type": "Skill",
         "target_type": "none", "description": "Gain 5 block."},
        {"name": "Defend", "cost": 1, "type": "Skill",
         "target_type": "none", "description": "Gain 5 block."},
        {"name": "Defend", "cost": 1, "type": "Skill",
         "target_type": "none", "description": "Gain 5 block."},
        {"name": "Bash", "cost": 2, "type": "Attack",
         "target_type": "enemy", "description": "Deal 8 damage. Apply 2 Vulnerable."},
    ]
    # Deal hand from top
    rng.shuffle(starter_deck)
    hand = starter_deck[:5]
    draw = starter_deck[5:]

    return {
        "state_type": "monster",
        "floor": 1,
        "turn": 1,
        "player": {
            "hp": player_hp,
            "max_hp": player_hp,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "gold": 99,
            "powers": [],
        },
        "hand": hand,
        "draw_pile": draw,
        "discard_pile": [],
        "exhaust_pile": [],
        "potions": [],
        "enemies": [
            {
                "entity_id": enemy_id,
                "name": enemy_name,
                "hp": enemy_hp,
                "max_hp": enemy_hp,
                "block": 0,
                "powers": [],
                "intent": {
                    "type": "attack",
                    "damage": enemy_intent_damage,
                    "hits": 1,
                },
            }
        ],
    }
