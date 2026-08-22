"""Unit tests for the game-service serializer (observation contract)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from sts2_service.serializer import (  # noqa: E402
    _clean_desc,
    _resolve_vars,
    render_piles,
    serialize,
)


def test_clean_desc_strips_bbcode_and_templates():
    assert _clean_desc("[b]Deal[/b] {Damage:diff()} damage.") == "Deal [Damage] damage."


def test_clean_desc_plural_and_conditionals():
    text = "Obtain {Cards} {Cards:plural:card reward|card rewards}."
    assert _clean_desc(text) == "Obtain [Cards] [Cards:card reward|card rewards]."
    assert _clean_desc("{IfUpgraded:show:Deal 12|Deal 8} damage.") == "Deal 8 damage."


def test_resolve_vars_case_insensitive_and_plural():
    assert _resolve_vars("Deal [Damage] damage.", {"damage": 10}) == "Deal 10 damage."
    assert _resolve_vars("Obtain [Cards] [Cards:card|cards].", {"cards": 2}) == "Obtain 2 cards."
    assert _resolve_vars("Obtain [Cards] [Cards:card|cards].", {"cards": 1}) == "Obtain 1 card."
    # unknown vars stay bracketed, never crash
    assert _resolve_vars("Gain [Mystery].", {}) == "Gain [Mystery]."


COMBAT_STATE = {
    "type": "decision",
    "decision": "combat_play",
    "context": {"act": 1, "act_name": "Overgrowth", "floor": 2, "room_type": "Monster",
                "boss": {"id": "X_BOSS", "name": "Ceremonial Beast"}},
    "round": 1,
    "energy": 3,
    "max_energy": 3,
    "hand": [
        {"index": 0, "name": "Strike", "cost": 1, "type": "Attack", "can_play": True,
         "target_type": "AnyEnemy", "stats": {"damage": 6},
         "description": "Deal {Damage:diff()} damage."},
        {"index": 1, "name": "Defend", "cost": 1, "type": "Skill", "can_play": False,
         "target_type": "None", "stats": {"block": 5},
         "description": "Gain {Block:diff()} Block."},
    ],
    "enemies": [
        {"index": 0, "name": "Nibbit", "hp": 42, "max_hp": 42, "block": 0,
         "intents": [{"type": "Attack", "damage": 4, "hits": 2, "total_damage": 8}],
         "powers": [{"name": "Curl Up", "amount": 3}]},
    ],
    "player": {"name": "The Ironclad", "hp": 80, "max_hp": 80, "block": 0, "gold": 99,
               "deck_size": 10,
               "relics": [{"name": "Burning Blood", "vars": {"Heal": 6}}],
               "potions": []},
    "player_powers": [{"name": "Strength", "amount": 2}],
    "draw_pile_count": 5,
    "discard_pile_count": 0,
}


def test_combat_serialization_content():
    text = serialize(COMBAT_STATE)
    assert "Deal 6 damage." in text                       # template resolved from stats
    assert "UNPLAYABLE" in text                           # can_play surfaced
    assert "Burning Blood(Heal:6)" in text                # relics in combat view
    assert "Strength(2)" in text                          # player powers
    assert "Attack 4x2=8" in text                         # multi-hit intent w/ total
    assert "Act boss: Ceremonial Beast" in text           # map-informing boss
    assert "Draw: 5" in text
    assert "[Actions]" in text and "play_card" in text    # action vocabulary
    assert "{" not in text.replace("{card_index", "").replace("{potion_index", "")


def test_serialization_deterministic():
    assert serialize(COMBAT_STATE) == serialize(COMBAT_STATE)


def test_draw_pile_sorted_hides_order():
    piles = {
        "draw": [{"name": "Strike"}, {"name": "Bash"}, {"name": "Defend", "upgraded": True}],
        "discard": [{"name": "Strike"}],
        "exhaust": [],
    }
    text = render_piles(piles)
    assert "Bash, Defend+, Strike" in text     # sorted, not draw order
    assert "order hidden" in text
    assert "[Discard — 1] Strike" in text
    assert "[Exhaust — 0] (empty)" in text
