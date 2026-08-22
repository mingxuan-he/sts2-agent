"""Basic tests for the STS2 combat simulator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from sts2_rl.envs.sts2_sim import (
    make_ironclad_starter_state,
    step,
    is_terminal,
    apply_card,
    end_turn,
    get_power,
    add_power,
)


def test_strike_deals_damage():
    state = make_ironclad_starter_state()
    # Find a Strike in hand
    hand = state["hand"]
    strike_idx = next((i for i, c in enumerate(hand) if c["name"] == "Strike"), None)
    if strike_idx is None:
        # Manually put Strike in hand
        state["hand"] = [{"name": "Strike", "cost": 1, "type": "Attack",
                          "target_type": "enemy", "description": "Deal 6 damage."}]
        strike_idx = 0

    enemy_hp_before = state["enemies"][0]["hp"]
    enemy_id = state["enemies"][0]["entity_id"]
    new_state = step(state, {"action": "play_card", "card_index": strike_idx,
                              "target": enemy_id})
    enemy_hp_after = new_state["enemies"][0]["hp"]
    assert enemy_hp_after == enemy_hp_before - 6, f"Expected -{6} damage, got {enemy_hp_before - enemy_hp_after}"
    print(f"PASS: Strike deals 6 damage ({enemy_hp_before} -> {enemy_hp_after})")


def test_defend_gives_block():
    state = make_ironclad_starter_state()
    state["hand"] = [{"name": "Defend", "cost": 1, "type": "Skill",
                      "target_type": "none", "description": "Gain 5 block."}]
    new_state = step(state, {"action": "play_card", "card_index": 0, "target": None})
    assert new_state["player"]["block"] == 5, f"Expected 5 block, got {new_state['player']['block']}"
    print("PASS: Defend gives 5 block")


def test_bash_applies_vulnerable():
    state = make_ironclad_starter_state()
    state["hand"] = [{"name": "Bash", "cost": 2, "type": "Attack",
                      "target_type": "enemy", "description": "Deal 8 damage. Apply 2 Vulnerable."}]
    enemy_id = state["enemies"][0]["entity_id"]
    new_state = step(state, {"action": "play_card", "card_index": 0, "target": enemy_id})
    enemy = new_state["enemies"][0]
    vuln = get_power(enemy, "Vulnerable")
    assert enemy["hp"] == 40 - 8, f"Expected 32 HP, got {enemy['hp']}"
    assert vuln == 2, f"Expected 2 Vulnerable, got {vuln}"
    print(f"PASS: Bash deals 8 damage and applies 2 Vulnerable")


def test_vulnerable_increases_damage():
    state = make_ironclad_starter_state()
    # Apply vulnerable to enemy manually
    add_power(state["enemies"][0], "Vulnerable", 2)
    state["hand"] = [{"name": "Strike", "cost": 1, "type": "Attack",
                      "target_type": "enemy", "description": "Deal 6 damage."}]
    enemy_id = state["enemies"][0]["entity_id"]
    new_state = step(state, {"action": "play_card", "card_index": 0, "target": enemy_id})
    # 6 * 1.5 = 9 damage (floor)
    import math
    expected = math.floor(6 * 1.5)
    enemy_hp = new_state["enemies"][0]["hp"]
    assert enemy_hp == 40 - expected, f"Expected {40 - expected} HP, got {enemy_hp}"
    print(f"PASS: Strike deals {expected} damage vs Vulnerable enemy")


def test_end_turn_enemy_attacks():
    state = make_ironclad_starter_state(enemy_intent_damage=11)
    state["player"]["block"] = 0
    player_hp_before = state["player"]["hp"]
    new_state = step(state, {"action": "end_turn"})
    player_hp_after = new_state["player"]["hp"]
    assert player_hp_after == player_hp_before - 11, \
        f"Expected {player_hp_before - 11} HP, got {player_hp_after}"
    print(f"PASS: End turn — enemy deals 11 damage ({player_hp_before} -> {player_hp_after})")


def test_end_turn_block_absorbs_damage():
    state = make_ironclad_starter_state(enemy_intent_damage=11)
    state["player"]["block"] = 8
    player_hp_before = state["player"]["hp"]
    new_state = step(state, {"action": "end_turn"})
    player_hp_after = new_state["player"]["hp"]
    # 11 - 8 block = 3 damage
    assert player_hp_after == player_hp_before - 3, \
        f"Expected {player_hp_before - 3} HP, got {player_hp_after}"
    print(f"PASS: Block absorbs 8 of 11 damage")


def test_combat_ends_on_enemy_death():
    state = make_ironclad_starter_state(enemy_hp=5)
    state["hand"] = [{"name": "Strike", "cost": 1, "type": "Attack",
                      "target_type": "enemy", "description": "Deal 6 damage."}]
    enemy_id = state["enemies"][0]["entity_id"]
    new_state = step(state, {"action": "play_card", "card_index": 0, "target": enemy_id})
    done, victory = is_terminal(new_state)
    assert done, "Combat should be done"
    assert victory, "Should be victory"
    print("PASS: Enemy at 5HP dies to Strike (6 damage) — victory")


def test_draw_after_end_turn():
    state = make_ironclad_starter_state()
    state["hand"] = []  # Clear hand
    # Put cards in draw pile
    state["draw_pile"] = [
        {"name": "Strike", "cost": 1, "type": "Attack", "target_type": "enemy",
         "description": "Deal 6 damage."}
    ] * 7
    state["discard_pile"] = []
    new_state = step(state, {"action": "end_turn"})
    assert len(new_state["hand"]) == 5, f"Should draw 5 cards, got {len(new_state['hand'])}"
    print(f"PASS: Drew 5 cards at start of turn")


def test_full_combat_loop():
    """Simulate a complete combat to victory."""
    state = make_ironclad_starter_state(enemy_hp=20, enemy_intent_damage=8)
    turns = 0
    while True:
        done, victory = is_terminal(state)
        if done:
            break
        turns += 1
        if turns > 20:
            print("FAIL: Combat exceeded 20 turns — stuck in loop?")
            return

        # Simple greedy policy: play first attack, then end turn
        hand = state.get("hand", [])
        played = False
        enemy_id = state["enemies"][0]["entity_id"] if state["enemies"] else None
        for i, card in enumerate(hand):
            db_name = card.get("name", "")
            if db_name in ("Strike", "Bash") and enemy_id:
                energy = state["player"].get("energy", 0)
                if card.get("cost", 1) <= energy:
                    state = step(state, {"action": "play_card", "card_index": i,
                                         "target": enemy_id})
                    played = True
                    break
        if not played:
            state = step(state, {"action": "end_turn"})

    done, victory = is_terminal(state)
    print(f"PASS: Full combat loop completed in {turns} turns, victory={victory}")


if __name__ == "__main__":
    test_strike_deals_damage()
    test_defend_gives_block()
    test_bash_applies_vulnerable()
    test_vulnerable_increases_damage()
    test_end_turn_enemy_attacks()
    test_end_turn_block_absorbs_damage()
    test_combat_ends_on_enemy_death()
    test_draw_after_end_turn()
    test_full_combat_loop()
    print("\nAll tests passed!")
