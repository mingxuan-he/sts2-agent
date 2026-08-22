#!/usr/bin/env python3
"""Record STS2 gameplay via the STS2MCP mod for SFT training data.

Run this on the Mac while playing STS2. It polls the mod's API for state
changes, and whenever it detects a new state (you played a card, ended a
turn, picked a card reward, etc.), it logs the before/after state pair.

The output is a JSONL file where each line is:
{
    "timestamp": "2026-06-04T18:30:00",
    "state_type": "monster",
    "state_before": { ... full game state JSON ... },
    "state_after": { ... full game state JSON ... },
    "action_inferred": { ... best guess at what action was taken ... }
}

Usage:
    python record_gameplay.py [--output traces.jsonl] [--poll-interval 0.3]

Just play normally. The script watches and records. Ctrl+C to stop.
"""

import argparse
import json
import hashlib
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Need httpx: pip install httpx")
    sys.exit(1)


STS2_BASE = "http://localhost:15526"
STATE_URL = f"{STS2_BASE}/api/v1/singleplayer"


def get_state(client: httpx.Client) -> dict | None:
    """Fetch current game state, return None if game not running."""
    try:
        resp = client.get(STATE_URL, params={"format": "json"}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except (httpx.ConnectError, httpx.ReadTimeout):
        return None


def state_hash(state: dict) -> str:
    """Quick hash to detect state changes without deep comparison."""
    # Hash a subset of frequently-changing fields
    key_parts = []
    if "player" in state:
        p = state["player"]
        key_parts.append(f"hp={p.get('hp')}")
        key_parts.append(f"block={p.get('block')}")
        key_parts.append(f"energy={p.get('energy')}")
        key_parts.append(f"gold={p.get('gold')}")
    if "hand" in state:
        key_parts.append(f"hand={len(state['hand'])}")
    if "enemies" in state:
        for e in state["enemies"]:
            key_parts.append(f"{e.get('entity_id')}={e.get('hp')}/{e.get('block')}")
    key_parts.append(f"type={state.get('state_type')}")
    key_parts.append(f"turn={state.get('turn')}")
    key_parts.append(f"floor={state.get('floor')}")
    raw = "|".join(key_parts)
    return hashlib.md5(raw.encode()).hexdigest()


def infer_action(before: dict, after: dict) -> dict:
    """Best-effort guess at what action was taken between two states.

    This is heuristic — the mod doesn't tell us what the player did,
    we infer from the state diff.
    """
    bt = before.get("state_type", "")
    at = after.get("state_type", "")

    # State type changed entirely (e.g., combat → rewards)
    if bt != at:
        return {"action": "state_transition", "from": bt, "to": at}

    # Same state type — look at diffs
    if bt in ("monster", "elite", "boss"):
        # Combat — check what changed
        hand_before = len(before.get("hand", []))
        hand_after = len(after.get("hand", []))
        energy_before = before.get("player", {}).get("energy", 0)
        energy_after = after.get("player", {}).get("energy", 0)
        turn_before = before.get("turn", 0)
        turn_after = after.get("turn", 0)

        if turn_after > turn_before:
            return {"action": "end_turn", "turn": turn_before}

        if hand_after < hand_before:
            # A card was played — try to figure out which one
            # Compare hand contents
            cards_before = [c.get("name") for c in before.get("hand", [])]
            cards_after = [c.get("name") for c in after.get("hand", [])]
            played = []
            remaining = list(cards_after)
            for c in cards_before:
                if c in remaining:
                    remaining.remove(c)
                else:
                    played.append(c)
            return {
                "action": "play_card",
                "cards_played": played,
                "energy_spent": energy_before - energy_after,
            }

    if bt == "card_reward":
        return {"action": "card_reward_decision"}

    if bt == "map":
        return {"action": "map_navigation"}

    if bt == "rest_site":
        return {"action": "rest_site_decision"}

    return {"action": "unknown", "state_type": bt}


def main():
    parser = argparse.ArgumentParser(description="Record STS2 gameplay")
    default_out = f"traces/{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    parser.add_argument("--output", "-o", default=default_out, help="Output JSONL file")
    parser.add_argument("--poll-interval", "-p", type=float, default=0.3,
                        help="Seconds between state polls")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client()

    print(f"STS2 Gameplay Recorder")
    print(f"Output: {output_path}")
    print(f"Poll interval: {args.poll_interval}s")
    print(f"Connecting to {STS2_BASE}...")
    print()

    prev_state = None
    prev_hash = None
    records = 0
    waiting_for_game = True

    try:
        while True:
            state = get_state(client)

            if state is None:
                if not waiting_for_game:
                    print("\n[!] Lost connection to game. Waiting...")
                    waiting_for_game = True
                time.sleep(1)
                continue

            if waiting_for_game:
                st = state.get("state_type", "?")
                print(f"[+] Connected! State: {st}")
                waiting_for_game = False

            h = state_hash(state)
            if h != prev_hash:
                # State changed!
                if prev_state is not None:
                    action = infer_action(prev_state, state)
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "state_type_before": prev_state.get("state_type"),
                        "state_type_after": state.get("state_type"),
                        "state_before": prev_state,
                        "state_after": state,
                        "action_inferred": action,
                    }
                    with open(output_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    records += 1

                    # Brief status line
                    st = state.get("state_type", "?")
                    act = action.get("action", "?")
                    print(f"\r[{records}] {act} → {st}", end="", flush=True)

                prev_state = state
                prev_hash = h

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(f"\n\nStopped. Recorded {records} state transitions to {output_path}")


if __name__ == "__main__":
    main()
