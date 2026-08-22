"""Serialize sts2-cli decision-point JSON into compact deterministic text.

This is the observation contract for the pod agent (and, later, Track 1 SFT/RL
data). Rules, per docs/track2-pod-design.md:

- Compact by default: the per-turn observation stays lean. Full detail (deck,
  piles, relics/potions with descriptions, act map) lives behind the service's
  detail endpoints and is never silently dropped from the agent's information
  set — only deferred behind a query.
- Deterministic: same state -> byte-identical text, stable ordering.
- Never leak hidden info: draw-pile ORDER is hidden (detail endpoint sorts);
  this module only ever sees counts for piles.

Input shapes are the payloads produced by sts2-cli's RunSimulator
(decision: combat_play / map_select / event_choice / card_reward / card_select
/ bundle_select / rest_site / shop / game_over / unknown).
"""

from __future__ import annotations

import re
from typing import Any

# Actions the agent may take at each decision point, shown inline so the model
# never has to guess the verb vocabulary.
_ACTIONS: dict[str, str] = {
    "combat_play": (
        "play_card {card_index, target_index?} | use_potion {potion_index, target_index?} | "
        "discard_potion {potion_index} | end_turn"
    ),
    "map_select": "select_map_node {col, row}",
    "event_choice": "choose_option {option_index}",
    "card_reward": "select_card_reward {card_index} | skip_card_reward",
    "card_select": 'select_cards {indices: "i,j,..."} | skip_select (if allowed)',
    "bundle_select": "select_bundle {bundle_index}",
    "rest_site": "choose_option {option_index}",
    "shop": (
        "buy_card {card_index} | buy_relic {relic_index} | buy_potion {potion_index} | "
        "remove_card {card_index} | leave_room"
    ),
}


def serialize(state: dict[str, Any]) -> str:
    """Dispatch on the CLI decision type."""
    decision = state.get("decision", "")
    fn = {
        "combat_play": _combat,
        "map_select": _map_select,
        "event_choice": _event,
        "card_reward": _card_reward,
        "card_select": _card_select,
        "bundle_select": _bundle_select,
        "rest_site": _rest_site,
        "shop": _shop,
        "game_over": _game_over,
    }.get(decision)
    if fn is None:
        return f"[State] {decision or state.get('type', '?')}\n{state}"
    parts = fn(state)
    actions = _ACTIONS.get(decision)
    if actions:
        parts.append(f"[Actions] {actions}")
    return "\n".join(parts)


# ── shared pieces ──────────────────────────────────────────────────────────


def _context_line(state: dict[str, Any]) -> str:
    ctx = state.get("context") or {}
    bits = [f"Act {ctx.get('act', '?')} ({ctx.get('act_name', '?')})", f"Floor {ctx.get('floor', '?')}"]
    boss = ctx.get("boss")
    if boss:
        bits.append(f"Act boss: {boss.get('name', '?')}")
    room = ctx.get("room_type")
    if room:
        bits.append(f"Room: {room}")
    return "[Run] " + " | ".join(bits)


def _player_line(state: dict[str, Any], energy: bool = False) -> str:
    p = state.get("player") or {}
    bits = [f"HP {p.get('hp', '?')}/{p.get('max_hp', '?')}"]
    if p.get("block"):
        bits.append(f"Block {p['block']}")
    if energy:
        bits.append(f"Energy {state.get('energy', '?')}/{state.get('max_energy', '?')}")
    bits.append(f"Gold {p.get('gold', 0)}")
    bits.append(f"Deck {p.get('deck_size', '?')} cards")
    return f"[Player] {p.get('name', '?')} | " + " | ".join(bits)


def _relics_line(state: dict[str, Any]) -> str | None:
    relics = (state.get("player") or {}).get("relics") or []
    if not relics:
        return None
    out = []
    for r in relics:
        name = r.get("name", "?")
        vars_ = r.get("vars") or {}
        if vars_:
            counters = ",".join(f"{k}:{v}" for k, v in vars_.items())
            out.append(f"{name}({counters})")
        else:
            out.append(name)
    return "[Relics] " + ", ".join(out) + "  (descriptions: relics endpoint)"


def _potions_line(state: dict[str, Any]) -> str | None:
    potions = (state.get("player") or {}).get("potions") or []
    named = [p for p in potions if p and p.get("name")]
    if not named:
        return None
    out = [f"{p.get('index', i)}: {p['name']}" for i, p in enumerate(named)]
    return "[Potions] " + " | ".join(out) + "  (descriptions: potions endpoint)"


def _powers_str(powers: list[dict[str, Any]] | None) -> str:
    if not powers:
        return ""
    return ", ".join(f"{p.get('name', '?')}({p.get('amount', '')})" for p in powers)


def _card_stats(card: dict[str, Any]) -> str:
    stats = card.get("stats") or {}
    if not stats:
        return ""
    return " [" + " ".join(f"{k}:{v}" for k, v in stats.items()) + "]"


def _clean_desc(text: str) -> str:
    """Strip BBCode and reduce SmartFormat expressions to [Var] placeholders.
    Ported from sts2-cli python/play.py desc()."""
    text = re.sub(r"\[/?[^\]]+\]", "", text)  # strip BBCode [tags]

    def smart_replace(m: re.Match) -> str:
        full = m.group(1)
        if full.startswith("IfUpgraded:show:"):
            parts = full[len("IfUpgraded:show:"):].split("|")
            return parts[1] if len(parts) > 1 else parts[0]  # non-upgraded default
        if full.startswith("IfUpgraded:"):
            parts = full[len("IfUpgraded:"):].split("|")
            return parts[1] if len(parts) > 1 else parts[0]
        if full.startswith("InCombat:"):
            return full[len("InCombat:"):].split("|")[0].lstrip("\n")
        if "energyIcons" in full:
            return f"[{full.split(':')[0]}] Energy"
        if "starIcons" in full:
            return f"[{full.split(':')[0]}] Stars"
        if ":plural:" in full:
            parts = full.split(":")
            plural_parts = ":".join(parts[2:]).split("|")
            second = plural_parts[1] if len(plural_parts) > 1 else plural_parts[0]
            return f"[{parts[0]}:{plural_parts[0]}|{second}]"
        if ":" in full and "|" in full:
            return ":".join(full.split(":")[1:]).split("|")[-1]  # false/last branch
        return f"[{full.split(':')[0]}]"

    for _ in range(3):  # up to 3 nesting levels
        text = re.sub(r"\{([^{}]+)\}", smart_replace, text)
    return text.strip()


def _resolve_vars(text: str, vars_dict: dict[str, Any] | None) -> str:
    """Substitute [Var] placeholders with values (case-insensitive).
    Ported from sts2-cli python/play.py resolve_template()."""
    lower_vars = {k.lower(): v for k, v in (vars_dict or {}).items()}

    def replacer(m: re.Match) -> str:
        key = m.group(1)
        if ":" in key and "|" in key:  # plural: [Cards:card|cards]
            var_name, plural_spec = key.split(":", 1)
            val = lower_vars.get(var_name.lower())
            if val is not None:
                forms = plural_spec.split("|")
                try:
                    return forms[0] if int(val) == 1 else (forms[1] if len(forms) > 1 else forms[0])
                except (TypeError, ValueError):
                    return forms[0]
            return f"[{key}]"
        kl = key.lower()
        val = lower_vars.get(kl)
        if val is not None:
            return str(val)
        if kl == "energyprefix":
            return ""
        return f"[{key}]"

    return re.sub(r"\[([^\]]+)\]", replacer, text)


def _fmt_desc(text: Any, vars_dict: dict[str, Any] | None = None) -> str:
    """Cleaned, var-resolved description as an inline ' — desc' suffix."""
    if not text:
        return ""
    resolved = _resolve_vars(_clean_desc(str(text)), vars_dict)
    return " — " + " ".join(resolved.split())


# ── decision serializers (each returns a list of lines) ───────────────────


def _combat(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state, energy=True)]

    powers = _powers_str(state.get("player_powers"))
    if powers:
        parts.append(f"[Player Powers] {powers}")

    relics = _relics_line(state)
    if relics:
        parts.append(relics)

    # Character mechanics
    if "orbs" in state:
        orb_strs = [
            f"{o.get('type', '?')}(passive:{o.get('passive')},evoke:{o.get('evoke')})"
            for o in state["orbs"]
        ]
        parts.append(f"[Orbs] {' | '.join(orb_strs)} | Slots: {state.get('orb_slots', '?')}")
    if "stars" in state:
        parts.append(f"[Stars] {state['stars']}")
    if "osty" in state:
        o = state["osty"]
        if o.get("alive"):
            parts.append(f"[Osty] HP {o.get('hp')}/{o.get('max_hp')} | Block {o.get('block', 0)}")
        else:
            parts.append("[Osty] dead")

    hand = state.get("hand") or []
    if hand:
        lines = []
        for c in hand:
            flags = []
            if not c.get("can_play", True):
                flags.append("UNPLAYABLE")
            tgt = c.get("target_type", "")
            if tgt == "AnyEnemy":
                flags.append("needs target")
            cost = c.get("cost", "?")
            if "star_cost" in c:
                cost = f"{cost}+{c['star_cost']}*"
            flag_str = f" ({'; '.join(flags)})" if flags else ""
            extras = ""
            if c.get("enchantment"):
                extras += f" <enchant: {c['enchantment']}>"
            if c.get("affliction"):
                extras += f" <affliction: {c['affliction']}>"
            dbt = c.get("damage_by_target")
            if dbt and len(dbt) > 1:
                per = ", ".join(
                    f"→{d['target_index']}:{d.get('total_damage', d.get('damage', '?'))}" for d in dbt
                )
                extras += f" [dmg {per}]"
            lines.append(
                f"  {c.get('index')}: {c.get('name', '?')} (cost:{cost}, {c.get('type', '')})"
                f"{_card_stats(c)}{flag_str}{extras}{_fmt_desc(c.get('description'), c.get('stats'))}"
            )
        parts.append("[Hand]\n" + "\n".join(lines))

    parts.append(
        f"[Piles] Draw: {state.get('draw_pile_count', '?')} | "
        f"Discard: {state.get('discard_pile_count', '?')}  (contents: piles endpoint)"
    )

    enemies = state.get("enemies") or []
    if enemies:
        lines = []
        for e in enemies:
            intents = e.get("intents") or []
            intent_strs = []
            for it in intents:
                s = it.get("type", "?")
                if "damage" in it:
                    if it.get("hits", 1) > 1:
                        s += f" {it['damage']}x{it['hits']}={it.get('total_damage', '?')}"
                    else:
                        s += f" {it['damage']}"
                intent_strs.append(s)
            intent = "; ".join(intent_strs) if intent_strs else ("Attack?" if e.get("intends_attack") else "Unknown")
            pw = _powers_str(e.get("powers"))
            pw_str = f" | {pw}" if pw else ""
            block = e.get("block", 0)
            block_str = f" Block:{block}" if block else ""
            lines.append(
                f"  {e.get('index')}: {e.get('name', '?')} HP:{e.get('hp')}/{e.get('max_hp')}"
                f"{block_str} | Intent: {intent}{pw_str}"
            )
        parts.append("[Enemies]\n" + "\n".join(lines))

    potions = _potions_line(state)
    if potions:
        parts.append(potions)

    parts.append(f"[Round] {state.get('round', '?')}")
    return parts


def _map_select(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state)]
    relics = _relics_line(state)
    if relics:
        parts.append(relics)
    potions = _potions_line(state)
    if potions:
        parts.append(potions)
    choices = state.get("choices") or []
    lines = [f"  (col:{c.get('col')}, row:{c.get('row')}): {c.get('type', '?')}" for c in choices]
    parts.append("[Map Choices]  (full act map: map endpoint)\n" + "\n".join(lines))
    return parts


def _event(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state)]
    parts.append(f"[Event] {state.get('event_name', '?')}")
    desc = state.get("description")
    if desc:
        parts.append(" ".join(str(desc).split()))
    options = state.get("options") or []
    lines = []
    for o in options:
        locked = " (LOCKED)" if o.get("is_locked") else ""
        vars_ = o.get("vars") or {}
        var_str = " [" + " ".join(f"{k}:{v}" for k, v in vars_.items()) + "]" if vars_ else ""
        lines.append(
            f"  {o.get('index')}: {o.get('title', '?')}{locked}{var_str}{_fmt_desc(o.get('description'), o.get('vars'))}"
        )
    parts.append("[Options]\n" + "\n".join(lines))
    return parts


def _upgrade_note(card: dict[str, Any]) -> str:
    au = card.get("after_upgrade")
    if not isinstance(au, dict):
        return ""
    bits = []
    if au.get("cost") is not None and au["cost"] != card.get("cost"):
        bits.append(f"cost:{au['cost']}")
    desc = _fmt_desc(au.get("description"), au.get("stats"))
    if desc:
        bits.append(desc[3:])  # drop the " — " prefix
    if au.get("added_keywords"):
        bits.append("+" + ",".join(au["added_keywords"]))
    if au.get("removed_keywords"):
        bits.append("-" + ",".join(au["removed_keywords"]))
    return f" <if upgraded: {'; '.join(bits)}>" if bits else ""


def _card_lines(cards: list[dict[str, Any]]) -> str:
    lines = []
    for c in cards:
        lines.append(
            f"  {c.get('index')}: {c.get('name', '?')} (cost:{c.get('cost', '?')}, "
            f"{c.get('type', '')}, {c.get('rarity', '')})"
            f"{_fmt_desc(c.get('description'), c.get('stats'))}{_upgrade_note(c)}"
        )
    return "\n".join(lines)


def _card_reward(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state)]
    gold = state.get("gold_earned")
    if gold:
        parts.append(f"[Gold earned] {gold}")
    skip = "may skip" if state.get("can_skip", True) else "MUST pick"
    parts.append(f"[Card Reward — {skip}]\n" + _card_lines(state.get("cards") or []))
    return parts


def _card_select(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state)]
    lo, hi = state.get("min_select", 1), state.get("max_select", 1)
    parts.append(f"[Card Select — pick {lo}" + (f"-{hi}" if hi != lo else "") + "]\n" + _card_lines(state.get("cards") or []))
    return parts


def _bundle_select(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state)]
    lines = []
    for i, b in enumerate(state.get("bundles") or []):
        cards = b.get("cards") or []
        card_strs = [f"{c.get('name', '?')}{_fmt_desc(c.get('description'), c.get('stats'))}" for c in cards]
        lines.append(f"  {i}: " + " // ".join(card_strs))
    parts.append("[Bundle Select]\n" + "\n".join(lines))
    return parts


def _rest_site(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state)]
    lines = []
    for o in state.get("options") or []:
        dis = "" if o.get("is_enabled", True) else " (DISABLED)"
        lines.append(f"  {o.get('index')}: {o.get('option_id', o.get('name', '?'))}{dis}")
    parts.append("[Rest Site]\n" + "\n".join(lines))
    return parts


def _shop(state: dict[str, Any]) -> list[str]:
    parts = [_context_line(state), _player_line(state)]
    cards = state.get("cards") or []
    if cards:
        lines = []
        for c in cards:
            if not c.get("is_stocked", True):
                continue
            sale = " ON SALE" if c.get("on_sale") else ""
            lines.append(
                f"  {c.get('index')}: {c.get('name', '?')} ({c.get('type', '')}, {c.get('rarity', '')}) "
                f"{c.get('cost', '?')}g{sale}{_fmt_desc(c.get('description'), c.get('stats'))}{_upgrade_note(c)}"
            )
        parts.append("[Shop Cards]\n" + "\n".join(lines))
    relics = state.get("relics") or []
    if relics:
        lines = [
            f"  {r.get('index')}: {r.get('name', '?')} {r.get('cost', '?')}g{_fmt_desc(r.get('description'))}"
            for r in relics if r.get("is_stocked", True)
        ]
        parts.append("[Shop Relics]\n" + "\n".join(lines))
    potions = state.get("potions") or []
    if potions:
        lines = [
            f"  {p.get('index')}: {p.get('name', '?')} {p.get('cost', '?')}g{_fmt_desc(p.get('description'))}"
            for p in potions if p.get("is_stocked", True)
        ]
        parts.append("[Shop Potions]\n" + "\n".join(lines))
    removal = state.get("card_removal_cost")
    if removal is not None:
        parts.append(f"[Card Removal] {removal}g")
    return parts


def _game_over(state: dict[str, Any]) -> list[str]:
    p = state.get("player") or {}
    outcome = "VICTORY" if state.get("victory") else "DEFEAT"
    return [
        f"[Game Over] {outcome} | Act {state.get('act', '?')}, Floor {state.get('floor', '?')} | "
        f"HP {p.get('hp', '?')}/{p.get('max_hp', '?')} | Gold {p.get('gold', 0)}"
    ]


# ── detail views (for the service's detail endpoints) ─────────────────────


def render_deck(player: dict[str, Any]) -> str:
    """Full deck with upgrade status and descriptions, from a player summary."""
    deck = player.get("deck") or []
    lines = []
    for c in deck:
        up = "+" if c.get("upgraded") else ""
        kw = f" [{','.join(c['keywords'])}]" if c.get("keywords") else ""
        extras = ""
        if c.get("enchantment"):
            extras += f" <enchant: {c['enchantment']}>"
        if c.get("affliction"):
            extras += f" <affliction: {c['affliction']}>"
        lines.append(
            f"{c.get('name', '?')}{up} (cost:{c.get('cost', '?')}, {c.get('type', '')})"
            f"{kw}{extras}{_fmt_desc(c.get('description'), c.get('stats'))}{_upgrade_note(c)}"
        )
    return f"[Deck — {len(deck)} cards]\n" + "\n".join(lines)


def render_piles(piles: dict[str, Any]) -> str:
    """Draw/discard/exhaust contents. Draw pile is SORTED by name — order is
    hidden information (contents are player-visible, order is not)."""
    out = []
    for key, label in (("draw", "Draw (sorted — order hidden)"), ("discard", "Discard"), ("exhaust", "Exhaust")):
        cards = list(piles.get(key) or [])
        if key == "draw":
            cards.sort(key=lambda c: (c.get("name", ""), c.get("upgraded", False)))
        strs = [f"{c.get('name', '?')}{'+' if c.get('upgraded') else ''}" for c in cards]
        out.append(f"[{label} — {len(cards)}] " + (", ".join(strs) if strs else "(empty)"))
    return "\n".join(out)


def render_relics(player: dict[str, Any]) -> str:
    relics = player.get("relics") or []
    lines = []
    for r in relics:
        vars_ = r.get("vars") or {}
        var_str = " [" + " ".join(f"{k}:{v}" for k, v in vars_.items()) + "]" if vars_ else ""
        lines.append(f"{r.get('name', '?')}{var_str}{_fmt_desc(r.get('description'), r.get('vars'))}")
    return f"[Relics — {len(relics)}]\n" + ("\n".join(lines) if lines else "(none)")


def render_potions(player: dict[str, Any]) -> str:
    potions = [p for p in (player.get("potions") or []) if p and p.get("name")]
    lines = [
        f"Slot {p.get('index', i)}: {p['name']} (target: {p.get('target_type', 'None')})"
        f"{_fmt_desc(p.get('description'), p.get('vars'))}"
        for i, p in enumerate(potions)
    ]
    return "[Potions]\n" + ("\n".join(lines) if lines else "(none)")


def render_map(map_state: dict[str, Any]) -> str:
    """Full act map: nodes by row with edges, boss identity, current position."""
    parts = []
    ctx = map_state.get("context") or {}
    boss = map_state.get("boss") or {}
    parts.append(
        f"[Act {ctx.get('act', '?')} Map — boss: {boss.get('name', '?')}]"
    )
    cur = map_state.get("current_coord")
    if cur:
        parts.append(f"Current position: (col:{cur.get('col')}, row:{cur.get('row')})")
    for row_nodes in map_state.get("rows") or []:
        row_idx = row_nodes[0].get("row", "?") if row_nodes else "?"
        node_strs = []
        for n in row_nodes:
            marks = ""
            if n.get("current"):
                marks += "*HERE*"
            elif n.get("visited"):
                marks += "(visited)"
            children = n.get("children") or []
            edges = ",".join(f"({c.get('col')},{c.get('row')})" for c in children)
            edge_str = f" →{edges}" if edges else ""
            node_strs.append(f"col{n.get('col')}:{n.get('type', '?')}{marks}{edge_str}")
        parts.append(f"row {row_idx}: " + " | ".join(node_strs))
    parts.append(f"boss node: (col:{boss.get('col')}, row:{boss.get('row')})")
    return "\n".join(parts)
