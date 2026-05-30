"""
Converts TargetMonsterOutput to Foundry VTT native dnd5e system actor JSON.
No Plutonium required — Actor.create() accepts this directly.

Targets dnd5e system v3.x (current as of 2025).
"""

import re
from typing import Optional
from schemas import TargetMonsterOutput

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_SIZE_MAP = {
    "Tiny": "tiny", "Small": "sm", "Medium": "med",
    "Large": "lg",  "Huge": "huge", "Gargantuan": "grg",
}
_TOKEN_SIZE = {
    "tiny": 0.5, "sm": 1, "med": 1, "lg": 2, "huge": 3, "grg": 4,
}

_CR_TO_FLOAT = {
    "0": 0, "1/8": 0.125, "1/4": 0.25, "1/2": 0.5,
    **{str(i): float(i) for i in range(1, 31)},
}

_ABILITY_KEYS = {
    "strength": "str", "dexterity": "dex", "constitution": "con",
    "intelligence": "int", "wisdom": "wis", "charisma": "cha",
}
_SKILL_MAP = {
    "acrobatics": "acr", "animal handling": "ani", "arcana": "arc",
    "athletics": "ath", "deception": "dec", "history": "his",
    "insight": "ins", "intimidation": "itm", "investigation": "inv",
    "medicine": "med", "nature": "nat", "perception": "prc",
    "performance": "prf", "persuasion": "per", "religion": "rel",
    "sleight of hand": "slt", "stealth": "ste", "survival": "sur",
}
_SAVE_ABBR = {
    "str": "str", "dex": "dex", "con": "con",
    "int": "int", "wis": "wis", "cha": "cha",
    "strength": "str", "dexterity": "dex", "constitution": "con",
    "intelligence": "int", "wisdom": "wis", "charisma": "cha",
}
_DAMAGE_TYPES = {
    "bludgeoning", "piercing", "slashing", "fire", "cold", "lightning",
    "thunder", "acid", "poison", "necrotic", "radiant", "force", "psychic",
}
_CONDITION_IDS = {
    "blinded": "blinded", "charmed": "charmed", "deafened": "deafened",
    "exhaustion": "exhaustion", "frightened": "frightened", "grappled": "grappled",
    "incapacitated": "incapacitated", "invisible": "invisible", "paralyzed": "paralyzed",
    "petrified": "petrified", "poisoned": "poisoned", "prone": "prone",
    "restrained": "restrained", "stunned": "stunned", "unconscious": "unconscious",
}

_BONUS_RE = re.compile(r"^([\w\s]+?)\s*([+-]\d+)$")
_SENSE_RE  = re.compile(r"(darkvision|blindsight|tremorsense|truesight)\s+(\d+)", re.I)
_ATK_RE    = re.compile(r"[+-](\d+)\s+to\s+hit", re.I)
_RANGE_RE  = re.compile(r"reach\s+(\d+)\s*ft", re.I)
_RRANGE_RE = re.compile(r"range\s+(\d+)/(\d+)\s*ft", re.I)
_DC_RE     = re.compile(r"DC\s*(\d+)\s+(\w+)\s+saving", re.I)


def _mod(score: int) -> int:
    return (score - 10) // 2


def _parse_bonus(s: str) -> Optional[tuple[str, int]]:
    m = _BONUS_RE.match(s.strip())
    if m:
        return m.group(1).strip().lower(), int(m.group(2))
    return None


def _parse_saves(saving_throws: list[str]) -> dict:
    out = {}
    for st in saving_throws:
        parsed = _parse_bonus(st)
        if parsed:
            name, _ = parsed
            key = _SAVE_ABBR.get(name)
            if key:
                out[key] = {"value": 1}
    return out


def _parse_skills(skills: list[str]) -> dict:
    out = {}
    for sk in skills:
        parsed = _parse_bonus(sk)
        if parsed:
            name, bonus = parsed
            key = _SKILL_MAP.get(name)
            if key:
                out[key] = {"value": 1, "ability": key[:3]}
    return out


def _parse_senses(senses: list[str]) -> dict:
    result = {"darkvision": 0, "blindsight": 0, "tremorsense": 0, "truesight": 0,
              "units": "ft", "special": ""}
    passive = 10
    for s in senses:
        for m in _SENSE_RE.finditer(s):
            result[m.group(1).lower()] = int(m.group(2))
        pm = re.search(r"passive\s+perception\s+(\d+)", s, re.I)
        if pm:
            passive = int(pm.group(1))
    return result, passive


def _parse_movement(speed: dict) -> dict:
    return {
        "burrow": speed.get("burrow", 0),
        "climb":  speed.get("climb",  0),
        "fly":    speed.get("fly",    0),
        "swim":   speed.get("swim",   0),
        "walk":   speed.get("walk",   0),
        "units":  "ft",
        "hover":  False,
    }


def _action_type(action_type_str: str) -> str:
    s = action_type_str.lower()
    if "melee" in s:   return "mwak"
    if "ranged" in s:  return "rwak"
    return "other"


def _parse_damage_parts(damage_dice: Optional[str], description: str) -> list:
    if not damage_dice:
        return []
    parts = []
    dice_re = re.compile(r"\d+d\d+(?:\s*[+-]\s*\d+)?")
    type_re = re.compile(
        r"(" + "|".join(_DAMAGE_TYPES) + r")\s+damage", re.I
    )
    dice_exprs = dice_re.findall(damage_dice)
    damage_types = type_re.findall(description)
    for i, expr in enumerate(dice_exprs):
        dtype = damage_types[i].lower() if i < len(damage_types) else ""
        parts.append([expr.replace(" ", ""), dtype])
    return parts if parts else [[damage_dice, ""]]


def _make_feat_item(name: str, description: str) -> dict:
    return {
        "name": name,
        "type": "feat",
        "system": {
            "description": {"value": f"<p>{description}</p>", "chat": ""},
            "source": {"custom": "Homebrew"},
            "activation": {"type": "", "cost": None, "condition": ""},
            "type": {"value": "monster", "subtype": ""},
        },
    }


def _make_action_item(action) -> dict:
    atype = _action_type(action.action_type)
    is_weapon = atype in ("mwak", "rwak")

    if not is_weapon:
        item = _make_feat_item(action.name, action.description)
        item["system"]["activation"] = {"type": "action", "cost": 1, "condition": ""}
        # Check for save-based actions
        dc_m = _DC_RE.search(action.description or "")
        if dc_m:
            ability_map = {"str": "str", "dex": "dex", "con": "con",
                           "int": "int", "wis": "wis", "cha": "cha",
                           "strength": "str", "dexterity": "dex"}
            ability = ability_map.get(dc_m.group(2).lower(), "dex")
            item["system"]["actionType"] = "save"
            item["system"]["save"] = {"ability": ability, "dc": int(dc_m.group(1)), "scaling": "flat"}
            if action.damage_dice:
                item["system"]["damage"] = {"parts": _parse_damage_parts(action.damage_dice, action.description)}
        return item

    # Weapon attack
    atk_m    = _ATK_RE.search(action.description or "")
    atk_bonus = int(atk_m.group(1)) if atk_m else 0
    range_m  = _RANGE_RE.search(action.description or "")
    rrange_m = _RRANGE_RE.search(action.description or "")

    reach = int(range_m.group(1)) if range_m else 5
    r_short = int(rrange_m.group(1)) if rrange_m else (60 if atype == "rwak" else None)
    r_long  = int(rrange_m.group(2)) if rrange_m else None

    return {
        "name": action.name,
        "type": "weapon",
        "system": {
            "description": {"value": f"<p>{action.description}</p>", "chat": ""},
            "source": {"custom": "Homebrew"},
            "quantity": 1, "weight": {"value": 0},
            "equipped": True, "identified": True, "rarity": "",
            "activation": {"type": "action", "cost": 1},
            "target": {"value": 1, "type": "creature"},
            "range": {
                "value": r_short or reach,
                "long": r_long,
                "units": "ft",
            },
            "actionType": atype,
            "attackBonus": str(atk_bonus),
            "damage": {
                "parts": _parse_damage_parts(action.damage_dice, action.description),
                "versatile": "",
            },
            "save": {"ability": "", "dc": None, "scaling": "spell"},
            "type": {"value": "natural", "baseItem": ""},
            "proficient": None,
        },
    }


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def to_dnd5e_actor(
    monster: TargetMonsterOutput,
    token_b64: Optional[str] = None,
    personality=None,
) -> dict:
    """Convert TargetMonsterOutput to a Foundry dnd5e v3 actor dict."""

    size_key  = _SIZE_MAP.get(monster.size, "med")
    cr_val    = _CR_TO_FLOAT.get(monster.challenge_rating, 0)
    ab        = monster.abilities
    senses, passive = _parse_senses(monster.senses)

    # Build items list
    items = []
    # Traits as feats
    for t in monster.traits:
        items.append(_make_feat_item(t.get("name", ""), t.get("description", "")))
    # Actions
    for a in monster.actions:
        if a.name.lower() == "multiattack":
            items.append(_make_feat_item(a.name, a.description))
        else:
            items.append(_make_action_item(a))
    # Legendary actions
    for la in (monster.legendary_actions or []):
        items.append(_make_feat_item(la.get("name", ""), la.get("description", "")))

    # Biography
    bio_html = ""
    if personality is not None:
        bio_html = personality.to_biography_html()
    elif monster.tactical_summary:
        bio_html = f"<p><strong>Tactics:</strong> {monster.tactical_summary}</p>"

    token_src = (
        f"data:image/webp;base64,{token_b64}" if token_b64
        else "icons/svg/mystery-man.svg"
    )

    actor = {
        "name":   monster.name,
        "type":   "npc",
        "img":    token_src,
        "system": {
            "abilities": {
                key: {"value": getattr(ab, full), "proficient": 0,
                      "bonuses": {"check": "", "save": ""}}
                for full, key in _ABILITY_KEYS.items()
            },
            "attributes": {
                "ac":  {"flat": monster.armor_class, "calc": "flat", "formula": ""},
                "hp":  {"value": monster.hit_points, "min": 0,
                        "max": monster.hit_points, "temp": 0, "tempmax": 0, "formula": ""},
                "init": {"ability": "dex", "bonus": ""},
                "movement": _parse_movement(monster.speed),
                "senses":   {**senses, "special": ""},
            },
            "details": {
                "biography": {"value": bio_html, "public": ""},
                "alignment":  monster.alignment,
                "race":       "",
                "type": {
                    "value":   monster.monster_type.lower(),
                    "subtype": "",
                    "swarm":   "",
                    "custom":  "",
                },
                "cr":          cr_val,
                "spellLevel":  0,
                "source":      {"custom": "Homebrew"},
            },
            "traits": {
                "size": size_key,
                "di":   {"value": [d.lower() for d in monster.damage_immunities   if d.lower() in _DAMAGE_TYPES], "bypasses": [], "custom": ""},
                "dr":   {"value": [d.lower() for d in monster.damage_resistances  if d.lower() in _DAMAGE_TYPES], "bypasses": [], "custom": ""},
                "dv":   {"value": [], "bypasses": [], "custom": ""},
                "ci":   {"value": [c.lower() for c in monster.condition_immunities if c.lower() in _CONDITION_IDS], "bypasses": [], "custom": ""},
                "languages": {"value": [], "custom": ", ".join(monster.languages)},
            },
            "skills": _parse_skills(monster.skills),
            "saves":  _parse_saves(monster.saving_throws),
            "resources": {
                "legact": {"value": 3 if monster.legendary_actions else 0, "max": 3 if monster.legendary_actions else 0},
            },
        },
        "items": items,
        "prototypeToken": {
            "name":        monster.name,
            "displayName": 20,
            "actorLink":   False,
            "width":       _TOKEN_SIZE.get(size_key, 1),
            "height":      _TOKEN_SIZE.get(size_key, 1),
            "texture":     {"src": token_src, "scaleX": 1, "scaleY": 1},
            "disposition": -1,
            "bar1":        {"attribute": "attributes.hp"},
            "bar2":        {"attribute": ""},
            "sight":       {"enabled": True, "range": senses.get("darkvision", 0) or 30},
        },
        "folder": None,
        "flags":  {},
    }

    return actor
