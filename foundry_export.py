"""
Converts a TargetMonsterOutput into 5etools bestiary JSON format,
importable into Foundry VTT via the Plutonium module.

Output shape:  {"monster": [<entry>]}

Usage:
    from foundry_export import to_5etools_json

    json_str = to_5etools_json(monster)          # string ready to write to file
    entry    = to_5etools_entry(monster)         # dict (single monster object)
"""

import json
import re
from typing import Optional

from schemas import TargetMonsterOutput

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_SIZE = {
    "Tiny": "T", "Small": "S", "Medium": "M",
    "Large": "L", "Huge": "H", "Gargantuan": "G",
}

_ALIGN_WORDS = {
    "lawful": "L", "chaotic": "C",
    "neutral": "N", "good": "G", "evil": "E",
    "unaligned": "U", "any": "A",
}

_SAVE_ABBR = {
    "str": "str", "dex": "dex", "con": "con",
    "int": "int", "wis": "wis", "cha": "cha",
    "strength": "str", "dexterity": "dex", "constitution": "con",
    "intelligence": "int", "wisdom": "wis", "charisma": "cha",
}

_SKILL_NAMES = {
    "acrobatics", "animal handling", "arcana", "athletics", "deception",
    "history", "insight", "intimidation", "investigation", "medicine",
    "nature", "perception", "performance", "persuasion", "religion",
    "sleight of hand", "stealth", "survival",
}


# ---------------------------------------------------------------------------
# Field converters
# ---------------------------------------------------------------------------

def _encode_alignment(text: str) -> list:
    lower = text.lower().strip()
    if not lower or lower in ("—", "-", ""):
        return ["U"]
    tokens = [_ALIGN_WORDS.get(w) for w in lower.split() if _ALIGN_WORDS.get(w)]
    # Deduplicate while preserving order
    seen, result = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result or ["U"]


def _parse_bonus_list(items: list[str], abbr_map: dict) -> dict:
    """Parse ['Str +10', 'Con +9'] or ['Perception +5'] into 5etools dicts."""
    result = {}
    for item in items:
        # Match "Word +N" or "Word Word +N"
        m = re.match(r"^([\w\s]+?)\s*([+-]\d+)$", item.strip())
        if not m:
            continue
        name  = m.group(1).strip().lower()
        bonus = m.group(2)
        key   = abbr_map.get(name) or name.replace(" ", " ")
        if key:
            result[key] = bonus
    return result


def _passive_perception(monster: TargetMonsterOutput) -> int:
    # Try to extract from senses list first
    for s in monster.senses:
        m = re.search(r"passive\s+perception\s+(\d+)", s, re.IGNORECASE)
        if m:
            return int(m.group(1))
    # Derive: 10 + WIS mod + proficiency if Perception in skills
    wis_mod = (monster.abilities.wisdom - 10) // 2
    perc_bonus = 0
    for sk in monster.skills:
        m = re.match(r"perception\s*([+-]\d+)", sk.strip(), re.IGNORECASE)
        if m:
            perc_bonus = int(m.group(1)) - wis_mod
            break
    return 10 + wis_mod + perc_bonus


def _senses_without_passive(senses: list[str]) -> list[str]:
    return [s for s in senses if "passive" not in s.lower()]


def _hp_formula(monster: TargetMonsterOutput) -> str:
    """Best-effort HP formula from avg HP and CON modifier. Returns '' if uncertain."""
    avg = monster.hit_points
    con_mod = (monster.abilities.constitution - 10) // 2
    size = monster.size

    die_by_size = {
        "Tiny": 4, "Small": 6, "Medium": 8,
        "Large": 10, "Huge": 12, "Gargantuan": 12,
    }
    die = die_by_size.get(size, 8)
    avg_die = (die + 1) / 2  # e.g. d10 avg = 5.5

    if avg_die + con_mod <= 0:
        return ""
    num_dice = round(avg / (avg_die + con_mod))
    if num_dice <= 0:
        return ""

    con_total = con_mod * num_dice
    if con_total > 0:
        return f"{num_dice}d{die} + {con_total}"
    elif con_total < 0:
        return f"{num_dice}d{die} - {abs(con_total)}"
    return f"{num_dice}d{die}"


def _encode_spellcasting_trait(monster: TargetMonsterOutput) -> Optional[dict]:
    """Convert SpellcastingBlock into a 5etools-style Spellcasting trait entry."""
    sc = monster.spellcasting
    if not sc:
        return None

    mod_names = {
        "strength": "Str", "dexterity": "Dex", "constitution": "Con",
        "intelligence": "Int", "wisdom": "Wis", "charisma": "Cha",
    }
    ability_short = mod_names.get(sc.ability.lower(), sc.ability)

    header = (
        f"{monster.name} is a spellcaster. Its spellcasting ability is "
        f"{sc.ability} (spell save DC {sc.spell_save_dc}, "
        f"+{sc.spell_attack_bonus} to hit with spell attacks)."
    )
    lines = [header]

    if sc.at_will:
        lines.append(f"At will: {', '.join(sc.at_will)}")

    if sc.innate_spells:
        lines.append(f"Innate: {', '.join(sc.innate_spells)}")

    if sc.spell_slots:
        ordinals = {1:"1st",2:"2nd",3:"3rd",4:"4th",5:"5th",
                    6:"6th",7:"7th",8:"8th",9:"9th"}
        for level_str, spells in sc.spell_slots.items():
            # Accept both "1st" strings and integer keys
            try:
                lvl_label = ordinals.get(int(level_str), f"{level_str}")
            except (ValueError, TypeError):
                lvl_label = level_str
            lines.append(f"{lvl_label} level: {', '.join(spells)}")

    return {"name": "Spellcasting", "entries": lines}


def _encode_traits(monster: TargetMonsterOutput) -> list[dict]:
    traits = []
    for t in monster.traits:
        traits.append({"name": t.get("name", ""), "entries": [t.get("description", "")]})

    # Spellcasting as a proper trait block
    sc_trait = _encode_spellcasting_trait(monster)
    if sc_trait:
        traits.append(sc_trait)

    # Tactical rules as a GM-facing trait
    if monster.tactical_rules or monster.tactical_summary:
        rule_lines = [monster.tactical_summary] if monster.tactical_summary else []
        for r in sorted(monster.tactical_rules, key=lambda x: x.priority):
            rule_lines.append(f"{r.priority}. IF {r.condition} → {r.action}")
        traits.append({
            "name": "Tactical Behavior (GM)",
            "entries": rule_lines,
        })

    return traits


def _encode_actions(actions) -> list[dict]:
    return [
        {"name": a.name, "entries": [a.description]}
        for a in actions
    ]


def _encode_legendary(legendary) -> Optional[list[dict]]:
    if not legendary:
        return None
    return [
        {"name": la.get("name", ""), "entries": [la.get("description", "")]}
        for la in legendary
    ]


def _encode_speed(speed: dict) -> dict:
    # 5etools speed values are integers (feet), keys match directly
    return {k: v for k, v in speed.items() if v}


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def to_5etools_entry(monster: TargetMonsterOutput, source: str = "Homebrew") -> dict:
    entry = {
        "name":   monster.name,
        "source": source,
        "size":   [_SIZE.get(monster.size, "M")],
        "type":   monster.monster_type.lower(),
        "alignment": _encode_alignment(monster.alignment),
        "ac": [{"ac": monster.armor_class, "from": ["natural armor"]}],
        "hp": {
            "average": monster.hit_points,
            "formula": _hp_formula(monster),
        },
        "speed": _encode_speed(monster.speed),
        "str": monster.abilities.strength,
        "dex": monster.abilities.dexterity,
        "con": monster.abilities.constitution,
        "int": monster.abilities.intelligence,
        "wis": monster.abilities.wisdom,
        "cha": monster.abilities.charisma,
        "passive": _passive_perception(monster),
        "cr": monster.challenge_rating,
        "trait":  _encode_traits(monster),
        "action": _encode_actions(monster.actions),
    }

    # Optional fields — only include if non-empty
    saves = _parse_bonus_list(monster.saving_throws, _SAVE_ABBR)
    if saves:
        entry["save"] = saves

    skills = _parse_bonus_list(monster.skills, {s.split()[-1].lower(): s.replace(" ", " ").lower()
                                                  for s in _SKILL_NAMES})
    # Remap skill keys to 5etools format (lowercase, spaces)
    skill_map = {}
    for item in monster.skills:
        m = re.match(r"^([\w\s]+?)\s*([+-]\d+)$", item.strip())
        if m:
            skill_map[m.group(1).strip().lower()] = m.group(2)
    if skill_map:
        entry["skill"] = skill_map

    senses = _senses_without_passive(monster.senses)
    if senses:
        entry["senses"] = senses

    if monster.languages:
        entry["languages"] = monster.languages

    if monster.damage_immunities:
        entry["immune"] = monster.damage_immunities

    if monster.damage_resistances:
        entry["resist"] = monster.damage_resistances

    if monster.condition_immunities:
        entry["conditionImmune"] = monster.condition_immunities

    legendary = _encode_legendary(monster.legendary_actions)
    if legendary:
        entry["legendary"] = legendary

    if monster.equipment:
        entry["gear"] = monster.equipment

    return entry


def to_5etools_json(
    monster: TargetMonsterOutput,
    source: str = "Homebrew",
    token_b64: Optional[str] = None,
    personality=None,   # Optional[NPCPersonality] — avoid circular import
    token_path: Optional[str] = None,
) -> str:
    """
    Returns a JSON string in 5etools bestiary format ready to save as a .json file.

    If token_path is provided (relative Foundry path like "monsters/infernocyte-shark.webp"),
    it is embedded in foundryImg and foundryPrototypeToken so Plutonium can resolve it.

    If token_b64 is provided but token_path is not, falls back to data URI
    (works for API import but not Plutonium file import).
    """
    entry = to_5etools_entry(monster, source)

    if token_path:
        entry["foundryImg"] = token_path
        entry["foundryPrototypeToken"] = {
            "texture": {"src": token_path},
            "name": monster.name,
        }
    elif token_b64:
        # Fallback: data URI (works for API, not for Plutonium file import)
        data_uri = f"data:image/webp;base64,{token_b64}"
        entry["foundryImg"] = data_uri
        entry["foundryPrototypeToken"] = {
            "texture": {"src": data_uri},
            "name": monster.name,
        }

    if personality is not None:
        entry["foundrySystem"] = {
            "details": {
                "biography": {
                    "value": personality.to_biography_html(),
                    "public": "",
                }
            }
        }

    # Embed full schema so the file can be reloaded without loss
    entry["_schema"] = monster.model_dump()

    payload = {
        "_meta": {
            "sources": [
                {
                    "json":         source,
                    "abbreviation": "HB",
                    "full":         "Homebrew Content",
                    "url":          "",
                    "authors":      [],
                    "convertedBy":  [],
                    "version":      "1.0.0",
                    "color":        "",
                }
            ],
            "dateAdded":          0,
            "dateLastModified":   0,
        },
        "monster": [entry],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
