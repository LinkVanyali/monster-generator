"""
Generator: retrieves reference context, assembles the generation prompt,
and calls Claude Sonnet to produce a structured monster stat block.

Usage:
    from generator import MonsterGenerator

    gen = MonsterGenerator()
    monster = gen.generate("A cybernetic laser-shark that lives in volcanic magma")
    print(monster.model_dump_json(indent=2))

    # Stateful mutation:
    monster2 = gen.mutate(monster, "Make it more aggressive and raise the CR to 10")
"""

import json
import warnings
import os
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import anthropic

from intent_parser import parse_intent, IntentResult
from retriever import HybridRetriever, load_reranker, load_embeddings
from schemas import TargetMonsterOutput

MODEL = "claude-sonnet-4-6"
CHROMA_DIR = "./chroma_db"

# ---------------------------------------------------------------------------
# CR mathematical anchors — derived from DMG Table 9-1
# ---------------------------------------------------------------------------

CR_ANCHORS = {
    "0":    dict(hp="1–6",     ac=13, prof=2, atk=3,  dpr="0–1"),
    "1/8":  dict(hp="7–35",    ac=13, prof=2, atk=3,  dpr="2–3"),
    "1/4":  dict(hp="36–49",   ac=13, prof=2, atk=3,  dpr="4–5"),
    "1/2":  dict(hp="50–70",   ac=13, prof=2, atk=3,  dpr="6–8"),
    "1":    dict(hp="71–85",   ac=13, prof=2, atk=3,  dpr="9–14"),
    "2":    dict(hp="86–100",  ac=13, prof=2, atk=3,  dpr="15–20"),
    "3":    dict(hp="101–115", ac=13, prof=2, atk=4,  dpr="21–26"),
    "4":    dict(hp="116–130", ac=14, prof=2, atk=5,  dpr="27–32"),
    "5":    dict(hp="131–145", ac=15, prof=3, atk=6,  dpr="33–38"),
    "6":    dict(hp="146–160", ac=15, prof=3, atk=6,  dpr="39–44"),
    "7":    dict(hp="161–175", ac=15, prof=3, atk=6,  dpr="45–50"),
    "8":    dict(hp="176–190", ac=16, prof=3, atk=7,  dpr="51–56"),
    "9":    dict(hp="191–205", ac=16, prof=3, atk=7,  dpr="57–62"),
    "10":   dict(hp="206–220", ac=17, prof=4, atk=7,  dpr="63–68"),
    "11":   dict(hp="221–235", ac=17, prof=4, atk=8,  dpr="69–74"),
    "12":   dict(hp="236–250", ac=17, prof=4, atk=8,  dpr="75–80"),
    "13":   dict(hp="251–265", ac=18, prof=5, atk=8,  dpr="81–86"),
    "14":   dict(hp="266–280", ac=18, prof=5, atk=8,  dpr="87–92"),
    "15":   dict(hp="281–295", ac=18, prof=5, atk=8,  dpr="93–98"),
    "16":   dict(hp="296–310", ac=18, prof=5, atk=9,  dpr="99–104"),
    "17":   dict(hp="311–325", ac=19, prof=6, atk=10, dpr="105–110"),
    "18":   dict(hp="326–340", ac=19, prof=6, atk=10, dpr="111–116"),
    "19":   dict(hp="341–355", ac=19, prof=6, atk=10, dpr="117–122"),
    "20":   dict(hp="356–400", ac=19, prof=6, atk=10, dpr="123–140"),
    "21":   dict(hp="401–445", ac=19, prof=7, atk=11, dpr="141–158"),
    "22":   dict(hp="446–490", ac=19, prof=7, atk=11, dpr="159–176"),
    "23":   dict(hp="491–535", ac=19, prof=7, atk=11, dpr="177–194"),
    "24":   dict(hp="536–580", ac=19, prof=7, atk=12, dpr="195–212"),
    "25":   dict(hp="581–625", ac=19, prof=8, atk=12, dpr="213–230"),
    "26":   dict(hp="626–670", ac=19, prof=8, atk=12, dpr="231–248"),
    "27":   dict(hp="671–715", ac=19, prof=8, atk=13, dpr="249–266"),
    "28":   dict(hp="716–760", ac=19, prof=8, atk=13, dpr="267–284"),
    "29":   dict(hp="761–805", ac=19, prof=8, atk=13, dpr="285–302"),
    "30":   dict(hp="806–850", ac=19, prof=9, atk=14, dpr="303–320"),
}

_CR_TO_FLOAT: dict[str, float] = {
    "0": 0, "1/8": 0.125, "1/4": 0.25, "1/2": 0.5,
    **{str(i): float(i) for i in range(1, 31)},
}


def _cr_proximity_filter(target_cr: str | None, delta: float = 3.0) -> dict | None:
    """
    Build a ChromaDB $in filter for CRs within ±delta of target_cr.
    Returns None if target_cr is unknown or unspecified (no filter applied).
    """
    if not target_cr:
        return None
    target_val = _CR_TO_FLOAT.get(str(target_cr).strip())
    if target_val is None:
        return None
    valid = [
        cr_str for cr_str, cr_val in _CR_TO_FLOAT.items()
        if abs(cr_val - target_val) <= delta
    ]
    return {"cr": {"$in": valid}} if valid else None


def _cr_anchor_block(target_cr: str | None) -> str:
    if target_cr and target_cr in CR_ANCHORS:
        a = CR_ANCHORS[target_cr]
        return (
            f"TARGET CR: {target_cr}\n"
            f"  HP range      : {a['hp']}\n"
            f"  Armour Class  : {a['ac']}\n"
            f"  Prof bonus    : +{a['prof']}\n"
            f"  Attack bonus  : +{a['atk']}\n"
            f"  Damage/round  : {a['dpr']}\n"
            "These values are MANDATORY. Do not deviate."
        )
    # No CR specified — provide the full table so Claude can pick appropriately
    rows = "\n".join(
        f"  CR {cr:<4} | HP {a['hp']:<10} | AC {a['ac']} | Prof +{a['prof']} | Atk +{a['atk']} | DPR {a['dpr']}"
        for cr, a in CR_ANCHORS.items()
    )
    return (
        "No target CR specified — choose one that fits the concept.\n"
        "Whatever CR you choose, its values below are MANDATORY:\n\n"
        + rows
    )


SYSTEM_PROMPT = """\
You are an expert D&D 5e mechanics engine and homebrew designer.
Your job is to generate completely original creature stat blocks that fulfil the user's \
thematic concept while being mechanically balanced and immediately usable at the table.

MECHANICAL RULES:
1. All numerical values must conform to the CR mandates provided.
2. Ability score modifiers must be correct: score 10-11 = +0, 12-13 = +1, 14-15 = +2, \
16-17 = +3, 18-19 = +4, 20-21 = +5, 22-23 = +6, 24-25 = +7, 26-27 = +8, 28-29 = +9, 30 = +10.
3. Attack bonuses = proficiency bonus + relevant ability modifier.
4. Saving throw DCs = 8 + proficiency bonus + relevant ability modifier.
5. Do NOT copy reference monsters — use them only as style and structure anchors.
6. Derive tactical rules logically from the creature's stats and abilities.
7. Keep all prose original — no direct quotes from source material.\
"""


def _format_reference_monsters(hits: list[dict], n: int = 3) -> str:
    blocks = []
    for hit in hits[:n]:
        try:
            monster = json.loads(hit["metadata"]["raw_json"])
            name = monster.get("name", "Unknown")
            cr   = monster.get("challenge_rating", "?")
            mtype = monster.get("monster_type", "?")
            traits  = "; ".join(t["name"] for t in monster.get("traits", [])[:4])
            actions = "; ".join(a["name"] for a in monster.get("actions", [])[:3])
            blocks.append(
                f"• {name} (CR {cr}, {mtype})\n"
                f"  Traits : {traits or '—'}\n"
                f"  Actions: {actions or '—'}"
            )
        except Exception:
            blocks.append(f"• {hit['metadata'].get('name', '?')}")
    return "\n".join(blocks) if blocks else "None retrieved."


def _format_spells(hits: list[dict], n: int = 8) -> str:
    lines = []
    for hit in hits[:n]:
        m = hit["metadata"]
        line = f"• {m['name']} (Level {m['level']}, {m['school']})"
        if m.get("damage"):
            line += f" — {m['damage']} damage"
        desc = m.get("description", "")[:120]
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines) if lines else "None retrieved."


def _format_items(hits: list[dict], n: int = 8) -> str:
    lines = []
    for hit in hits[:n]:
        m = hit["metadata"]
        line = f"• {m['name']} ({m['type']}"
        if m.get("rarity"):
            line += f", {m['rarity']}"
        if m.get("dmg"):
            line += f", {m['dmg']}"
        line += ")"
        desc = m.get("description", "")[:100]
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines) if lines else "None retrieved."


def _format_tactical_context(hits: list[dict], n: int = 3) -> str:
    blocks = []
    for hit in hits[:n]:
        title   = hit["metadata"].get("title", "Untitled")
        content = hit["content_text"][:1200]
        blocks.append(f"[{title}]\n{content}")
    return "\n\n---\n\n".join(blocks) if blocks else "None retrieved."


# ---------------------------------------------------------------------------
# Tool schema (drives structured JSON output)
# ---------------------------------------------------------------------------

GENERATE_TOOL = {
    "name": "generate_monster",
    "description": "Output a complete D&D 5e monster stat block.",
    "input_schema": {
        "type": "object",
        "required": [
            "name", "size", "monster_type", "alignment", "challenge_rating",
            "armor_class", "hit_points", "speed", "abilities",
            "saving_throws", "skills", "senses", "languages",
            "traits", "actions", "tactical_rules", "tactical_summary", "appearance",
        ],
        "properties": {
            "name":             {"type": "string"},
            "size":             {"type": "string", "enum": ["Tiny","Small","Medium","Large","Huge","Gargantuan"]},
            "monster_type":     {"type": "string"},
            "alignment":        {"type": "string"},
            "challenge_rating": {"type": "string"},
            "armor_class":      {"type": "integer"},
            "hit_points":       {"type": "integer"},
            "speed": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
                "description": "e.g. {\"walk\": 30, \"swim\": 40}",
            },
            "abilities": {
                "type": "object",
                "required": ["strength","dexterity","constitution","intelligence","wisdom","charisma"],
                "properties": {
                    "strength":     {"type": "integer", "minimum": 1, "maximum": 30},
                    "dexterity":    {"type": "integer", "minimum": 1, "maximum": 30},
                    "constitution": {"type": "integer", "minimum": 1, "maximum": 30},
                    "intelligence": {"type": "integer", "minimum": 1, "maximum": 30},
                    "wisdom":       {"type": "integer", "minimum": 1, "maximum": 30},
                    "charisma":     {"type": "integer", "minimum": 1, "maximum": 30},
                },
            },
            "saving_throws":       {"type": "array", "items": {"type": "string"}},
            "skills":              {"type": "array", "items": {"type": "string"}},
            "damage_immunities":   {"type": "array", "items": {"type": "string"}},
            "damage_resistances":  {"type": "array", "items": {"type": "string"}},
            "condition_immunities":{"type": "array", "items": {"type": "string"}},
            "senses":              {"type": "array", "items": {"type": "string"}},
            "languages":           {"type": "array", "items": {"type": "string"}},
            "traits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "description"],
                    "properties": {
                        "name":        {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "action_type", "description"],
                    "properties": {
                        "name":        {"type": "string"},
                        "action_type": {"type": "string"},
                        "description": {"type": "string"},
                        "damage_dice": {"type": "string"},
                    },
                },
            },
            "legendary_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "tactical_rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["condition", "action", "priority"],
                    "properties": {
                        "condition": {"type": "string"},
                        "action":    {"type": "string"},
                        "priority":  {"type": "integer"},
                    },
                },
            },
            "spellcasting": {
                "type": "object",
                "description": "Include only if the creature casts spells. Omit entirely otherwise.",
                "properties": {
                    "ability":            {"type": "string", "description": "e.g. 'Intelligence'"},
                    "spell_save_dc":      {"type": "integer"},
                    "spell_attack_bonus": {"type": "integer"},
                    "at_will":    {"type": "array", "items": {"type": "string"}},
                    "innate_spells": {"type": "array", "items": {"type": "string"},
                                     "description": "X/day spells, format 'Spell Name (3/day)'"},
                    "spell_slots": {
                        "type": "object",
                        "description": "Slot level → spell list. e.g. {'1st': ['Magic Missile', 'Shield'], '3rd': ['Fireball']}",
                        "additionalProperties": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["ability", "spell_save_dc", "spell_attack_bonus"],
            },
            "equipment": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Items and gear carried. Include only for humanoids or creatures that use equipment. Omit for pure beasts.",
            },
            "tactical_summary": {"type": "string"},
            "appearance": {
                "type": "string",
                "description": (
                    "Physical description for an artist: body shape, size, colours, "
                    "notable features, materials, aura. 2-4 sentences. No lore or backstory."
                ),
            },
        },
    },
}


# ---------------------------------------------------------------------------
# MonsterGenerator
# ---------------------------------------------------------------------------

class MonsterGenerator:
    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        # Load embedding model once — shared by both retrievers.
        embeddings = load_embeddings()

        # Load reranker in background while retrievers initialise sequentially.
        # (ChromaDB's Rust bindings are not safe to init from concurrent threads.)
        with ThreadPoolExecutor(max_workers=1) as ex:
            reranker_f = ex.submit(load_reranker)
            self.monsters = HybridRetriever(CHROMA_DIR, "monsters",    embeddings=embeddings)
            self.tactics  = HybridRetriever(CHROMA_DIR, "blog_tactics", embeddings=embeddings)
            self.spells   = HybridRetriever(CHROMA_DIR, "spells",       embeddings=embeddings)
            self.items    = HybridRetriever(CHROMA_DIR, "items",        embeddings=embeddings)
            reranker = reranker_f.result()

        self.monsters.reranker = reranker
        self.tactics.reranker  = reranker
        self.spells.reranker   = reranker
        self.items.reranker    = reranker

    # ------------------------------------------------------------------

    def generate(self, user_prompt: str) -> TargetMonsterOutput:
        """Generate a fresh monster from a free-form description."""
        intent = parse_intent(user_prompt, self.client)
        return self._call_generator(intent, user_prompt)

    def mutate(self, current: TargetMonsterOutput, change_request: str) -> TargetMonsterOutput:
        """Apply incremental changes to an existing monster."""
        # Re-parse intent from the change request to update retrieval context
        intent = parse_intent(
            f"{current.name} {current.monster_type}: {change_request}", self.client
        )
        intent.target_cr = intent.target_cr or current.challenge_rating
        return self._call_generator(intent, change_request, current_monster=current)

    # ------------------------------------------------------------------

    def _call_generator(
        self,
        intent: IntentResult,
        raw_prompt: str,
        current_monster: TargetMonsterOutput | None = None,
    ) -> TargetMonsterOutput:

        # Retrieval
        monster_query = f"{intent.concept} {intent.monster_type}"
        tactics_query = f"{intent.concept} {intent.monster_type} combat tactics behaviour"
        spell_query   = f"{intent.concept} {intent.monster_type} magic abilities"
        item_query    = f"{intent.concept} {intent.monster_type} weapon equipment gear"

        cr_filter    = _cr_proximity_filter(intent.target_cr)
        monster_hits = self.monsters.search(monster_query, k=3, filters=cr_filter)
        tactics_hits = self.tactics.search(tactics_query, k=3)
        spell_hits   = self.spells.search(spell_query, k=8)
        item_hits    = self.items.search(item_query, k=8)

        # Prompt assembly
        cr_block       = _cr_anchor_block(intent.target_cr)
        ref_block      = _format_reference_monsters(monster_hits)
        tactical_block = _format_tactical_context(tactics_hits)
        spell_block    = _format_spells(spell_hits)
        item_block     = _format_items(item_hits)

        if current_monster is None:
            user_content = (
                f"CONCEPT: {intent.concept}\n"
                f"Monster type : {intent.monster_type}\n"
                f"Environments : {', '.join(intent.environments) or 'any'}\n"
                f"Size         : {intent.size or 'infer from concept'}\n\n"
                f"CR MANDATES:\n{cr_block}\n\n"
                f"REFERENCE MONSTERS (style and structure only — do not copy):\n{ref_block}\n\n"
                f"TACTICAL CONTEXT (archetype behaviour patterns):\n{tactical_block}\n\n"
                f"EXISTING SPELLS YOU MAY USE OR ADAPT (choose freely, invent your own if none fit):\n{spell_block}\n\n"
                f"EXISTING ITEMS/EQUIPMENT YOU MAY REFERENCE (choose freely, invent your own if none fit):\n{item_block}"
            )
        else:
            user_content = (
                f"MODIFICATION REQUEST: {raw_prompt}\n\n"
                f"CURRENT MONSTER (mutate this — preserve unchanged fields):\n"
                f"{current_monster.model_dump_json(indent=2)}\n\n"
                f"CR MANDATES (apply if CR is changing):\n{cr_block}\n\n"
                f"REFERENCE MONSTERS:\n{ref_block}\n\n"
                f"TACTICAL CONTEXT:\n{tactical_block}\n\n"
                f"EXISTING SPELLS YOU MAY USE OR ADAPT:\n{spell_block}\n\n"
                f"EXISTING ITEMS/EQUIPMENT YOU MAY REFERENCE:\n{item_block}"
            )

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[GENERATE_TOOL],
            tool_choice={"type": "tool", "name": "generate_monster"},
            messages=[{"role": "user", "content": user_content}],
        )

        tool_use = next(b for b in response.content if b.type == "tool_use")
        return TargetMonsterOutput(**tool_use.input)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or "A cybernetic laser-shark that lives in volcanic magma"
    print(f"Generating: {prompt}\n")
    gen     = MonsterGenerator()
    monster = gen.generate(prompt)
    print(monster.model_dump_json(indent=2))
