"""
Encounter builder — selects monsters from the existing catalog to build
a balanced D&D 5e encounter for a given party.

Usage:
    from encounter import build_encounter

    result = build_encounter(
        "4 level-6 players, hard difficulty, forest ambush by bandits",
        monster_retriever,
        client,
    )
    result.display()
"""

import json
from dataclasses import dataclass, field
from typing import Optional
import anthropic

from retriever import HybridRetriever

MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# D&D 5e encounter math tables
# ---------------------------------------------------------------------------

# XP thresholds per player per level  {level: (easy, medium, hard, deadly)}
XP_THRESHOLDS = {
    1:  (25,   50,   75,   100),
    2:  (50,   100,  150,  200),
    3:  (75,   150,  225,  400),
    4:  (125,  250,  375,  500),
    5:  (250,  500,  750,  1100),
    6:  (300,  600,  900,  1400),
    7:  (350,  750,  1100, 1700),
    8:  (450,  900,  1400, 2100),
    9:  (550,  1100, 1600, 2400),
    10: (600,  1200, 1900, 2800),
    11: (800,  1600, 2400, 3600),
    12: (1000, 2000, 3000, 4500),
    13: (1100, 2200, 3400, 5100),
    14: (1250, 2500, 3800, 5700),
    15: (1400, 2800, 4300, 6400),
    16: (1600, 3200, 4800, 7200),
    17: (2000, 3900, 5900, 8800),
    18: (2100, 4200, 6300, 9500),
    19: (2400, 4900, 7300, 10900),
    20: (2800, 5700, 8500, 12700),
}

DIFFICULTY_IDX = {"easy": 0, "medium": 1, "hard": 2, "deadly": 3}

# XP multiplier by monster count (standard party size 3-5)
XP_MULTIPLIERS = [
    (1,   1.0),
    (2,   1.5),
    (6,   2.0),
    (10,  2.5),
    (14,  3.0),
    (999, 4.0),
]

CR_TO_XP = {
    "0": 10, "1/8": 25, "1/4": 50, "1/2": 100,
    "1": 200, "2": 450, "3": 700, "4": 1100,
    "5": 1800, "6": 2300, "7": 2900, "8": 3900,
    "9": 5000, "10": 5900, "11": 7200, "12": 8400,
    "13": 10000, "14": 11500, "15": 13000, "16": 15000,
    "17": 18000, "18": 20000, "19": 22000, "20": 25000,
    "21": 33000, "22": 41000, "23": 50000, "24": 62000,
    "25": 75000, "26": 90000, "27": 105000, "28": 120000,
    "29": 135000, "30": 155000,
}


def xp_budget(party_size: int, party_level: int, difficulty: str) -> tuple[int, int]:
    """Returns (min_xp, max_xp) raw XP budget for the encounter (before multiplier)."""
    level = max(1, min(20, party_level))
    idx   = DIFFICULTY_IDX.get(difficulty.lower(), 1)
    per_player = XP_THRESHOLDS[level][idx]
    # Budget is between this difficulty and the next
    next_idx    = min(idx + 1, 3)
    per_upper   = XP_THRESHOLDS[level][next_idx]
    return per_player * party_size, per_upper * party_size


def xp_multiplier(n_monsters: int) -> float:
    for threshold, mult in XP_MULTIPLIERS:
        if n_monsters <= threshold:
            return mult
    return 4.0


def adjusted_xp(monster_xps: list[int]) -> int:
    total = sum(monster_xps)
    return int(total * xp_multiplier(len(monster_xps)))


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------

PARSE_TOOL = {
    "name": "parse_encounter_request",
    "description": "Extract structured parameters from an encounter description.",
    "input_schema": {
        "type": "object",
        "required": ["party_size", "party_level", "difficulty", "theme"],
        "properties": {
            "party_size":  {"type": "integer", "description": "Number of players"},
            "party_level": {"type": "integer", "description": "Average character level"},
            "difficulty":  {"type": "string", "enum": ["easy", "medium", "hard", "deadly"]},
            "theme":       {"type": "string", "description": "Thematic description for monster search"},
        },
    },
}

SELECT_TOOL = {
    "name": "select_encounter_monsters",
    "description": "Select a monster combination from the candidates to build a balanced encounter.",
    "input_schema": {
        "type": "object",
        "required": ["selections", "gm_note"],
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "count", "role"],
                    "properties": {
                        "name":  {"type": "string", "description": "Exact monster name from the candidates list"},
                        "count": {"type": "integer", "description": "Number of this monster in the encounter"},
                        "role":  {"type": "string", "description": "Tactical role: leader, brute, skirmisher, controller, lurker, minion"},
                        "tactic": {"type": "string", "description": "One sentence on how this monster behaves in the encounter"},
                    },
                },
            },
            "gm_note": {
                "type": "string",
                "description": "2-3 sentence GM summary: how the encounter opens, escalates, and ends.",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EncounterMonster:
    name:    str
    cr:      str
    count:   int
    role:    str
    tactic:  str
    hp:      int = 0
    ac:      int = 0
    source:  str = ""


@dataclass
class EncounterResult:
    theme:       str
    party_size:  int
    party_level: int
    difficulty:  str
    xp_budget:   tuple[int, int]
    monsters:    list[EncounterMonster] = field(default_factory=list)
    total_xp:    int = 0
    adjusted:    int = 0
    gm_note:     str = ""

    def display(self, use_colour: bool = True):
        def _c(code, t): return f"\033[{code}m{t}\033[0m" if use_colour else t
        bold = lambda t: _c("1", t)
        dim  = lambda t: _c("2", t)
        cyan = lambda t: _c("96", t)
        yel  = lambda t: _c("93", t)
        grn  = lambda t: _c("92", t)

        print()
        print(_c("2", "─" * 60))
        print(bold(cyan("ENCOUNTER")))
        print(
            f"{self.party_size} players, level {self.party_level}  "
            f"│  {bold(yel(self.difficulty.upper()))}  "
            f"│  Budget {self.xp_budget[0]:,}–{self.xp_budget[1]:,} XP"
        )
        print(
            f"Adjusted XP: {bold(str(self.adjusted))}  "
            f"(raw {self.total_xp:,} × {xp_multiplier(sum(m.count for m in self.monsters))})"
        )
        print()

        for m in self.monsters:
            count_str = f"×{m.count}" if m.count > 1 else "  "
            print(
                f"  {bold(m.name)} {count_str}  "
                f"{dim('CR')} {m.cr}  {dim('HP')} {m.hp}  {dim('AC')} {m.ac}  "
                f"{dim('['+ m.role +']')}"
            )
            print(f"     {dim('→')} {m.tactic}")

        print()
        print(bold("GM Note:"))
        print(self.gm_note)
        print(_c("2", "─" * 60))
        print()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_encounter(
    prompt: str,
    monster_retriever: HybridRetriever,
    client: anthropic.Anthropic,
) -> EncounterResult:

    # 1. Parse intent
    parse_resp = client.messages.create(
        model=MODEL, max_tokens=256,
        tools=[PARSE_TOOL],
        tool_choice={"type": "tool", "name": "parse_encounter_request"},
        messages=[{"role": "user", "content": prompt}],
    )
    params = next(b for b in parse_resp.content if b.type == "tool_use").input
    party_size  = params["party_size"]
    party_level = params["party_level"]
    difficulty  = params["difficulty"]
    theme       = params["theme"]

    budget_lo, budget_hi = xp_budget(party_size, party_level, difficulty)

    # 2. Search catalog for thematically relevant monsters
    #    Target CR: roughly budget_hi / (2 * party_size) as a starting point
    # Pull enough candidates for Claude to have choices
    candidates = monster_retriever.search(theme, k=20)

    if not candidates:
        raise RuntimeError("No monsters found in catalog — ensure build_catalog.py has been run.")

    # Build candidate summary for the selection prompt
    candidate_lines = []
    candidate_map   = {}
    for hit in candidates:
        m = json.loads(hit["metadata"]["raw_json"])
        cr  = m.get("challenge_rating", "?")
        xp  = CR_TO_XP.get(str(cr), 0)
        hp  = m.get("hit_points", {}).get("average", 0) if isinstance(m.get("hit_points"), dict) else 0
        ac  = m.get("armor_class", 0)
        line = (
            f"- {m['name']} | CR {cr} | {xp} XP | HP {hp} | AC {ac} | "
            f"{m.get('monster_type','?')}"
        )
        candidate_lines.append(line)
        candidate_map[m["name"]] = {"cr": cr, "hp": hp, "ac": ac,
                                     "source": m.get("source", ""),
                                     "xp": xp}

    selection_prompt = (
        f"Build a D&D 5e encounter for {party_size} players at level {party_level}.\n"
        f"Difficulty: {difficulty.upper()}\n"
        f"XP budget (raw, before multiplier): {budget_lo:,}–{budget_hi:,} XP\n"
        f"Theme: {theme}\n\n"
        f"XP multiplier guide: 1 monster=×1, 2=×1.5, 3-6=×2, 7-10=×2.5\n\n"
        f"Available monsters:\n" + "\n".join(candidate_lines) + "\n\n"
        f"Select 1-8 monster types (with counts) whose adjusted XP falls within the budget. "
        f"Prioritise synergistic tactics — mix roles (leader, brute, skirmisher, controller, lurker, minion). "
        f"Keep total monster count reasonable (2-8 for most encounters)."
    )

    select_resp = client.messages.create(
        model=MODEL, max_tokens=1024,
        tools=[SELECT_TOOL],
        tool_choice={"type": "tool", "name": "select_encounter_monsters"},
        messages=[{"role": "user", "content": selection_prompt}],
    )
    selection = next(b for b in select_resp.content if b.type == "tool_use").input

    # 3. Assemble result
    enc_monsters = []
    total_xp = 0
    for sel in selection["selections"]:
        name  = sel["name"]
        count = sel.get("count", 1)
        info  = candidate_map.get(name, {})
        total_xp += info.get("xp", 0) * count
        enc_monsters.append(EncounterMonster(
            name=name, cr=info.get("cr", "?"), count=count,
            role=sel.get("role", ""), tactic=sel.get("tactic", ""),
            hp=info.get("hp", 0), ac=info.get("ac", 0),
            source=info.get("source", ""),
        ))

    adj = adjusted_xp([
        CR_TO_XP.get(str(m.cr), 0)
        for m in enc_monsters
        for _ in range(m.count)
    ])

    return EncounterResult(
        theme=theme, party_size=party_size, party_level=party_level,
        difficulty=difficulty, xp_budget=(budget_lo, budget_hi),
        monsters=enc_monsters, total_xp=total_xp, adjusted=adj,
        gm_note=selection.get("gm_note", ""),
    )
