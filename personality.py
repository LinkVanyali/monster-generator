"""
NPC personality layer — generates roleplay context for humanoid monsters.

Uses Claude Haiku (fast, cheap) with tool use for structured output.
Grounded in the stat block so personality reflects the creature's abilities.

Usage:
    from personality import generate_personality, NPCPersonality

    p = generate_personality(monster, client)   # ~2-3 seconds
    p.display()
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
import anthropic

from schemas import TargetMonsterOutput

MODEL = "claude-haiku-4-5-20251001"

PERSONALITY_TOOL = {
    "name": "generate_npc_personality",
    "description": "Generate roleplay personality details for a D&D 5e NPC.",
    "input_schema": {
        "type": "object",
        "required": [
            "name", "occupation", "voice",
            "personality_traits", "ideal", "bond", "flaw",
            "motivation", "secret", "roleplay_hooks",
        ],
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name, culturally appropriate to the creature's type and alignment.",
            },
            "pronouns": {
                "type": "string",
                "description": "e.g. 'he/him', 'she/her', 'they/them'. Infer from context or default to they/them.",
            },
            "occupation": {
                "type": "string",
                "description": "What this NPC does — their role in the world. 1 short sentence.",
            },
            "voice": {
                "type": "string",
                "description": (
                    "How they speak: pace, vocabulary level, verbal tics, accent hints, "
                    "tone. 1-2 sentences a DM can act on immediately."
                ),
            },
            "personality_traits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 distinct personality traits grounded in the stat block (e.g. high INT → calculating).",
            },
            "ideal": {
                "type": "string",
                "description": "The principle or belief that drives them. 1 sentence.",
            },
            "bond": {
                "type": "string",
                "description": "The person, place, or thing they care about most. 1 sentence.",
            },
            "flaw": {
                "type": "string",
                "description": "Their weakness, vice, or blind spot. 1 sentence.",
            },
            "motivation": {
                "type": "string",
                "description": "What they want right now, in this encounter. 1 sentence.",
            },
            "secret": {
                "type": "string",
                "description": "Something they're hiding from the party. Can be used as a plot hook.",
            },
            "roleplay_hooks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 specific ways the party might interact with or be affected by this NPC.",
            },
            "mannerisms": {
                "type": "string",
                "description": "A physical habit or distinctive behaviour a DM can use at the table. 1 sentence.",
            },
        },
    },
}

SYSTEM_PROMPT = """\
You are a D&D 5e NPC designer. Given a monster stat block, generate rich roleplay context
that a DM can use immediately at the table. Ground the personality in the stat block:
high Charisma suggests charm or intimidation; low Intelligence suggests bluntness;
Constitution implies toughness or stubbornness; the creature's alignment and type
inform their worldview. Make details specific and evocative — avoid generic fantasy clichés.\
"""


class NPCPersonality(BaseModel):
    name:               str
    pronouns:           Optional[str] = "they/them"
    occupation:         str
    voice:              str
    personality_traits: list[str]
    ideal:              str
    bond:               str
    flaw:               str
    motivation:         str
    secret:             str
    roleplay_hooks:     list[str]
    mannerisms:         Optional[str] = None

    def display(self, use_colour: bool = True):
        def _c(code, t): return f"\033[{code}m{t}\033[0m" if use_colour else t
        bold = lambda t: _c("1", t)
        dim  = lambda t: _c("2", t)
        cyan = lambda t: _c("96", t)
        yel  = lambda t: _c("93", t)

        print()
        print(_c("2", "─" * 60))
        print(bold(cyan("NPC PERSONALITY")))
        print(f"{bold(self.name)}  {dim(self.pronouns or '')}  —  {self.occupation}")
        print()
        print(f"{bold('Voice')}       {self.voice}")
        if self.mannerisms:
            print(f"{bold('Mannerism')}   {self.mannerisms}")
        print()
        for trait in self.personality_traits:
            print(f"  • {trait}")
        print()
        print(f"{bold('Ideal')}       {self.ideal}")
        print(f"{bold('Bond')}        {self.bond}")
        print(f"{bold('Flaw')}        {self.flaw}")
        print(f"{bold('Motivation')}  {self.motivation}")
        print(f"{bold('Secret')}      {yel(self.secret)}")
        print()
        print(bold("Roleplay hooks:"))
        for hook in self.roleplay_hooks:
            print(f"  → {hook}")
        print(_c("2", "─" * 60))

    def to_biography_html(self) -> str:
        """Formats personality as Foundry actor biography HTML."""
        traits_html = "".join(f"<li>{t}</li>" for t in self.personality_traits)
        hooks_html  = "".join(f"<li>{h}</li>" for h in self.roleplay_hooks)
        return (
            f"<h2>{self.name}</h2>"
            f"<p><em>{self.occupation}</em></p>"
            f"<p><strong>Voice:</strong> {self.voice}</p>"
            + (f"<p><strong>Mannerism:</strong> {self.mannerisms}</p>" if self.mannerisms else "")
            + f"<ul>{traits_html}</ul>"
            f"<p><strong>Ideal:</strong> {self.ideal}</p>"
            f"<p><strong>Bond:</strong> {self.bond}</p>"
            f"<p><strong>Flaw:</strong> {self.flaw}</p>"
            f"<p><strong>Motivation:</strong> {self.motivation}</p>"
            f"<p><strong>Secret:</strong> {self.secret}</p>"
            f"<h3>Roleplay Hooks</h3><ul>{hooks_html}</ul>"
        )


def is_humanoid(monster: TargetMonsterOutput) -> bool:
    t = monster.monster_type.lower()
    return t in ("humanoid", "npc", "human", "elf", "dwarf", "halfling",
                 "gnome", "half-orc", "tiefling", "dragonborn")


def generate_personality(
    monster: TargetMonsterOutput,
    client: Optional[anthropic.Anthropic] = None,
) -> NPCPersonality:
    if client is None:
        client = anthropic.Anthropic()

    # Summarise the stat block for the prompt — key mechanics that inform personality
    ab = monster.abilities
    stat_summary = (
        f"Name: {monster.name}\n"
        f"Type: {monster.size} {monster.monster_type}, {monster.alignment}\n"
        f"CR: {monster.challenge_rating}\n"
        f"STR {ab.strength} DEX {ab.dexterity} CON {ab.constitution} "
        f"INT {ab.intelligence} WIS {ab.wisdom} CHA {ab.charisma}\n"
        f"Traits: {', '.join(t.get('name','') for t in monster.traits[:4])}\n"
        f"Actions: {', '.join(a.name for a in monster.actions[:4])}\n"
        f"Appearance: {getattr(monster, 'appearance', '') or ''}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[PERSONALITY_TOOL],
        tool_choice={"type": "tool", "name": "generate_npc_personality"},
        messages=[{"role": "user", "content": f"Generate NPC personality for:\n\n{stat_summary}"}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return NPCPersonality(**tool_use.input)
