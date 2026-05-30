"""
Intent parser: extracts structured metadata from a free-form user creature description.

Uses Claude Haiku (fast, cheap) with tool use to guarantee structured output.

Usage:
    from intent_parser import parse_intent

    intent = parse_intent("A cybernetic laser-shark that lives in volcanic magma")
    # IntentResult(
    #     concept="cybernetic shark with laser attacks living in volcanic/magma environment",
    #     monster_type="monstrosity",
    #     environments=["volcanic", "underwater"],
    #     target_cr=None,
    #     size=None,
    # )
"""

from typing import Optional
from pydantic import BaseModel
import anthropic

MODEL = "claude-haiku-4-5-20251001"

VALID_TYPES = [
    "aberration", "beast", "celestial", "construct", "dragon",
    "elemental", "fey", "fiend", "giant", "humanoid",
    "monstrosity", "ooze", "plant", "undead",
]

VALID_ENVIRONMENTS = [
    "arctic", "coastal", "desert", "forest", "grassland",
    "hill", "mountain", "swamp", "underdark", "underwater",
    "urban", "volcanic",
]

VALID_SIZES = ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"]

EXTRACT_TOOL = {
    "name": "extract_intent",
    "description": "Extract D&D 5e monster metadata from a user's creature description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concept": {
                "type": "string",
                "description": (
                    "A clean 1-2 sentence description of the creature optimised for "
                    "semantic search. Preserve the thematic flavour but remove filler "
                    "words and restate in plain creature-description prose."
                ),
            },
            "monster_type": {
                "type": "string",
                "enum": VALID_TYPES,
                "description": (
                    "The D&D 5e creature type that best fits the concept. "
                    "Use 'monstrosity' for hybrid or engineered creatures with no better fit."
                ),
            },
            "environments": {
                "type": "array",
                "items": {"type": "string", "enum": VALID_ENVIRONMENTS},
                "description": "Environments where this creature plausibly lives. May be empty.",
            },
            "target_cr": {
                "type": "string",
                "description": (
                    "Challenge rating if the user specified one (e.g. '5', '1/2', '17'). "
                    "Null if not mentioned."
                ),
            },
            "size": {
                "type": "string",
                "enum": VALID_SIZES,
                "description": "Physical size if inferable from the description. Null if unclear.",
            },
        },
        "required": ["concept", "monster_type", "environments"],
    },
}

SYSTEM_PROMPT = """\
You are a D&D 5e creature concept analyser. Given a user's description of a creature they want \
to generate, extract structured metadata that will be used to search a monster database and \
guide generation. Be generous in inferring creature type and environment from thematic cues.\
"""


class IntentResult(BaseModel):
    concept: str
    monster_type: str
    environments: list[str]
    target_cr: Optional[str] = None
    size: Optional[str] = None


def parse_intent(user_prompt: str, client: Optional[anthropic.Anthropic] = None) -> IntentResult:
    """
    Parse a free-form creature description into structured search filters.

    Args:
        user_prompt: Raw user input, e.g. "A cybernetic laser-shark in volcanic magma"
        client: Optional shared Anthropic client; creates one if not provided.

    Returns:
        IntentResult with extracted fields.
    """
    if client is None:
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_intent"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    data = tool_use.input

    return IntentResult(
        concept=data["concept"],
        monster_type=data["monster_type"],
        environments=data.get("environments", []),
        target_cr=data.get("target_cr"),
        size=data.get("size"),
    )


if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or "A cybernetic laser-shark that lives in volcanic magma"
    print(f"Input: {prompt}\n")
    result = parse_intent(prompt)
    print(result.model_dump_json(indent=2))
