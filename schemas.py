"""
Pydantic output schema for generated monsters — matches the design doc exactly.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class AbilityScores(BaseModel):
    strength:     int = Field(..., ge=1, le=30)
    dexterity:    int = Field(..., ge=1, le=30)
    constitution: int = Field(..., ge=1, le=30)
    intelligence: int = Field(..., ge=1, le=30)
    wisdom:       int = Field(..., ge=1, le=30)
    charisma:     int = Field(..., ge=1, le=30)


class TacticalRule(BaseModel):
    condition: str = Field(description="Trigger criteria")
    action:    str = Field(description="What the monster does")
    priority:  int = Field(description="Evaluation order — 1 is highest")


class MonsterAction(BaseModel):
    name:        str
    action_type: str = Field(description="Melee Weapon Attack, Ranged Weapon Attack, or Special Action")
    description: str
    damage_dice: Optional[str] = Field(None, description="e.g. '2d6 + 4'")


class SpellcastingBlock(BaseModel):
    ability:           str = Field(description="Spellcasting ability e.g. 'Intelligence'")
    spell_save_dc:     int
    spell_attack_bonus: int
    innate_spells:     Optional[List[str]] = Field(None, description="Spells usable at will or X/day without slots")
    at_will:           Optional[List[str]] = Field(None, description="Spells castable at will")
    spell_slots:       Optional[Dict[str, List[str]]] = Field(
        None, description="Slot level → spell list e.g. {'1st': ['Magic Missile'], '3rd': ['Fireball']}"
    )


class TargetMonsterOutput(BaseModel):
    name:              str
    size:              str = Field(description="Tiny, Small, Medium, Large, Huge, or Gargantuan")
    monster_type:      str = Field(description="e.g. Fiend, Undead, Dragon, Humanoid")
    alignment:         str
    challenge_rating:  str
    armor_class:       int
    hit_points:        int
    speed:             Dict[str, int] = Field(description="e.g. {'walk': 30, 'fly': 60}")
    abilities:         AbilityScores
    saving_throws:     List[str]
    skills:            List[str]
    damage_immunities: List[str] = Field(default_factory=list)
    damage_resistances: List[str] = Field(default_factory=list)
    condition_immunities: List[str] = Field(default_factory=list)
    senses:            List[str]
    languages:         List[str]
    traits:            List[Dict[str, str]] = Field(description="Passive mechanics e.g. Pack Tactics")
    actions:           List[MonsterAction]
    legendary_actions: Optional[List[Dict[str, str]]] = None
    spellcasting:      Optional[SpellcastingBlock] = None
    equipment:         Optional[List[str]] = Field(None, description="Items and gear carried by the creature")
    tactical_rules:    List[TacticalRule]
    tactical_summary:  str = Field(description="How to run this monster effectively")
    appearance:        str = Field(description="Physical description for an artist: body shape, size, colours, notable features, materials, aura. 2-4 sentences. No lore, no backstory.")
