# Product Design Document: AI-Powered D&D 5e Monster Generator

**Target LLM Context / Engineering Execution: Claude Code**

## 1. Executive Summary & Objective

The goal of this project is to build a production-grade, local-first Dungeons & Dragons 5e monster generator. Users interact with the system via a free-form chat interface, entering highly creative or thematic concepts (e.g., "A cybernetic laser-shark that lives in volcanic magma").

Unlike traditional generation tools that suffer from severe mechanical hallucinations, this system anchors its logic using a Hybrid Retrieval-Augmented Generation (RAG) pipeline. The generator references an existing, standardized catalog of approximately 2,500 official monsters to guide the mathematical balance and structural rules of the output, yielding balanced, homebrew stat blocks formatted in strict JSON.

## 2. System Architecture

The pipeline processes user input through a decoupled extraction, search, and synthesis loop.

```
[User Chat Input]
       │
       ▼
[1. Intent & Filter Extraction] ───> Outputs structural filters (CR, Type, Biome)
       │
       ▼
[2. Local Hybrid Search] ──────────> Queries ChromaDB (Hard Filters + Semantic Search)
       │
       ▼
[3. Prompt Synthesis] ─────────────> Stitches extracted context + Reference JSONs
       │
       ▼
[4. Structured Generation LLM] ────> Outputs clean Pydantic/JSON Monster Stat Block
```

### Components

- **Chat UI**: A terminal or web-based conversation window handling multi-turn generation.
- **Intent Parser**: A lightweight LLM call to extract structural metadata filters from messy user prose.
- **Local Vector Store**: ChromaDB running locally, storing 2,500 vector-embedded monster profiles.
- **Structured Generator**: A high-reasoning LLM (e.g., Claude 3.5 Sonnet / GPT-4o) configured with Structured Outputs / JSON Mode via Pydantic.

## 3. Data Engineering & Local RAG Setup

Because the original 2,500-monster dataset lacks environmental biomes, a pre-processing step applies automated keyword mapping and type-based heuristics to inject these tags before database hydration.

### Database Schema Strategy

To optimize search, data is bifurcated inside the local vector store:

- **The Vector Payload**: Natural language text compiling name, type, biomes, traits, and description to match thematic user vocabulary.
- **The Metadata Object**: Houses structural attributes (`cr`, `monster_type`, `environments`) for explicit database filtering, alongside the stringified `raw_json` of the original catalog entry.

### Pre-Processing & DB Build Script

```python
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Environment Tagging Framework
ENVIRONMENT_KEYWORDS = {
    "arctic": ["snow", "ice", "tundra", "frost", "cold", "glacier"],
    "coastal": ["beach", "coast", "shore", "island", "reef"],
    "desert": ["desert", "sand", "dunes", "wastes", "oasis", "arid"],
    "forest": ["forest", "woods", "jungle", "tree", "grove", "canopy"],
    "grassland": ["grassland", "plains", "savanna", "prairie", "steppe"],
    "mountain": ["mountain", "peak", "cliff", "summit", "alpine"],
    "swamp": ["swamp", "marsh", "bog", "mire", "quagmire", "bayou"],
    "underdark": ["underdark", "cavern", "cave", "subterranean", "dungeon"],
    "underwater": ["underwater", "ocean", "sea", "swim", "aquatic", "abyssal"],
    "urban": ["urban", "city", "town", "sewer", "castle", "street", "village"],
    "hill": ["hill", "foothills", "highland"]
}

def auto_tag_monster(monster):
    tags = set()
    search_space = f"{monster.get('name', '')} {monster.get('description', '')} ".lower()
    if "traits" in monster:
        search_space += " ".join([t.get("description", "") for t in monster["traits"]]).lower()
    
    for env, keywords in ENVIRONMENT_KEYWORDS.items():
        if any(keyword in search_space for keyword in keywords):
            tags.add(env)
            
    monster_type = monster.get("monster_type", "").lower()
    if any(ft in monster_type for ft in ["demon", "devil", "fiend"]):
        tags.add("underdark")
    if any(wt in monster_type for wt in ["fish", "merfolk", "kraken"]):
        tags.add("underwater")
        
    return list(tags) if tags else ["grassland"]

# 2. Pipeline Execution
def build_local_vector_db(source_json_path, db_dir):
    with open(source_json_path, "r") as f:
        catalog = json.load(f)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = []

    for idx, monster in enumerate(catalog):
        environments = auto_tag_monster(monster)
        
        search_payload = f"Name: {monster['name']}\nType: {monster['monster_type']}\nEnvironments: {', '.join(environments)}\nDescription: {monster.get('description', '')}"
        
        metadata = {
            "cr": str(monster['challenge_rating']),
            "monster_type": monster['monster_type'].lower(),
            "environments": ", ".join(environments),
            "raw_json": json.dumps(monster)
        }
        
        documents.append(Document(page_content=search_payload.strip(), metadata=metadata, id=f"monster_{idx}"))

    return Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=db_dir)
```

## 4. Master Output Schema (Pydantic Implementation)

To guarantee syntactic accuracy and programmatic usability at the UI layer, the generation model must strictly adhere to this schema.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class AbilityScores(BaseModel):
    strength: int = Field(..., ge=1, le=30)
    dexterity: int = Field(..., ge=1, le=30)
    constitution: int = Field(..., ge=1, le=30)
    intelligence: int = Field(..., ge=1, le=30)
    wisdom: int = Field(..., ge=1, le=30)
    charisma: int = Field(..., ge=1, le=30)

class TacticalRule(BaseModel):
    condition: str = Field(description="Trigger criteria, e.g., 'Targeted by a ranged spell'")
    action: str = Field(description="Tactical countermeasure, e.g., 'Uses reaction to gain total cover behind terrain'")
    priority: int = Field(description="Evaluation hierarchy order")

class CombatBehavior(BaseModel):
    combat_philosophy: str = Field(description="Aggressive, elusive, or strategic mindset summary")
    target_priority: str = Field(description="Target selection criteria based on armor, range, or spellcasting status")
    morale_trigger: str = Field(description="Specific HP thresholds or environmental factors that cause retreat/surrender")
    round_by_round_tactics: List[str] = Field(description="Conditional sequencing for rounds 1, 2, and 3+")

class MonsterAction(BaseModel):
    name: str
    action_type: str = Field(description="Melee Weapon Attack, Ranged Weapon Attack, or Special Action")
    description: str
    damage_dice: Optional[str] = Field(None, description="e.g., '2d6 + 4'")

class TargetMonsterOutput(BaseModel):
    name: str
    size: str = Field(description="Tiny, Small, Medium, Large, Huge, or Gargantuan")
    monster_type: str = Field(description="e.g., Fiend, Undead, Dragon, Humanoid")
    alignment: str
    challenge_rating: str
    armor_class: int
    hit_points: int
    speed: Dict[str, int] = Field(description="e.g., {'walk': 30, 'fly': 60}")
    abilities: AbilityScores
    saving_throws: List[str]
    skills: List[str]
    senses: List[str]
    languages: List[str]
    traits: List[Dict[str, str]] = Field(description="Passive mechanics, e.g., Pack Tactics")
    actions: List[MonsterAction]
    legendary_actions: Optional[List[Dict[str, str]]] = None
    tactical_rules: List[TacticalRule] = Field(description="Mechanized behavior rules extracted logically from tactical constraints")
    tactical_summary: str = Field(description="A cohesive, non-copyrighted guide explaining how to run the monster effectively")
```

## 5. Prompt Engineering & Mechanical Safeguards

### The CR Math Problem

LLMs are inherently inaccurate at mathematical scaling. To counteract this, a static lookup array derived from the Dungeon Master's Guide is hardcoded directly inside the generation prompt context.

```python
CR_MATHEMATICAL_ANCHORS = """
CR STATISTICAL MANDATES (Strict Alignment Required):
- CR 1: HP 71-85, AC 13, Prof +2, Attack Bonus +3, Damage/Round 9-14
- CR 2: HP 86-100, AC 13, Prof +2, Attack Bonus +3, Damage/Round 15-20
- CR 3: HP 101-115, AC 13, Prof +2, Attack Bonus +4, Damage/Round 21-26
- CR 5: HP 131-145, AC 15, Prof +3, Attack Bonus +6, Damage/Round 33-38
- CR 10: HP 206-220, AC 17, Prof +4, Attack Bonus +7, Damage/Round 63-68
- CR 15: HP 296-310, AC 18, Prof +5, Attack Bonus +9, Damage/Round 93-98
"""
```

### Prompt Strategy

The system instructs the LLM to use the retrieved RAG files solely for syntactic style and feature concepts, while strictly calculating numerical indices using the mathematical anchors mapped to the target CR.

```python
GENERATOR_SYSTEM_PROMPT = """
You are an expert D&D 5e mechanics engine and systems designer. 
Your objective is to generate a completely novel creature that fulfills the user's thematic intent.

[CR_MATHEMATICAL_ANCHORS]

INSTRUCTIONS FOR CREATIVE FUSION:
1. Do not copy the reference monsters. Instead, isolate their trait phrasing styles and action economies.
2. Ensure that Ability Scores correspond mathematically to their respective modifiers (e.g., STR 16 must yield a +3 bonus to hit/damage calculations).
3. Derive tactical rules and behavior patterns cleanly from the underlying statistics, prioritizing self-preservation and thematic capability.
"""
```

## 6. System Decisions & Open Scope

Before implementing this specification inside Claude Code, the following tactical architectural parameters must be finalized.

### 1. The Generation LLM Provider

Because ChromaDB and the embedding model run completely locally, the heavy-lifting generation model can take one of two paths:

- **Option A (Hybrid Cloud API - Recommended)**: Route the generation query via Anthropic's Claude 3.5 Sonnet API. This provides maximum reasoning capabilities, flawless compliance with intricate Pydantic nested structures, and precise mathematical calculations.
- **Option B (100% Air-Gapped Local)**: Route the generation query through a local tool like Ollama running Llama-3-8b or Mistral-7b. This satisfies an entirely local compute constraint but increases the risk of JSON serialization dropouts and minor balancing math variances.

### 2. Conversational State & Multi-Turn Iteration

The lifecycle handling strategy must resolve how modifications are processed:

- **Option A (Stateless)**: Every single chat message generates a standalone, completely new monster from scratch.
- **Option B (Stateful Session)**: When a user asks for modifications (e.g., "Reduce its armor class and change fire damage to acid"), the app appends the previously outputted JSON structure into the conversation window background history. This passes the active object back to the model as context so it treats edits as incremental mutations.

### 3. The Extraction of "The Monsters Know" Logic

To incorporate tactical behaviors cleanly without ingesting copyrighted prose, select a collection strategy for the background dataset:

- **Option A (Static Engine Rulebook)**: Hardcode a robust, deterministic system prompt explaining how monsters act based on their stats (e.g., Int ≤ 7 implies primitive hunter mechanics; Dex ≥ 16 implies hit-and-run skirmishing).
- **Option B (A Secondary Pre-Processing Script)**: Prior to deployment, process textual strategy summaries through a translation script that strips prose and converts behavioral insights into structured if/then conditionals. Save these rules directly inside your primary monster catalog JSON database.
