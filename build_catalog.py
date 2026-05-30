"""
Aggregates all 5etools bestiary JSON files into monster_catalog.json,
then embeds them into ChromaDB as the 'monsters' collection.

Run from the monster-generator directory:
    python build_catalog.py

Outputs:
    monster_catalog.json   — flat normalized array of all monsters
    ./chroma_db/monsters   — ChromaDB collection for RAG retrieval
"""

import json
import re
import glob
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

BESTIARY_DIR   = "./5etools-src/data/bestiary"
CATALOG_FILE   = "monster_catalog.json"
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "monsters"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ENVIRONMENT_KEYWORDS = {
    "arctic":     ["snow", "ice", "tundra", "frost", "cold", "glacier"],
    "coastal":    ["beach", "coast", "shore", "island", "reef"],
    "desert":     ["desert", "sand", "dunes", "wastes", "oasis", "arid"],
    "forest":     ["forest", "woods", "jungle", "tree", "grove", "canopy"],
    "grassland":  ["grassland", "plains", "savanna", "prairie", "steppe"],
    "mountain":   ["mountain", "peak", "cliff", "summit", "alpine"],
    "swamp":      ["swamp", "marsh", "bog", "mire", "quagmire", "bayou"],
    "underdark":  ["underdark", "cavern", "cave", "subterranean", "dungeon"],
    "underwater": ["underwater", "ocean", "sea", "swim", "aquatic", "abyssal"],
    "urban":      ["urban", "city", "town", "sewer", "castle", "street"],
    "hill":       ["hill", "foothills", "highland"],
    "volcanic":   ["volcano", "volcanic", "magma", "lava", "fire", "molten"],
}


# ---------------------------------------------------------------------------
# Field normalisers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"\{@atk\s+[^}]+\}|\{@h\}")
_INLINE_RE = re.compile(r"\{@\w+\s+([^|}]+)[^}]*\}")

def strip_tags(text: str) -> str:
    """Remove 5etools {@tag ...} markers, keeping inner text where useful."""
    text = _TAG_RE.sub("", text)           # drop attack/hit decorators
    text = _INLINE_RE.sub(r"\1", text)     # {@skill Perception} → Perception
    return text.strip()


def entries_to_text(entries: list) -> str:
    parts = []
    for e in entries:
        if isinstance(e, str):
            parts.append(strip_tags(e))
        elif isinstance(e, dict):
            # Nested entry objects (tables, lists, etc.) — extract any 'entries'
            sub = e.get("entries") or e.get("items") or []
            if sub:
                parts.append(entries_to_text(sub))
    return " ".join(parts)


def normalize_cr(cr) -> str:
    if cr is None:
        return "unknown"
    if isinstance(cr, dict):
        return str(cr.get("cr", "unknown"))
    return str(cr)


def normalize_type(t) -> str:
    if t is None:
        return "unknown"
    if isinstance(t, dict):
        return str(t.get("type", "unknown"))
    return str(t)


def normalize_ac(ac) -> int:
    if not ac:
        return 10
    first = ac[0]
    if isinstance(first, dict):
        return int(first.get("ac", 10))
    return int(first)


def normalize_hp(hp) -> dict:
    if not hp:
        return {"average": 0, "formula": ""}
    return {"average": hp.get("average", 0), "formula": hp.get("formula", "")}


# ---------------------------------------------------------------------------
# Environment tagging (from design doc, extended)
# ---------------------------------------------------------------------------

def auto_tag_environments(monster: dict) -> list[str]:
    # If 5etools already has environment data, use it directly
    if monster.get("environment"):
        return monster["environment"]

    search_text = f"{monster.get('name', '')} ".lower()
    for trait in monster.get("trait", []):
        search_text += entries_to_text(trait.get("entries", [])).lower() + " "
    for action in monster.get("action", []):
        search_text += entries_to_text(action.get("entries", [])).lower() + " "

    tags = set()
    for env, keywords in ENVIRONMENT_KEYWORDS.items():
        if any(kw in search_text for kw in keywords):
            tags.add(env)

    monster_type = normalize_type(monster.get("type")).lower()
    if any(ft in monster_type for ft in ["demon", "devil", "fiend"]):
        tags.add("underdark")
    if any(wt in monster_type for wt in ["fish", "merfolk", "kraken"]):
        tags.add("underwater")

    return sorted(tags) if tags else ["grassland"]


# ---------------------------------------------------------------------------
# Monster normalisation
# ---------------------------------------------------------------------------

def normalize_monster(raw: dict) -> dict:
    traits = [
        {"name": t.get("name", ""), "description": entries_to_text(t.get("entries", []))}
        for t in (raw.get("trait") or [])
    ]
    actions = [
        {"name": a.get("name", ""), "description": entries_to_text(a.get("entries", []))}
        for a in (raw.get("action") or [])
    ]
    environments = auto_tag_environments(raw)

    return {
        "name":              raw["name"],
        "source":            raw.get("source", ""),
        "size":              (raw.get("size") or ["M"])[0] if isinstance(raw.get("size"), list) else raw.get("size", "M"),
        "monster_type":      normalize_type(raw.get("type")),
        "alignment":         str(raw.get("alignment", "")),
        "challenge_rating":  normalize_cr(raw.get("cr")),
        "armor_class":       normalize_ac(raw.get("ac")),
        "hit_points":        normalize_hp(raw.get("hp")),
        "speed":             raw.get("speed", {}),
        "str": raw.get("str", 10), "dex": raw.get("dex", 10),
        "con": raw.get("con", 10), "int": raw.get("int", 10),
        "wis": raw.get("wis", 10), "cha": raw.get("cha", 10),
        "environments":      environments,
        "traits":            traits,
        "actions":           actions,
        "languages":         raw.get("languages", []),
        "senses":            raw.get("senses", {}),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_all_monsters() -> list[dict]:
    files = sorted(glob.glob(f"{BESTIARY_DIR}/bestiary-*.json"))
    all_monsters = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as fh:
            data = json.load(fh)
        for raw in data.get("monster", []):
            try:
                all_monsters.append(normalize_monster(raw))
            except Exception as e:
                print(f"  [SKIP] {raw.get('name', '?')} ({fpath}): {e}")
    return all_monsters


def build_search_payload(monster: dict) -> str:
    trait_text = " ".join(t["description"] for t in monster["traits"])
    action_text = " ".join(a["description"] for a in monster["actions"])
    return (
        f"Name: {monster['name']}\n"
        f"Type: {monster['monster_type']}\n"
        f"CR: {monster['challenge_rating']}\n"
        f"Environments: {', '.join(monster['environments'])}\n"
        f"Traits: {trait_text}\n"
        f"Actions: {action_text}"
    ).strip()


def embed_catalog(monsters: list[dict]):
    print(f"Loading embedding model ({EMBEDDING_MODEL}) …")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Clear existing collection
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        print(f"Deleting existing '{COLLECTION}' collection …")
        client.delete_collection(COLLECTION)

    print(f"Building {len(monsters)} documents …")
    documents = []
    for idx, monster in enumerate(monsters):
        documents.append(Document(
            page_content=build_search_payload(monster),
            metadata={
                "name":             monster["name"],
                "monster_type":     monster["monster_type"],
                "cr":               monster["challenge_rating"],
                "environments":     ", ".join(monster["environments"]),
                "source":           monster["source"],
                "armor_class":      monster["armor_class"],
                "hit_points_avg":   monster["hit_points"]["average"],
                # Full JSON for prompt injection at generation time
                "raw_json":         json.dumps(monster),
            },
            id=f"monster_{idx}",
        ))

    print(f"Embedding into ChromaDB collection '{COLLECTION}' …")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. {db._collection.count()} documents in '{COLLECTION}'")


def main():
    print("Loading and normalising bestiary files …")
    monsters = load_all_monsters()
    print(f"  {len(monsters)} monsters loaded")

    print(f"Writing {CATALOG_FILE} …")
    with open(CATALOG_FILE, "w", encoding="utf-8") as fh:
        json.dump(monsters, fh, indent=2, ensure_ascii=False)

    embed_catalog(monsters)
    print(f"\nAll done. Catalog + ChromaDB ready.")


if __name__ == "__main__":
    main()
