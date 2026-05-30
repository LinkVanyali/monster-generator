"""
Embeds D&D 5e spells and items into ChromaDB for RAG retrieval at generation time.

Collections produced:
    spells  — 936 spells with level, school, damage tags, and description
    items   — 2,417 magic items + 230 base items with type, rarity, and description

Run from the monster-generator directory (venv active):
    python embed_game_data.py
"""

import json
import re
import glob
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

SPELLS_DIR   = "./5etools-src/data/spells"
ITEMS_FILE   = "./5etools-src/data/items.json"
ITEMS_BASE   = "./5etools-src/data/items-base.json"
CHROMA_DIR   = "./chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"

SCHOOL_NAMES = {
    "A": "Abjuration", "C": "Conjuration", "D": "Divination",
    "E": "Enchantment", "I": "Illusion",   "N": "Necromancy",
    "T": "Transmutation", "V": "Evocation",
}
DAMAGE_TYPES = {
    "A": "acid", "B": "bludgeoning", "C": "cold", "F": "fire",
    "O": "force", "L": "lightning", "N": "necrotic", "P": "piercing",
    "I": "poison", "Y": "psychic", "R": "radiant", "S": "slashing",
    "T": "thunder",
}
ITEM_TYPES = {
    "A": "Armor", "G": "Adventuring Gear", "HA": "Heavy Armor",
    "LA": "Light Armor", "MA": "Medium Armor", "S": "Shield",
    "M": "Melee Weapon", "R": "Ranged Weapon", "AT": "Artisan Tools",
    "T": "Tools", "GS": "Gaming Set", "INS": "Instrument",
    "SCF": "Spellcasting Focus", "W": "Wand", "ST": "Staff",
    "RD": "Rod", "RG": "Ring", "WD": "Wondrous Item",
    "P": "Potion", "SC": "Scroll", "EXP": "Explosive",
}

_TAG_RE    = re.compile(r"\{@atk\s+[^}]+\}|\{@h\}")
_INLINE_RE = re.compile(r"\{@\w+\s+([^|}]+)[^}]*\}")

def strip_tags(text: str) -> str:
    text = _TAG_RE.sub("", text)
    return _INLINE_RE.sub(r"\1", text).strip()

def entries_to_text(entries: list, max_chars: int = 800) -> str:
    parts = []
    for e in entries:
        if isinstance(e, str):
            parts.append(strip_tags(e))
        elif isinstance(e, dict):
            sub = e.get("entries") or e.get("items") or []
            if sub:
                parts.append(entries_to_text(sub))
    return " ".join(parts)[:max_chars]


# ---------------------------------------------------------------------------
# Spells
# ---------------------------------------------------------------------------

def load_spells() -> list[dict]:
    spells = []
    for fpath in sorted(glob.glob(f"{SPELLS_DIR}/spells-*.json")):
        with open(fpath, encoding="utf-8") as f:
            spells.extend(json.load(f).get("spell", []))
    return spells


def spell_search_payload(spell: dict) -> str:
    level  = spell.get("level", 0)
    school = SCHOOL_NAMES.get(spell.get("school", ""), spell.get("school", ""))
    dmg    = ", ".join(spell.get("damageInflict", []))
    saves  = ", ".join(spell.get("savingThrow", []))
    desc   = entries_to_text(spell.get("entries", []))
    return (
        f"Spell: {spell['name']}\n"
        f"Level: {level} | School: {school}"
        + (f" | Damage: {dmg}" if dmg else "")
        + (f" | Save: {saves}" if saves else "") + "\n"
        + desc
    ).strip()


def embed_spells(embeddings):
    spells = load_spells()
    print(f"  {len(spells)} spells loaded")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if "spells" in [c.name for c in client.list_collections()]:
        print("  Deleting existing 'spells' collection …")
        client.delete_collection("spells")

    documents = []
    for idx, spell in enumerate(spells):
        level  = spell.get("level", 0)
        school = SCHOOL_NAMES.get(spell.get("school", ""), "")
        dmg    = ", ".join(spell.get("damageInflict", []))
        documents.append(Document(
            page_content=spell_search_payload(spell),
            metadata={
                "name":   spell["name"],
                "level":  level,
                "school": school,
                "damage": dmg,
                "source": spell.get("source", ""),
                "description": entries_to_text(spell.get("entries", []), max_chars=1200),
            },
            id=f"spell_{idx}",
        ))

    Chroma.from_documents(
        documents=documents, embedding=embeddings,
        collection_name="spells", persist_directory=CHROMA_DIR,
    )
    print(f"  {len(documents)} spells embedded → 'spells'")


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def load_items() -> list[dict]:
    with open(ITEMS_FILE, encoding="utf-8") as f:
        items = json.load(f).get("item", [])
    with open(ITEMS_BASE, encoding="utf-8") as f:
        items += json.load(f).get("baseitem", [])
    return items


def item_search_payload(item: dict) -> str:
    type_label = ITEM_TYPES.get(item.get("type", ""), item.get("type", "item"))
    rarity     = item.get("rarity", "")
    attune     = "requires attunement" if item.get("reqAttune") else ""
    weapon_cat = item.get("weaponCategory", "")
    dmg        = f"{item.get('dmg1','')} {DAMAGE_TYPES.get(item.get('dmgType',''),'')}" .strip()
    desc       = entries_to_text(item.get("entries", []))
    return (
        f"Item: {item['name']}\n"
        f"Type: {type_label}"
        + (f" | {weapon_cat}" if weapon_cat else "")
        + (f" | Rarity: {rarity}" if rarity else "")
        + (f" | {attune}" if attune else "")
        + (f" | Damage: {dmg}" if dmg else "") + "\n"
        + desc
    ).strip()


def embed_items(embeddings):
    items = load_items()
    print(f"  {len(items)} items loaded")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if "items" in [c.name for c in client.list_collections()]:
        print("  Deleting existing 'items' collection …")
        client.delete_collection("items")

    documents = []
    for idx, item in enumerate(items):
        type_label = ITEM_TYPES.get(item.get("type", ""), item.get("type", ""))
        documents.append(Document(
            page_content=item_search_payload(item),
            metadata={
                "name":        item["name"],
                "type":        type_label,
                "rarity":      item.get("rarity", ""),
                "weapon_cat":  item.get("weaponCategory", ""),
                "dmg":         item.get("dmg1", ""),
                "source":      item.get("source", ""),
                "req_attune":  str(bool(item.get("reqAttune"))),
                "description": entries_to_text(item.get("entries", []), max_chars=800),
            },
            id=f"item_{idx}",
        ))

    Chroma.from_documents(
        documents=documents, embedding=embeddings,
        collection_name="items", persist_directory=CHROMA_DIR,
    )
    print(f"  {len(documents)} items embedded → 'items'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading embedding model ({EMBED_MODEL}) …")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Embedding spells …")
    embed_spells(embeddings)

    print("Embedding items …")
    embed_items(embeddings)

    print("\nDone.")


if __name__ == "__main__":
    main()
