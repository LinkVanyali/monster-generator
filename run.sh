#!/bin/bash
# Start the D&D monster generator chat UI.
# On first run, set up the venv and build the ChromaDB collections.

set -e
cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment…"
    python3 -m venv venv
fi

source venv/bin/activate

# Install / sync dependencies
pip install -q -r requirements.txt

# Build ChromaDB collections if not yet done
if [ ! -d "chroma_db/monsters" ]; then
    echo "Building monster catalog (first run — takes a few minutes)…"
    python build_catalog.py
fi

if [ ! -d "chroma_db/blog_tactics" ]; then
    echo "Embedding Monsters Know blog posts…"
    python embed_blog_posts.py
fi

if [ ! -d "chroma_db/spells" ] || [ ! -d "chroma_db/items" ]; then
    echo "Embedding spells and items…"
    python embed_game_data.py
fi

# Launch
python chat.py
