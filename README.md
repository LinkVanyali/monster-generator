# D&D 5e Monster Generator

Production-grade, local-first monster generator using Hybrid RAG (BM25 + semantic search + reranking) and Claude AI.

## Features

- **Hybrid RAG**: References 2,500+ monsters, 429 tactical blog posts, 936 spells, 2,417 items
- **CR-balanced generation**: Validates HP, AC, attack bonus, and DPR against DMG tables
- **NPC personality layer**: Automatic character generation for roleplay context
- **Token images**: FLUX.1-schnell via HuggingFace Inference API (~$0.001/image)
- **Encounter builder**: XP-balanced encounters from existing catalog
- **Foundry VTT integration**: 
  - Direct import via API module (click "AI Monster" in Actors sidebar)
  - Manual import via Plutonium-compatible 5etools JSON
- **Stateful mutations**: Multi-turn refinement of generated creatures
- **Session management**: Auto-save with same-day resume

## Quick Start

### Backend (Terminal UI)

```bash
./run.sh
```

### Backend (Docker)

```bash
docker-compose up -d
```

### Foundry VTT Module

1. In Foundry, go to **Add-on Modules** → **Install Module**
2. Paste manifest URL:
   ```
   https://raw.githubusercontent.com/LinkVanyali/monster-generator/main/foundry-module/module.json
   ```
3. Enable the module in your world
4. Go to **Settings → Configure Settings → AI Monster Generator**
5. Set API URL to `http://YOUR_SERVER_IP:8765`

Two new buttons appear in the **Actors** sidebar:
- **AI Monster** — generate a single creature
- **AI Encounter** — build a balanced encounter

## Architecture

- **Claude Opus 4.8** for generation (structured outputs via Anthropic tool use)
- **Claude Haiku 4.5** for intent parsing and encounter selection
- **ChromaDB** with `all-MiniLM-L6-v2` embeddings
- **BM25Okapi + semantic search + cross-encoder reranking** (ms-marco-MiniLM-L-6-v2)
- **Reciprocal Rank Fusion** (RRF_K=60) for result merging
- **FastAPI** backend for Foundry module integration

## Requirements

- Python 3.12+
- Anthropic API key
- HuggingFace API key (for image generation)
- Optional: `FOUNDRY_DATA` env var for auto-copying token images

See `.env.example` for configuration.

## Commands (Terminal UI)

```
/new                    Generate a new monster
/load [file|#]          Load from file or session log (no args = picker)
/image                  Generate token image
/personality            Generate NPC personality
/encounter <desc>       Build an encounter
/json                   Export raw schema JSON
/foundry                Export 5etools/Plutonium JSON
/save [file]            Save to file
/log                    Show session history
/quit                   Exit
```

Mutations work by just typing requests:
```
raise CR to 12
add fire immunity
give it a multiattack
```

## API Endpoints

```
GET  /health
POST /generate       {"prompt": "..."}
POST /mutate         {"current": {...}, "request": "..."}
POST /encounter      {"prompt": "4 players level 6 hard forest bandits"}
```

## Data Sources

- **5etools bestiary** (2,500+ monsters) from [5etools-mirror-3/5etools-src](https://github.com/5etools-mirror-3/5etools-src)
- **The Monsters Know** (429 tactical posts) from [themonstersknow.com](https://www.themonstersknow.com/)
- **Spells and items** extracted from 5etools data

## License

MIT — built for personal/homebrew use. 5etools data remains under its original license.
