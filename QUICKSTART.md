# Quick Start Guide (Pre-Built Data)


## Setup

### 1. Download

Get both releases from https://github.com/LinkVanyali/monster-generator/releases/v1.0.0:
- **Source code** (zip or tar.gz)
- **monster-generator-data.tar.gz** (59MB — pre-built ChromaDB + catalog)

### 2. Extract

```bash
# Extract source
unzip monster-generator-main.zip
cd monster-generator-main

# Extract data into the project directory
tar -xzf ../monster-generator-data.tar.gz
```

You should now have:
```
monster-generator-main/
├── docker-compose.yml
├── monster_catalog.json       ← from data bundle
├── monsters_know_posts.json   ← from data bundle
├── chroma_db/                 ← from data bundle
└── ...
```

### 3. Add API Keys

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

Get keys from:
- Anthropic: https://console.anthropic.com/
- HuggingFace: https://huggingface.co/settings/tokens

### 4. Start

```bash
docker-compose up -d
```

### 5. Test

```bash
curl http://localhost:8765/health
# Should return: {"status":"ok","ready":true}
```

Done! The API is live at `http://localhost:8765`.

## Foundry Integration

1. **Add-on Modules → Install Module**
2. Paste: `https://raw.githubusercontent.com/LinkVanyali/monster-generator/main/foundry/module.json`
3. Enable in your world
4. **Settings → AI Monster Generator** → set API URL to `http://localhost:8765`

Click **AI Monster** or **AI Encounter** in the Actors sidebar.

## Notes

- First run downloads ~1.2GB of models (embeddings + reranker) — cached after that
- Cost: ~$0.02-0.05 per monster, ~$0.001 per image
- No manual data processing needed — everything pre-built in the data bundle
