# Deployment Guide

## Prerequisites

- Docker + Docker Compose
- Anthropic API key (for generation)
- HuggingFace API key (for image generation)
- ~5GB disk space for data + ChromaDB

## Initial Setup (One-Time)

### 1. Clone the repository

```bash
git clone https://github.com/LinkVanyali/monster-generator.git
cd monster-generator
```

### 2. Get API keys

- **Anthropic**: https://console.anthropic.com/
- **HuggingFace**: https://huggingface.co/settings/tokens

### 3. Create `.env` file

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

### 4. Download source data

This downloads the 5etools bestiary (2,500+ monsters) and The Monsters Know blog posts (429 tactical guides):

```bash
# Download 5etools bestiary data (~200MB)
curl -L https://github.com/5etools-mirror-3/5etools-src/archive/refs/tags/v2.28.3.tar.gz | tar xz
mv 5etools-src-2.28.3 5etools-src

# Fetch The Monsters Know blog posts
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python fetch_monsters_know.py
```

### 5. Build the monster catalog and ChromaDB collections

This processes the 5etools data and embeds everything into ChromaDB (~10-15 minutes):

```bash
python build_catalog.py
python embed_blog_posts.py
python embed_game_data.py  # Spells + items (optional but recommended)
```

You should now have:
- `monster_catalog.json` (processed bestiary)
- `monsters_know_posts.json` (blog posts)
- `chroma_db/` directory with 4 collections

### 6. Start the Docker container

```bash
docker-compose up -d
```

The API will be available at `http://localhost:8765`.

### 7. Test the API

```bash
curl http://localhost:8765/health
# Should return: {"status":"ok","ready":true}
```

## Foundry VTT Integration

### Install the module

1. In Foundry: **Add-on Modules → Install Module**
2. Paste manifest URL:
   ```
   https://raw.githubusercontent.com/LinkVanyali/monster-generator/main/foundry/module.json
   ```
3. Enable the module in your world

### Configure

1. **Settings → Configure Settings → AI Monster Generator**
2. Set API URL to `http://YOUR_SERVER_IP:8765`
   - If Foundry and the API are on the same machine: `http://localhost:8765`
   - If remote: use the server's LAN or public IP

### Usage

Two new buttons appear in the **Actors** sidebar:

- **AI Monster**: Generate a single creature from a concept
- **AI Encounter**: Build a balanced encounter for your party

## Maintenance

### View logs

```bash
docker-compose logs -f monster-generator
```

### Restart

```bash
docker-compose restart
```

### Stop

```bash
docker-compose down
```

### Update to latest version

```bash
git pull
docker-compose down
docker-compose up -d --build
```

## Troubleshooting

### "Generator not ready" error

The backend is still loading. Wait 30-60 seconds after `docker-compose up` for the models to download and ChromaDB to initialize.

### "No module manifest found" in Foundry

The repository must be public for Foundry to fetch the manifest. Check at https://github.com/LinkVanyali/monster-generator

### ChromaDB collections missing

If you see warnings about missing collections, run the embedding scripts:

```bash
python embed_blog_posts.py
python embed_game_data.py
```

Then restart: `docker-compose restart`

### Image generation fails

Ensure `HF_TOKEN` is set in `.env` and you have credits at https://huggingface.co/settings/billing

## System Requirements

- **CPU**: 2+ cores recommended (embedding + reranker are CPU-bound)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 5GB (2GB for ChromaDB, 2GB for embeddings model cache, 1GB for data)
- **Network**: Outbound HTTPS for Anthropic/HuggingFace APIs

## Cost Estimate

- **Anthropic API**: ~$0.02-0.05 per monster (Opus 4.8 generation + Haiku intent parsing)
- **HuggingFace**: ~$0.001 per token image (FLUX.1-schnell)
- **Encounter builder**: ~$0.01 per encounter (Haiku only, no generation)

Free tier sufficient for personal use (~50-100 monsters/month).
