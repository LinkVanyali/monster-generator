"""
FastAPI backend for the D&D 5e Monster Generator.
Exposes the generator as a REST API for the Foundry VTT module.

Endpoints:
    GET  /health
    POST /generate       {prompt: str}
    POST /mutate         {current: dict, request: str}
    POST /encounter      {prompt: str}

Run:
    uvicorn api:app --host 0.0.0.0 --port 8765
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from generator import MonsterGenerator
from encounter import build_encounter
from foundry_actor import to_dnd5e_actor
from schemas import TargetMonsterOutput
from image_generator import generate_token


# ---------------------------------------------------------------------------
# App lifecycle — load generator once at startup
# ---------------------------------------------------------------------------

_gen: Optional[MonsterGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gen
    print("Loading MonsterGenerator …")
    _gen = MonsterGenerator()
    print("Ready.")
    yield
    _gen = None


app = FastAPI(title="D&D 5e Monster Generator API", lifespan=lifespan)

# Allow requests from Foundry (same-origin on the home server, but be permissive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str

class MutateRequest(BaseModel):
    current: dict
    request: str

class EncounterRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "ready": _gen is not None}


@app.post("/generate")
def generate(req: GenerateRequest):
    if _gen is None:
        raise HTTPException(503, "Generator not ready")
    try:
        monster = _gen.generate(req.prompt)

        # Generate token image (base64)
        token_b64 = None
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                token_b64 = generate_token(monster, api_key=hf_token)
        except Exception as img_err:
            print(f"Token generation failed: {img_err}")

        return {
            "actor":   to_dnd5e_actor(monster),  # Module will set image after upload
            "monster": monster.model_dump(),
            "token":   token_b64,  # Base64 for module to upload
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/mutate")
def mutate(req: MutateRequest):
    if _gen is None:
        raise HTTPException(503, "Generator not ready")
    try:
        current = TargetMonsterOutput(**req.current)
        monster = _gen.mutate(current, req.request)

        # Generate token image
        token_b64 = None
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                token_b64 = generate_token(monster, api_key=hf_token)
        except Exception as img_err:
            print(f"Token generation failed: {img_err}")

        return {
            "actor":   to_dnd5e_actor(monster),
            "monster": monster.model_dump(),
            "token":   token_b64,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/encounter")
def encounter(req: EncounterRequest):
    if _gen is None:
        raise HTTPException(503, "Generator not ready")
    try:
        result = build_encounter(req.prompt, _gen.monsters, _gen.client)
        return {
            "theme":       result.theme,
            "party_size":  result.party_size,
            "party_level": result.party_level,
            "difficulty":  result.difficulty,
            "xp_budget":   result.xp_budget,
            "adjusted_xp": result.adjusted,
            "gm_note":     result.gm_note,
            "monsters":    [
                {
                    "name":   m.name,
                    "cr":     m.cr,
                    "count":  m.count,
                    "role":   m.role,
                    "tactic": m.tactic,
                    "hp":     m.hp,
                    "ac":     m.ac,
                }
                for m in result.monsters
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
