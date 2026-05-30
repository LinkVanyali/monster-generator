"""
Token image generation via Hugging Face Inference API.

Uses FLUX.1-schnell by default — fast, ~$0.001/image, deducted from your
HF monthly credits ($0.10 free/month ≈ 100 images).

Set HF_TOKEN in .env. Optionally override the model:
    IMAGE_MODEL=black-forest-labs/FLUX.1-dev   # slower, higher quality
"""

import base64
import io
import os

from huggingface_hub import InferenceClient
from PIL import Image

from schemas import TargetMonsterOutput

TOKEN_SIZE   = 512
IMAGE_MODEL  = os.environ.get("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")

STYLE_SUFFIX = (
    "fantasy RPG creature token, dramatic lighting, dark neutral background, "
    "detailed digital painting, square composition, no text, no UI, no border"
)


def _build_prompt(monster: TargetMonsterOutput) -> str:
    appearance = getattr(monster, "appearance", None)
    if appearance:
        subject = appearance
    else:
        subject = (
            f"A {monster.size.lower()} {monster.monster_type.lower()} "
            f"creature named {monster.name}"
        )
    return f"{subject}. {STYLE_SUFFIX}"


def generate_token(
    monster: TargetMonsterOutput,
    api_key: str | None = None,
) -> str:
    """
    Generate a token image via HF Inference API.
    Returns base64-encoded WebP string (no data URI prefix).
    """
    token  = api_key or os.environ.get("HF_TOKEN")
    client = InferenceClient(token=token)
    prompt = _build_prompt(monster)

    image = client.text_to_image(prompt, model=IMAGE_MODEL)

    # Resize to standard token size and convert to WebP
    image = image.convert("RGBA").resize((TOKEN_SIZE, TOKEN_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
