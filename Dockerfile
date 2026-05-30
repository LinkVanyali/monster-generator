FROM python:3.12-slim

WORKDIR /app

# System deps for chromadb / torch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn[standard] fastapi

COPY . .

# Download pre-built data bundle if not present (for Railway/cloud deploys)
RUN if [ ! -f monster_catalog.json ]; then \
        echo "Downloading data bundle..." && \
        curl -L https://github.com/LinkVanyali/monster-generator/releases/download/v1.0.0/monster-generator-data.tar.gz | tar xz; \
    fi

# ChromaDB data and HuggingFace model cache
ENV TRANSFORMERS_CACHE=/app/cache/hf
ENV HF_HOME=/app/cache/hf

EXPOSE 8765

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"]
