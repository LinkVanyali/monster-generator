FROM python:3.12-slim

WORKDIR /app

# System deps for chromadb / torch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn[standard] fastapi

COPY . .

# Download pre-built data bundle (mandatory for cloud deploys)
RUN echo "Downloading data bundle..." && \
    curl -fL https://github.com/LinkVanyali/monster-generator/releases/download/v1.0.0/monster-generator-data.tar.gz -o /tmp/data.tar.gz && \
    tar -xzf /tmp/data.tar.gz && \
    rm /tmp/data.tar.gz && \
    ls -lh chroma_db/ monster_catalog.json monsters_know_posts.json && \
    echo "Data bundle extracted successfully"

# ChromaDB data and HuggingFace model cache
ENV TRANSFORMERS_CACHE=/app/cache/hf
ENV HF_HOME=/app/cache/hf

EXPOSE 8765

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"]
