FROM python:3.12-slim

WORKDIR /app

# System deps for chromadb / torch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn[standard] fastapi

COPY . .

# ChromaDB data and HuggingFace model cache are volume-mounted
ENV TRANSFORMERS_CACHE=/cache/hf
ENV HF_HOME=/cache/hf

EXPOSE 8765

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"]
