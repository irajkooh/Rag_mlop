FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF + ChromaDB native libs
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Heavy deps — cached independently of requirements.txt ──────────────────
# These rarely change; keeping them in their own layer avoids re-downloading
# torch (~300 MB) on every requirements.txt update.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    sentence-transformers==3.0.1 \
    onnxruntime==1.24.4 \
    pymupdf==1.24.5 \
    gradio==4.44.1

# Pre-bake the embedding model so cold-start is fast
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')"

# ── Light deps — reinstalled when requirements.txt changes ─────────────────
COPY requirements.txt .
RUN grep -vE '^(torch|sentence-transformers|onnxruntime|pymupdf|gradio)==' requirements.txt \
    > /tmp/req_light.txt && \
    pip install --no-cache-dir -r /tmp/req_light.txt

COPY . .

# Persistent directories
RUN mkdir -p data logs chroma_db

EXPOSE 7860

# ── Environment variable reference (set in HF Space Secrets) ──────────────
# GROQ_API_KEY   — Groq API key (used when Ollama is not available)
# OLLAMA_URL     — Ollama server URL (default: http://localhost:11434)
# OLLAMA_MODEL   — model name served by Ollama (default: llama3)
# BACKEND_URL    — URL frontend uses to reach backend (default: http://localhost:8000)

ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]
