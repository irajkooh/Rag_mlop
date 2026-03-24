FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF + ChromaDB native libs
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the embedding model so cold-start is fast
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Persistent directories
RUN mkdir -p data logs chroma_db

EXPOSE 7860

# ── Environment variable reference (set in HF Space Secrets) ──────────────
# GROQ_API_KEY   — Groq API key (used when Ollama is not available)
# OLLAMA_URL     — Ollama server URL (default: http://localhost:11434)
# OLLAMA_MODEL   — model name served by Ollama (default: llama3)
# BACKEND_URL    — URL frontend uses to reach backend (default: http://localhost:8000)

CMD ["python", "app.py"]
