FROM python:3.11-slim

# cache bust: 2026-03-28
WORKDIR /app

# ── Heavy deps — cached independently of requirements.txt ──────────────────
# All packages ship pre-compiled manylinux wheels — no gcc/system libs needed.
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

RUN find /app -name "*.pyc" -delete && find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN mkdir -p data logs chroma_db

EXPOSE 7860
ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]
