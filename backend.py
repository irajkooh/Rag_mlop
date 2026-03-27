"""
backend.py — FastAPI RAG backend
══════════════════════════════════════════════════════════════════════
Vector store : ChromaDB  (persistent, cosine similarity)
Embeddings   : sentence-transformers/all-MiniLM-L6-v2
               auto-moved to MPS → CUDA → CPU
LLM          : Ollama (primary / local)
               Groq llama-3.1-8b-instant (cloud fallback on HF Space)
Memory       : per-session history + LLM-based summarisation
Logging      : JSONL prediction log for MLOps monitoring
Status API   : /system/info → powers the UI status bar
"""

import json
import logging
import os
import platform
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz                          # PyMuPDF — PDF extraction
import requests as _http             # renamed to avoid shadowing
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
LOGS_DIR   = BASE_DIR / "logs"
CHROMA_DIR = BASE_DIR / "chroma_db"

for _d in [DATA_DIR, LOGS_DIR, CHROMA_DIR]:
    _d.mkdir(exist_ok=True)

# ── Environment ────────────────────────────────────────────────────────────────
OLLAMA_URL       = os.getenv("OLLAMA_URL",       "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "llama3")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY",     "")
GROQ_MODEL       = os.getenv("GROQ_MODEL",       "llama-3.1-8b-instant")
IS_HF_SPACE      = bool(os.getenv("SPACE_ID",    ""))
HF_DATASET_REPO  = os.getenv("HF_DATASET_REPO",  "")   # e.g. "username/my_private_storage"
HF_TOKEN         = os.getenv("HF_TOKEN",         "")   # HF write token
_HF_DATA_FILE    = "chromadb_data.json"

logger.info(f"GROQ_API_KEY set: {bool(GROQ_API_KEY)} | model: {GROQ_MODEL} | HF Space: {IS_HF_SPACE} | HF persistence: {bool(HF_DATASET_REPO)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Device detection — MPS (Apple Silicon) → CUDA → CPU
# ══════════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    """
    Check for MPS (Apple Silicon GPU), then CUDA, then fall back to CPU.
    Returns a short label used both for moving the model and for the UI bar.
    """
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"cuda ({name})"
    except ImportError:
        pass
    return "cpu"


DEVICE = detect_device()
logger.info(f"Compute device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Embedding model
# ══════════════════════════════════════════════════════════════════════════════

logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

_torch_device = DEVICE.split(" ")[0]   # strip "(name)" suffix
if _torch_device in ("mps", "cuda"):
    try:
        import torch
        embedder = embedder.to(torch.device(_torch_device))
        logger.info(f"Embedder running on {_torch_device.upper()}")
    except Exception as _e:
        logger.warning(f"Could not move embedder to {_torch_device}: {_e} — using CPU")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ChromaDB — persistent vector store
# ══════════════════════════════════════════════════════════════════════════════

# On HF Space use an in-memory client so every restart starts clean.
# Locally use PersistentClient so documents survive restarts.
if IS_HF_SPACE:
    chroma_client = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False),
    )
else:
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"},
)
logger.info(f"ChromaDB ready — {collection.count()} chunks already indexed")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Session memory  (in-process)
# ══════════════════════════════════════════════════════════════════════════════

sessions: dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════════════
# 5. FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="RAG MLOps API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# 6. LLM — Ollama primary, Groq fallback
# ══════════════════════════════════════════════════════════════════════════════

def ollama_available() -> bool:
    try:
        r = _http.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_ollama(messages: list[dict], max_tokens: int = 512) -> str:
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"num_predict": max_tokens, "temperature": 0.1},
    }
    r = _http.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def call_groq(messages: list[dict], max_tokens: int = 512) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set and Ollama unavailable.")
    import groq as _groq
    resp = _groq.Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def call_llm(messages: list[dict], max_tokens: int = 512) -> str:
    if ollama_available():
        try:
            return call_ollama(messages, max_tokens)
        except Exception as e:
            logger.warning(f"Ollama error: {e} — falling back to Groq")
    return call_groq(messages, max_tokens)


def active_llm_label() -> str:
    if ollama_available():
        return f"Ollama · {OLLAMA_MODEL}"
    if GROQ_API_KEY:
        return "Groq · llama-3.1-8b-instant"
    return "No LLM configured"


# ══════════════════════════════════════════════════════════════════════════════
# 7. PDF ingestion → ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

def extract_chunks(pdf_path: Path, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    doc        = fitz.open(str(pdf_path))
    pages      = []
    all_links  = []
    for page in doc:
        text  = page.get_text()
        links = [l["uri"] for l in page.get_links() if l.get("uri")]
        if links:
            all_links.extend(links)
        pages.append(text)
    full = "\n".join(pages)
    doc.close()
    chunks, start = [], 0
    while start < len(full):
        end   = min(start + chunk_size, len(full))
        chunk = full[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start += chunk_size - overlap
    # Deduplicate and append a dedicated hyperlinks summary chunk
    seen = []
    for lnk in all_links:
        if lnk not in seen:
            seen.append(lnk)
    if seen:
        link_chunk = (
            f"Hyperlinks and URLs found in {pdf_path.name}:\n"
            + "\n".join(f"• {lnk}" for lnk in seen)
        )
        chunks.append(link_chunk)
    return chunks


def index_pdf(pdf_path: Path) -> int:
    chunks = extract_chunks(pdf_path)
    if not chunks:
        logger.warning(f"No extractable text in {pdf_path.name}")
        return 0
    fname      = pdf_path.name
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids        = [f"{fname}::{i}" for i in range(len(chunks))]
    metadatas  = [{"source": fname, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    logger.info(f"ChromaDB <- {len(chunks)} chunks from {fname}")
    return len(chunks)


def _is_pdf_url(url: str, resp) -> bool:
    """Return True if the response is a PDF."""
    ct = resp.headers.get("Content-Type", "")
    return "application/pdf" in ct or url.lower().split("?")[0].endswith(".pdf")


def fetch_url_chunks(url: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fetch a URL (HTML page or PDF) and return text chunks."""
    import trafilatura
    import tempfile

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    resp = None
    try:
        resp = _http.get(url, headers=headers, timeout=30, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Could not fetch URL: {e}")

    # ── PDF URL ────────────────────────────────────────────────────────────
    if _is_pdf_url(url, resp):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)
        try:
            chunks = extract_chunks(tmp_path, chunk_size=chunk_size, overlap=overlap)
        finally:
            tmp_path.unlink(missing_ok=True)
        if not chunks:
            raise ValueError(f"No extractable text in PDF at {url}")
        chunks.append(f"Source URL: {url}")
        return chunks

    # ── HTML page ─────────────────────────────────────────────────────────
    html = resp.text
    if not html:
        html = trafilatura.fetch_url(url)
    if not html:
        raise ValueError(f"Could not fetch page (blocked or unreachable): {url}")

    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        text = trafilatura.extract(html, favour_recall=True)
    if not text:
        raise ValueError(
            f"No extractable text at this URL — the page likely requires "
            f"JavaScript or login. Try a direct PDF link instead."
        )
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start += chunk_size - overlap
    chunks.append(f"Source URL: {url}")
    return chunks


def _index_single_url(url: str) -> int:
    """Index one URL; returns number of chunks added."""
    from urllib.parse import urlparse
    chunks = fetch_url_chunks(url)
    if not chunks:
        return 0
    parsed      = urlparse(url)
    source_name = (parsed.netloc + parsed.path).strip("/")[:80] or url[:80]
    embeddings  = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids         = [f"{source_name}::{i}" for i in range(len(chunks))]
    metadatas   = [{"source": source_name, "chunk_index": i, "url": url} for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    logger.info(f"ChromaDB <- {len(chunks)} chunks from {url}")
    return len(chunks)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Return same-domain absolute links found in html using the stdlib HTML parser."""
    from urllib.parse import urlparse, urljoin
    from html.parser import HTMLParser

    class _LinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs: list[str] = []
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, val in attrs:
                    if name == "href" and val and not val.startswith(("#", "javascript:", "mailto:")):
                        self.hrefs.append(val)

    parser = _LinkParser()
    parser.feed(html)

    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    seen: set[str] = set()
    links: list[str] = []
    for href in parser.hrefs:
        abs_url = urljoin(base_url, href)
        p = urlparse(abs_url)
        if p.netloc == base_domain and p.scheme in ("http", "https"):
            clean = p.scheme + "://" + p.netloc + p.path.rstrip("/")
            if clean not in seen and clean != base_url.rstrip("/"):
                seen.add(clean)
                links.append(clean)
    return links[:30]   # cap at 30 child links per page


def index_url(url: str, depth: int = 2) -> int:
    """Index url and, if depth > 1, follow same-domain links one level deeper."""
    total = _index_single_url(url)

    if depth > 1:
        # Fetch root HTML to extract child links
        try:
            import requests as _req
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = _req.get(url, headers=headers, timeout=20, allow_redirects=True)
            child_links = _extract_links(resp.text, url) if resp.status_code == 200 else []
        except Exception:
            child_links = []

        seen = {url}
        for child in child_links:
            if child in seen:
                continue
            seen.add(child)
            try:
                total += _index_single_url(child)
            except Exception as e:
                logger.warning(f"Skipped {child}: {e}")

    return total


def build_knowledge_base() -> int:
    total = 0
    for pdf in DATA_DIR.glob("*.pdf"):
        try:
            total += index_pdf(pdf)
        except Exception as e:
            logger.error(f"Failed to index {pdf.name}: {e}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 8. Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_relevant_chunks(query: str, top_k: int = 4) -> list[dict]:
    """
    Embed query → query ChromaDB → convert cosine distance to similarity.
    ChromaDB cosine distance in [0,2]; similarity = 1 - dist/2 in [0,1].
    Filter: similarity < 0.25 discarded as irrelevant.
    """
    n = collection.count()
    if n == 0:
        return []
    q_emb   = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, n),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1.0 - (dist / 2.0)
        if similarity > 0.25:
            chunks.append({
                "text":   doc,
                "source": meta.get("source", "unknown"),
                "url":    meta.get("url"),          # None for PDFs, full URL for web docs
                "score":  round(similarity, 3),
            })
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 9. RAG answer + memory
# ══════════════════════════════════════════════════════════════════════════════

def summarize_history(history: list[dict]) -> str:
    if not history:
        return ""
    convo = "\n".join(f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history)
    try:
        return call_llm(
            [
                {"role": "system",  "content": "Summarise this conversation in 3-5 sentences. Preserve key facts."},
                {"role": "user",    "content": convo},
            ],
            max_tokens=200,
        )
    except Exception as e:
        logger.error(f"Summarisation failed: {e}")
        return convo[-500:]


def answer_question(question: str, chunks: list[dict], summary: str, system_info: dict | None = None) -> str:
    def _src_label(c):
        url = c.get("url")
        return f"[Source: {url if url else c['source']}]"
    context = "\n\n".join(f"{_src_label(c)}\n{c['text']}" for c in chunks)
    history_block = f"\nConversation so far:\n{summary}\n" if summary else ""

    # Build system meta-context so the LLM can answer questions about itself
    if system_info:
        files     = list(system_info.get("files", {}).keys())
        n_chunks  = system_info.get("total_chunks", 0)
        n_files   = len(files)
        file_list = ", ".join(files) if files else "none"
        meta = (
            f"\nSYSTEM INFO (AUTHORITATIVE — always use this for questions about file count, "
            f"document names, or system status; never count from CONTEXT for these):\n"
            f"- You are a RAG-powered document Q&A assistant.\n"
            f"- You can answer questions from uploaded PDF documents using semantic search.\n"
            f"- Exactly {n_files} document(s) are currently indexed ({n_chunks} total chunks).\n"
            f"- Indexed document names: {file_list}.\n"
            f"- You can also summarise, compare, and reason over the document contents.\n"
        )
    else:
        meta = ""

    doc_instruction = (
        "Answer using the CONTEXT section below for questions about document content. "
        "For questions about the system, file count, or document names, "
        "use SYSTEM INFO — it is the authoritative source; do NOT count sources in CONTEXT. "
        "If neither contains the answer, respond with: "
        "\"I Don't Know — the answer is not in the provided documents.\" "
    ) if chunks else (
        "No document context is available for this question. "
        "Answer using SYSTEM INFO if relevant, otherwise respond with: "
        "\"I Don't Know — the answer is not in the provided documents.\" "
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document Q&A assistant. "
                + doc_instruction +
                "Do NOT guess. Be concise and factual. "
                "After your answer, always append a section titled '**Key Points:**' "
                "with 2-4 concise bullet points summarising the most important facts. "
                "Skip the Key Points section only if you answered with I Don't Know."
            ),
        },
        {
            "role": "user",
            "content": f"{meta}{history_block}\nCONTEXT:\n{context}\n\nQUESTION: {question}",
        },
    ]
    try:
        return call_llm(messages, max_tokens=600)
    except Exception as e:
        logger.error(f"LLM answer failed: {e}")
        return "I Don't Know — the language model is currently unavailable."


# ══════════════════════════════════════════════════════════════════════════════
# 10. MLOps logging
# ══════════════════════════════════════════════════════════════════════════════

def log_prediction(session_id, question, answer, chunks, latency_ms):
    record = {
        "id":          str(uuid.uuid4()),
        "timestamp":   datetime.utcnow().isoformat(),
        "session_id":  session_id,
        "question":    question,
        "answer":      answer,
        "top_sources": [c["source"] for c in chunks],
        "top_scores":  [c["score"]  for c in chunks],
        "latency_ms":  round(latency_ms, 1),
        "i_dont_know": answer.startswith("I Don't Know"),
        "llm":         active_llm_label(),
        "device":      DEVICE,
    }
    with open(LOGS_DIR / "predictions.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 11. HF Dataset persistence
# ══════════════════════════════════════════════════════════════════════════════

def load_from_hf_dataset():
    """On Space startup: download chromadb_data.json from HF dataset and upsert into ChromaDB."""
    if not (IS_HF_SPACE and HF_DATASET_REPO and HF_TOKEN):
        return
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=_HF_DATA_FILE,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
        )
        with open(local) as f:
            data = json.load(f)
        if data.get("ids"):
            collection.upsert(
                ids=data["ids"],
                documents=data["documents"],
                embeddings=data["embeddings"],
                metadatas=data["metadatas"],
            )
            logger.info(f"Restored {len(data['ids'])} chunks from HF dataset ({HF_DATASET_REPO})")
    except Exception as e:
        logger.info(f"HF dataset load skipped (first run or error): {e}")


def save_to_hf_dataset():
    """Serialize all ChromaDB chunks to JSON and upload to HF private dataset."""
    if not (IS_HF_SPACE and HF_DATASET_REPO and HF_TOKEN):
        return
    import tempfile
    from huggingface_hub import HfApi
    try:
        data = collection.get(include=["documents", "embeddings", "metadatas"])
        payload = {
            "ids":        data["ids"],
            "documents":  data["documents"],
            "embeddings": data["embeddings"],
            "metadatas":  data["metadatas"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            tmp = Path(f.name)
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=_HF_DATA_FILE,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
        )
        tmp.unlink(missing_ok=True)
        logger.info(f"Persisted {len(data['ids'])} chunks to HF dataset")
    except Exception as e:
        logger.error(f"HF dataset save failed: {e}")


def _save_bg():
    """Fire-and-forget background save to HF dataset."""
    import threading
    threading.Thread(target=save_to_hf_dataset, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# 12. API routes
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    load_from_hf_dataset()   # restore previously indexed data (Space + HF_DATASET_REPO only)
    logger.info(f"Startup complete — ChromaDB: {collection.count()} chunks indexed")
    logger.info(f"LLM: {active_llm_label()} | Device: {DEVICE}")


@app.get("/health")
def health():
    return {"status": "ok", "chroma_chunks": collection.count(), "llm": active_llm_label(), "device": DEVICE}


@app.get("/system/info")
def system_info():
    """Drives the blue status bar in the Gradio UI."""
    return {
        "environment":   "HuggingFace Space" if IS_HF_SPACE else "Local",
        "device":        DEVICE,
        "llm":           active_llm_label(),
        "platform":      platform.system(),
        "chroma_chunks": collection.count(),
    }


@app.get("/kb/status")
def kb_status():
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    files: dict[str, int] = {}
    urls: dict[str, dict] = {}
    for m in all_meta:
        s = m.get("source", "unknown")
        url = m.get("url")
        if url:
            # It's a web doc
            if url not in urls:
                urls[url] = {"source": s, "count": 0}
            urls[url]["count"] += 1
        else:
            # It's a PDF
            files[s] = files.get(s, 0) + 1
    return {
        "total_chunks": collection.count(),
        "files": files,         # PDFs by filename
        "urls": urls           # URLs by full URL
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    dest = DATA_DIR / file.filename
    with open(dest, "wb") as fh:
        fh.write(await file.read())
    n = index_pdf(dest)
    _save_bg()
    return {"message": f"Uploaded and indexed {file.filename}", "chunks_added": n, "total_chunks": collection.count()}


class UrlRequest(BaseModel):
    url: str


@app.post("/upload_url")
def upload_url(req: UrlRequest):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    try:
        n = index_url(url)
        _save_bg()
        return {"message": f"Indexed {url}", "chunks_added": n, "total_chunks": collection.count()}
    except (ValueError, Exception) as e:
        logger.error(f"URL indexing error for {url}: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/reload")
def reload_kb():
    n = build_knowledge_base()
    return {"message": "Reloaded", "indexed": n, "total_chunks": collection.count()}



# Delete by filename (PDF) or by full URL (web doc)
@app.delete("/files/{identifier:path}")
def delete_file(identifier: str):
    # Try as PDF filename first
    ids = collection.get(where={"source": identifier})["ids"]
    if not ids:
        # Try as full URL (for web docs)
        ids = collection.get(where={"url": identifier})["ids"]
    if ids:
        collection.delete(ids=ids)
    logger.info(f"Removed {identifier} from vectorstore ({len(ids)} chunks)")
    _save_bg()
    return {"message": f"Removed {identifier} from index", "chunks_removed": len(ids), "total_chunks": collection.count()}


@app.delete("/files")
def delete_all_files():
    all_ids = collection.get()["ids"]
    batch = 5000
    for i in range(0, len(all_ids), batch):
        collection.delete(ids=all_ids[i:i + batch])
    logger.info("Cleared all chunks from vectorstore")
    _save_bg()
    return {"message": "Vectorstore cleared", "total_chunks": 0}


class QuestionRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None


@app.post("/ask")
def ask(req: QuestionRequest):
    session_id = req.session_id or str(uuid.uuid4())
    question   = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if session_id not in sessions:
        sessions[session_id] = {"history": [], "summary": ""}
    session = sessions[session_id]

    # Space: if ChromaDB is empty, clear any stale session summary so the LLM
    # cannot read old "4 documents indexed" context from a previous session.
    if IS_HF_SPACE and collection.count() == 0:
        session["history"] = []
        session["summary"] = ""

    t0         = time.time()
    chunks     = retrieve_relevant_chunks(question, top_k=8)
    all_meta   = collection.get(include=["metadatas"])["metadatas"]
    sources_map: dict[str, int] = {}
    for m in all_meta:
        s = m.get("source", "unknown")
        sources_map[s] = sources_map.get(s, 0) + 1
    sys_info   = {"total_chunks": collection.count(), "files": sources_map}
    answer     = answer_question(question, chunks, session["summary"], sys_info)
    latency_ms = (time.time() - t0) * 1000

    session["history"].append({"user": question, "assistant": answer})
    session["summary"] = summarize_history(session["history"])
    log_prediction(session_id, question, answer, chunks, latency_ms)

    idk = answer.startswith("I Don't Know")
    return {
        "answer":     answer,
        "session_id": session_id,
        "sources":    [] if idk else list({c["source"] for c in chunks}),
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/session/clear")
def clear_session(req: QuestionRequest):
    sid = req.session_id or ""
    if sid in sessions:
        del sessions[sid]
    return {"message": "Session cleared"}


@app.get("/logs/stats")
def log_stats():
    log_file = LOGS_DIR / "predictions.jsonl"
    if not log_file.exists():
        return {"total": 0, "i_dont_know_rate": 0.0, "avg_latency_ms": 0.0}
    records = []
    with open(log_file) as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        return {"total": 0, "i_dont_know_rate": 0.0, "avg_latency_ms": 0.0}
    idk = sum(1 for r in records if r.get("i_dont_know"))
    return {
        "total":            len(records),
        "i_dont_know_rate": round(idk / len(records), 3),
        "avg_latency_ms":   round(sum(r["latency_ms"] for r in records) / len(records), 1),
        "last_query_at":    records[-1]["timestamp"],
    }
