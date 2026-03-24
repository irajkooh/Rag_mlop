# 🔍 How It Works — Code Explained

---

## app.py — Entry Point

```python
t = threading.Thread(target=run_backend, daemon=True)
t.start()
ui = build_ui()
ui.launch(server_name="0.0.0.0", server_port=7860)
```

HuggingFace Spaces expose one port (7860). We need both FastAPI (8000) and Gradio (7860) in the same container. FastAPI runs in a **daemon thread** (dies when main thread exits), Gradio runs in the main thread and blocks, keeping the process alive.

---

## backend.py — The RAG Engine

### Device Detection

```python
def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return f"cuda:{torch.cuda.get_device_name(0)}"
    return "cpu"
```

Checked once at startup. The result is stored in `DEVICE` and:
- Used to move the embedding model to the right device
- Exposed via `/system/info` so the UI status bar can display it
- Logged alongside every prediction for MLOps tracking

On your Mac with Apple Silicon, this returns `"mps"` — the embedding model then runs on the GPU cores of the M-chip, significantly faster than CPU.

### ChromaDB — Persistent Vector Store

```python
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"},
)
```

ChromaDB stores vectors to disk (`./chroma_db/`). Unlike the previous in-memory numpy approach:
- Data **survives restarts** — you don't re-index PDFs every time the app starts
- Scales to hundreds of thousands of chunks without fitting in RAM
- Uses HNSW (Hierarchical Navigable Small World) index for fast approximate nearest-neighbour search
- `hnsw:space: cosine` tells ChromaDB to use cosine distance (0 = identical, 2 = opposite)

### PDF → Chunks → ChromaDB

```python
def index_pdf(pdf_path):
    chunks     = extract_chunks(pdf_path)           # 500-char overlapping windows
    embeddings = embedder.encode(chunks).tolist()   # shape (N, 384)
    ids        = [f"{fname}::{i}" for i in range(len(chunks))]

    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=...)
```

`upsert` (update + insert) is idempotent — calling it twice on the same PDF replaces the old chunks rather than duplicating them. The IDs are deterministic (`filename::chunk_index`), so re-uploading the same PDF is safe.

### Retrieval from ChromaDB

```python
results = collection.query(
    query_embeddings=q_emb,
    n_results=top_k,
    include=["documents", "metadatas", "distances"],
)
# Convert distance to similarity
score = 1.0 - (dist / 2.0)
```

ChromaDB returns distances (lower = more similar). We convert to similarity scores (higher = more similar) and filter out anything below 0.25. This threshold prevents the LLM from receiving loosely-related noise as context.

### Ollama LLM — Local Inference

```python
def call_ollama(messages, max_tokens=512):
    payload = {
        "model":    OLLAMA_MODEL,     # e.g. "llama3"
        "messages": messages,
        "stream":   False,
        "options":  {"num_predict": max_tokens, "temperature": 0.1},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    return r.json()["message"]["content"].strip()
```

Ollama runs as a local HTTP server (`http://localhost:11434`). The app calls it via REST — same interface as any cloud API, but inference runs entirely on your machine (CPU or GPU). No data leaves your system, no API key, no cost per token.

`temperature: 0.1` keeps answers factual and deterministic — near-zero randomness is correct for document Q&A.

### Ollama → Groq Fallback

```python
def call_llm(messages, max_tokens=512):
    if ollama_available():     # pings /api/tags with 3s timeout
        try:
            return call_ollama(messages, max_tokens)
        except Exception as e:
            logger.warning(f"Ollama failed: {e} — trying Groq")
    return call_groq(messages, max_tokens)
```

On HuggingFace Space, Ollama is not running — `ollama_available()` returns False and the code falls through to Groq automatically. No code change needed between local and cloud deployment — just set `GROQ_API_KEY` as an HF Space secret.

### /system/info — Powers the Status Bar

```python
@app.get("/system/info")
def system_info():
    return {
        "environment": "HuggingFace Space" if IS_HF_SPACE else "Local",
        "device":      DEVICE,
        "llm":         active_llm_label(),   # "Ollama · llama3" or "Groq · llama3-8b"
    }
```

`IS_HF_SPACE` is detected by checking if the `SPACE_ID` environment variable is set — HuggingFace sets this automatically in every Space container. This means no manual config flag is needed.

---

## frontend.py — The Gradio UI

### Status Bar

```python
def build_status_bar() -> str:
    info = get_system_info()   # GET /system/info from backend
    return f"""
    <div class="status-bar">
        <span class="status-pill">🌐 {env}</span>
        <span class="status-pill">🖥️ device: <strong>{dev}</strong></span>
        <span class="status-pill">🤖 LLM: <strong>{llm}</strong></span>
    </div>
    """
```

Called once when the UI loads. The **Refresh Status** button triggers it again so you can see live changes (e.g. if you start Ollama after the app launched). The pills are styled with blue backgrounds via CSS.

### Session State

```python
session_id  = gr.State(str(uuid.uuid4()))
last_answer = gr.State("")
```

`gr.State` holds per-user values invisible to the UI. `session_id` is a UUID linking this browser tab to the backend's conversation memory. `last_answer` stores the most recent assistant reply for TTS — it's updated on every `on_send()` call.

### TTS — Browser SpeechSynthesis

```javascript
(text) => {
    text = text.replace(/[*_`#>~]/g, '').replace(/\n/g, ' ');
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95;
    window.speechSynthesis.cancel();   // stop any current speech
    window.speechSynthesis.speak(u);
}
```

Runs in the browser via Gradio's `js=` parameter — no Python code executes. The markdown symbols (`*`, `_`, `#`) are stripped before speaking so they're not read aloud. `cancel()` before `speak()` means clicking Read again replaces the current reading immediately.

---

## The Full Request Lifecycle

```
1. User types question → presses Send
2. frontend: on_send() called
3. frontend: POST /ask {question, session_id}
4. backend: retrieve_relevant_chunks(question)
       → embed question (MPS/CPU)
       → ChromaDB HNSW search → top-4 chunks
       → filter score < 0.25
5. backend: answer_question(question, chunks, summary)
       → if no chunks: return "I Don't Know" immediately (no LLM call)
       → else: build prompt → call_llm()
              → try Ollama (local) → fallback to Groq (cloud)
6. backend: update session["history"]
7. backend: summarize_history() → compress to ≤200 tokens
8. backend: log_prediction() → append to predictions.jsonl
9. backend: return {answer, sources, latency_ms}
10. frontend: append to chatbot history
11. frontend: store answer in last_answer state (for TTS)
12. Gradio re-renders chat
```

---

## ChromaDB vs In-Memory (previous version)

| | Previous (numpy) | Now (ChromaDB) |
|---|---|---|
| Storage | RAM only — lost on restart | Disk — persists across restarts |
| Upload new PDF | Rebuilds all embeddings | Upserts only new chunks |
| Scale | ~10k chunks max (RAM limit) | Millions of chunks |
| Search | Linear scan O(N) | HNSW approximate O(log N) |
| Duplicate safety | Appends duplicates | Upsert by ID prevents duplicates |


---

## ChromaDB — Persistent Vector Store

```python
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"},
)
```

Unlike the previous in-memory numpy approach, ChromaDB:
- **Persists to disk** in `/chroma_db` — survives restarts, no re-indexing needed
- Uses **HNSW (Hierarchical Navigable Small World)** graph index — sub-linear
  search time even at millions of chunks
- Configured with `cosine` space so distances are comparable across embeddings
- `upsert()` is used instead of `add()` — re-indexing the same PDF twice is
  safe and idempotent (same chunk ID = overwrite, not duplicate)

ChromaDB returns distances in `[0, 2]` for cosine space.
We convert: `similarity = 1.0 - (distance / 2.0)` → result in `[0, 1]`.
Chunks with `similarity < 0.25` are filtered out (irrelevance guard unchanged).

---

## Ollama — Local LLM

```python
def call_ollama(messages, max_tokens=512):
    payload = {
        "model":   OLLAMA_MODEL,     # default: "llama3"
        "messages": messages,
        "stream":  False,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
```

Ollama exposes a local REST API compatible with the OpenAI chat format.
No Python SDK is needed — plain `requests` suffices.

`call_llm()` checks `ollama_available()` (a fast GET to `/api/tags`) before
every call. If Ollama is reachable it is used; otherwise Groq is the fallback.
This means the same codebase works:
- **Locally**: Ollama running on your machine
- **HuggingFace Space**: no Ollama → automatic Groq fallback (set `GROQ_API_KEY`)

---

## Device Detection — MPS → CUDA → CPU

```python
def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"
```

`mps` is Apple Silicon GPU (Metal Performance Shaders).
The check `is_built()` guards against PyTorch builds that expose the API
but weren't compiled with MPS support.

The embedder is then moved to the detected device:
```python
embedder = embedder.to(torch.device("mps"))   # or "cuda"
```

This accelerates batch encoding of PDF chunks at index time.
LLM inference runs inside Ollama's own process — device selection there is
controlled by Ollama's own configuration (it auto-detects GPU as well).

---

## Blue Status Bar — /system/info

```python
@app.get("/system/info")
def system_info():
    return {
        "environment":   "HuggingFace Space" if IS_HF_SPACE else "Local",
        "device":        DEVICE,
        "llm":           active_llm_label(),
        "chroma_chunks": collection.count(),
    }
```

`IS_HF_SPACE` is `True` when the `SPACE_ID` environment variable is set —
HuggingFace injects this automatically into every Space container.

`active_llm_label()` calls `ollama_available()` at request time, so the
label reflects the *current* state — if Ollama goes down mid-session, the
status bar will show "Groq" on the next refresh.

The frontend calls this endpoint once at startup to build the HTML pill bar,
and again whenever the user clicks **🔄 Refresh Status**.
