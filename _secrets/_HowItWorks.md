# How It Works — Code Explained

```
Local machine  →  ./deploy_changes.sh  →  GitHub (main)
                                       →  HuggingFace Space (force-push)
                                       →  Data always persisted in HF dataset (irajkoohi/AgenticMultiModalRag_dataset)
```

---

## app.py — Entry Point

```python
api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()
wait_for_api()      # polls /status until it responds (up to 5 min — HF Hub sync can be slow)
ui = build_ui()
ui.queue().launch(server_name="0.0.0.0", server_port=7860)
```

HuggingFace Spaces expose one port (7860). We need both FastAPI (8000) and Gradio (7860).
FastAPI runs as a **background thread**, Gradio runs in the main thread and blocks.

On local dev: also kills stale processes on 8000/7860, starts Ollama if not running, and opens the browser automatically.

---

## backend.py — FastAPI RAG Backend

### Cold-start persistence (HF Spaces)

On module load, before anything else is initialized:

```python
if _IS_HF_SPACE:
     sync_vectorstore_from_hf_hub()   # restore ChromaDB files from irajkoohi/AgenticMultiModalRag_dataset
     sync_from_hf_hub()               # restore data/ source files from dataset
     sync_tables_from_hf_hub()        # restore data/tables/ SQLite DBs from dataset
     sync_images_from_hf_hub()        # restore data/images/ extracted images from dataset
```

Then `VectorStoreManager` is created — ChromaDB loads the restored files and the index is ready without re-uploading anything.

After every upload/delete/re-extract, all four components are pushed back:

```python
push_to_hf_hub(filename)
push_vectorstore_to_hf_hub()
push_tables_to_hf_hub()
push_images_to_hf_hub()
```

### Environment variables

```
AgenticMultiModalRag_Token    — HF write token (Space secret)
MultiModalRag_dataset  — "irajkoohi/AgenticMultiModalRag_dataset" (Space variable)
GROQ_API_KEY           — Groq API key (Space secret)
OLLAMA_MODEL           — default "llama3.2" (optional override)
```

### Document processing pipeline

```
Upload file (PDF / DOCX / XLSX / CSV / TXT / image)
     │
     ├── process_document_chunked()  → text chunks → embed → ChromaDB upsert
     ├── extract_dataframes()         → tables → TableStore (SQLite per source)
     └── extract_images()             → images → ImageStore (PNG per source)
```

**Text**: 800-char chunks with 150-char overlap, embedded with `all-MiniLM-L6-v2`.

**Tables**: Extracted via `camelot` (PDF) or `pandas` (XLSX/CSV). Stored in `data/tables/{source}.db` — one SQLite file per source document, one table per `t{i}` table.

**Images**: Extracted from PDFs page-by-page. Stored in `data/images/{source}/p{page}_i{idx}.png` — per-source subdirectory matching the table layout.

### RAG query pipeline (LangGraph StateGraph)

```
User question
     │
     ▼
WorkflowState initialised — query, memory, n_results, temperature, source_filter
     │
     ▼
[chitchat_detector node]
  ├── greeting / meta / docs-list pattern?  → answer, route="chitchat" → END
  ├── total_chunks() == 0?                  → "no documents" answer   → END
  └── substantive question                  → route="continue"
          │
          ▼
[router node]
  ├── table intent regex matches?  → route="table"
  └── otherwise                   → route="doc"
          │
  ┌───────┴───────┐
  │               │
route="table"   route="doc"
  │               │
  ▼               ▼
[table_agent]  [doc_image_agent]
  │    │              │
  │   (no tables)     │
  │    └──────────────┤
  │                   │
  │ (tables found)    │
  └──────────┐        │
             ▼        ▼
           [grader node]
             │
             ▼
     [hallucination_checker node]
             │
             ▼
           END → answer + sources + sql_query + answer_method + grade + hallucinated
```

Each node returns only the state fields it updates; LangGraph merges them automatically.

### LLM selection (`utils/rag_engine.py`)

**Startup backend** (chosen once at import time):

```text
1. GROQ_API_KEY set          →  Groq llama-3.3-70b-versatile          (BACKEND = "groq")
2. USE_HF_LLM=1 + HF_TOKEN  →  HF Inference Llama-3.1-8B-Instruct    (BACKEND = "hf")
3. Otherwise                 →  Ollama llama3.2                        (BACKEND = "ollama")
```

**Runtime fallback chain** (Groq backend, fires automatically on rate-limit):

```text
1. llama-3.3-70b-versatile  (primary — tighter daily quota)
      ↓ rate-limited
2. llama-3.1-8b-instant     (Groq fallback — separate, larger daily quota)
      ↓ also rate-limited
3. HF Inference API         (if AgenticMultiModalRag_Token is set on Space)
      ↓ credits depleted / unavailable
4. Ollama                   (local only — not running on HF Space)
      ↓ not reachable
5. Error message shown to user
```

On HF Space, Ollama is not running — Groq is used and falls back through the chain above.
On local dev with Ollama running, Ollama is used as the final fallback.

### Source filtering

```python
relevant = [r for r in results if r["distance"] <= RELEVANCE_THRESHOLD]  # 1.2
best_dist = min(r["distance"] for r in relevant)
source_chunks = [r for r in relevant if r["distance"] <= best_dist + 0.3]
sources = list({r["metadata"]["source"] for r in source_chunks})
```

This prevents sources 0.4+ worse than the best match from appearing in the answer.

---

## Agents/ — LangGraph Agentic Workflow

| Module                  | Responsibility                                                    |
|-------------------------|-------------------------------------------------------------------|
| `workflow_state.py`     | `WorkflowState` TypedDict — shared state passed between all nodes |
| `rag_workflow.py`       | `RAGWorkflow` — LangGraph `StateGraph`, all node + edge logic     |
| `supervisor_agent.py`   | `SupervisorAgent` — thin facade preserving `backend.py` API       |
| `router_agent.py`       | `RouterAgent` — regex-based table vs doc routing                  |
| `sql_gen_agent.py`      | `SQLGenAgent` — NL→SQL generation via Groq/Ollama LLM + execution |
| `table_agent.py`        | `TableAgent` — runs SQL against in-memory SQLite + LLM synthesis  |
| `doc_image_agent.py`    | `DocImageAgent` — vector search + image RAG                       |
| `grading_agent.py`      | `GradingAgent` — answer quality grading                           |
| `hallucination_agent.py`| `HallucinationAgent` — checks answer is grounded in context       |

## utils/ — Supporting Modules

| Module                  | Responsibility                                                                         |
|-------------------------|----------------------------------------------------------------------------------------|
| `document_processor.py` | PDF/DOCX/XLSX/CSV/TXT/image text extraction, OCR, chunking, table extraction, image extraction |
| `vector_store.py`       | ChromaDB VectorStoreManager — upsert, query, delete                                   |
| `rag_engine.py`         | RAGEngine — LLM selection, prompt building, streaming, table SQL queries               |
| `memory.py`             | ConversationMemory — per-session history with summarization                            |
| `image_store.py`        | ImageStore — save/list/remove extracted images per source                              |
| `table_store.py`        | TableStore — save/load/remove extracted tables per source (SQLite)                     |
| `url_processor.py`      | URL indexing — 2-level deep crawl, same-domain links                                   |
| `get_workflow_mermaid.py` | Stub LangGraph graph → native mermaid diagram (no agent imports)                     |
| `device.py`             | Device detection — MPS → CUDA → CPU                                                   |

---

## frontend.py — Gradio UI

### Keep-alive (client-side)

```javascript
setInterval(() => fetch('/status'), 30000);   // pings backend every 30s
```

Prevents the FastAPI process from being killed by the HF container runtime while the user is actively using the app.

### Keep-alive (server-side)

`.github/workflows/keep-alive.yml` — GitHub Actions cron every 20 minutes:
```
curl https://irajkoohi-multimodalrag.hf.space
```

Weekly heartbeat commit on Monday 09:00 UTC keeps the Space from being archived.

### Status bar

The status bar pills show environment (Local / HuggingFace Space), compute device (MPS / CPU), and active LLM. Populated by `GET /status` at page load and on **Refresh Status** click.

### TTS — Browser SpeechSynthesis

```javascript
text = text.replace(/[*_`#>~]/g, '').replace(/\n/g, ' ');
const u = new SpeechSynthesisUtterance(text);
window.speechSynthesis.cancel();
window.speechSynthesis.speak(u);
```

Runs in-browser via Gradio `js=` — no Python involved. Markdown symbols stripped before speaking.

---

## Data persistence layout

```
HF Dataset repo (irajkoohi/MultiModalRag_dataset)
├── data/            source files uploaded by users (PDFs, DOCX, etc.)
├── vectorstore/     ChromaDB files (chroma.sqlite3 + segment files)
├── tables/          SQLite DBs — one per source document
└── images/          extracted PNG images — one subdirectory per source document

Local (gitignored — restored from HF Hub on cold start)
├── vectorstore/
├── data/tables/
└── data/images/
```

---

## Full request lifecycle

```text
1.  User types question → Send
2.  frontend: POST /ask {question, session_id}
3.  backend: SupervisorAgent.handle() → RAGWorkflow.invoke()
4.  LangGraph: initialise WorkflowState {query, memory, n_results, …}
5.  Node chitchat_detector: greeting / meta / no-docs? → answer + END
6.  Node router: table intent? → route="table", else route="doc"
7a. route="table"  → Node table_agent: SQL query via in-memory SQLite
                       → tables found? → answer+sql set, continue to grader
                       → no tables?   → fall through to doc_image_agent
7b. route="doc"    → Node doc_image_agent:
                       embed question → ChromaDB search → filter chunks
                       → build prompt → LLM (Groq / Ollama) → answer+sources
8.  Node grader: grade answer quality → grade="PASS"|"FAIL"
9.  Node hallucination_checker: grounding check → hallucinated=True|False
10. WorkflowState final → return answer + sources + metadata
11. backend: stream answer to frontend
12. frontend: render streamed answer + sources
```
