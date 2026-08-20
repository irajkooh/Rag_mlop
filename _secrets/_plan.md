# Project Plan — MultiModalRag

## Goal

Multi-modal RAG system that answers questions from PDFs, DOCX, XLSX, CSV, TXT, images, and URLs:
- Extracts and indexes text chunks, tables (SQLite), and images (PNG)
- Strict no-hallucination policy ("I Don't Know" when context is absent)
- Conversation memory with rolling summarization
- Persistent storage across HF Space cold starts (HF Dataset: irajkoohi/AgenticMultiModalRag_dataset)
- Deployed on HuggingFace Spaces via `./deploy_changes.sh`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HuggingFace Space                      │
│                                                             │
│   app.py  (entry point)                                     │
│      │                                                      │
│      ├── Background thread: FastAPI backend  (port 8000)    │
│      │      backend.py                                      │
│      │      • Multi-modal doc processing (text+tables+imgs) │
│      │      • URL crawl (2-level deep, same domain)         │
│      │      • Embedding (all-MiniLM-L6-v2)                  │
│      │      • Vector search (ChromaDB HNSW cosine)          │
│      │      • Table SQL queries (in-memory SQLite)          │
│      │      • LLM answer (Groq cloud / Ollama local)        │
│      │      • Conversation memory + summarization           │
│      │      • HF Hub 4-way persistence (via irajkoohi/AgenticMultiModalRag_dataset) │
│      │                                                      │
│      │      Agents/  (LangGraph agentic workflow)           │
│      │      • SupervisorAgent  — thin facade                │
│      │      • RAGWorkflow      — LangGraph StateGraph        │
│      │      • WorkflowState   — TypedDict shared state      │
│      │      • RouterAgent     — table vs doc routing        │
│      │      • SQLGenAgent     — NL→SQL generation           │
│      │      • TableAgent      — SQLite query execution      │
│      │      • DocImageAgent   — vector + image RAG          │
│      │      • GradingAgent    — answer quality grading      │
│      │      • HallucinationAgent — grounding check          │
│      │                                                      │
│      └── Main thread: Gradio frontend  (port 7860)          │
│             frontend.py                                     │
│             • Chat tab (streaming, TTS, copy)               │
│             • Upload tab (auto-index on file select)        │
│             • Documents tab (list + delete per doc)         │
│             • Status bar (env, device, LLM)                 │
│             • Workflow tab (LangGraph mermaid diagram)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Document Processing Pipeline

```
Upload file (PDF / DOCX / XLSX / CSV / TXT / image)
     │
     ├── process_document_chunked()
     │      text chunks (800 chars, 150 overlap)
     │      → embed (all-MiniLM-L6-v2) → ChromaDB upsert
     │
     ├── extract_dataframes()
     │      tables → data/tables/{source}.db (SQLite, one table per t{i})
     │
     └── extract_images()
            images → data/images/{source}/p{page}_i{idx}.png
```

---

## RAG Query Pipeline (LangGraph-driven)

```text
User question
     │
     ▼
WorkflowState initialised {query, memory, n_results, temperature, source_filter}
     │
     ▼
LangGraph StateGraph — node execution:

  [chitchat_detector]
    ├── greeting / meta / docs-list? → answer set, route="chitchat" → END
    ├── no documents indexed?        → answer set, route="chitchat" → END
    └── substantive question         → route="continue"
          │
          ▼
  [router]
    ├── table intent? → route="table"
    └── otherwise    → route="doc"
          │
          ├─ route="table" ──→ [table_agent]
          │                       ├── tables found → answer+sql set → [grader]
          │                       └── no tables   → answer empty   → [doc_image_agent]
          │
          └─ route="doc"  ──→ [doc_image_agent]
                                  → answer+sources+chunks set
                                        │
                                        ▼
                                  [grader]       → grade set
                                        │
                                        ▼
                                  [hallucination_checker] → hallucinated set
                                        │
                                        ▼
                                       END

Return answer + sources + sql_query + answer_method + grade + hallucinated
```

---

## LLM Selection (priority order)

| Priority | Condition | Backend | Model |
|----------|-----------|---------|-------|
| 1 | `GROQ_API_KEY` set | Groq | `llama-3.3-70b-versatile` |
| 2 | `USE_HF_LLM=1` + `HF_TOKEN` set | HF Inference | `meta-llama/Llama-3.1-8B-Instruct` |
| 3 | Ollama reachable | Ollama | `llama3.2` |

---

## Memory Management

Full history is summarized into ≤200 tokens per turn, passed as context to the next turn.
---

## HF Hub Persistence (4-way)

```
After every upload/delete/re-extract: all four are pushed back up.

---

## Keep-Alive

- **Client-side JS**: `fetch('/status')` every 30s — prevents HF from killing the FastAPI process
- **GitHub Actions cron**: pings `https://irajkoohi-multimodalrag.hf.space` every 20 min
- **Weekly heartbeat commit**: Monday 09:00 UTC — prevents Space from going archived

---

## File Structure

```text
MultiModalRag/
├── app.py                Entry point (thread + Gradio launch)
├── backend.py            FastAPI RAG backend
├── frontend.py           Gradio UI
├── deploy_changes.sh     One-command deploy to GitHub + HF Space
├── data/                 Source files (baked into Docker image if committed)
├── vectorstore/          ChromaDB files (gitignored, restored from HF Hub)
├── Agents/
│   ├── workflow_state.py       WorkflowState TypedDict (shared state between all nodes)
│   ├── rag_workflow.py         RAGWorkflow — LangGraph StateGraph orchestrator
│   ├── supervisor_agent.py     SupervisorAgent — thin facade (preserves backend.py API)
│   ├── router_agent.py         RouterAgent — table vs doc routing
│   ├── sql_gen_agent.py        SQLGenAgent — NL→SQL generation via Groq/Ollama LLM
│   ├── table_agent.py          TableAgent — SQLite table query runner + LLM synthesis
│   ├── doc_image_agent.py      DocImageAgent — vector search + image RAG
│   ├── grading_agent.py        GradingAgent — answer quality grading
│   └── hallucination_agent.py  HallucinationAgent — grounding check
├── utils/
│   ├── document_processor.py   Multi-modal extraction, OCR, chunking
│   ├── vector_store.py          ChromaDB VectorStoreManager
│   ├── rag_engine.py            LLM, prompt building, streaming, SQL queries
│   ├── memory.py                ConversationMemory with summarization
│   ├── image_store.py           Per-source image save/list/remove
│   ├── table_store.py           Per-source SQLite table save/load/remove
│   ├── url_processor.py         2-level deep URL crawler
│   ├── get_workflow_mermaid.py  LangGraph stub graph → native mermaid diagram
│   └── device.py                MPS → CUDA → CPU detection
└── .github/
    └── workflows/
        └── keep-alive.yml       20-min ping + weekly heartbeat commit
```

---

## Technology Choices

| Component        | Choice                                             | Why                                          |
|------------------|----------------------------------------------------|----------------------------------------------|
| Agent workflow   | LangGraph StateGraph                               | Graph-driven, state-typed, native mermaid    |
| Text extraction  | PyMuPDF + python-docx                              | Fast, no external service                    |
| Table extraction | camelot (PDF) + pandas                             | Accurate table parsing, SQL-queryable        |
| Image extraction | PyMuPDF page rendering                             | Per-page PNG, works offline                  |
| Embedding        | all-MiniLM-L6-v2                                   | Lightweight, CPU-friendly, good quality      |
| Vector store     | ChromaDB (HNSW cosine)                             | Persistent, fast, no separate server         |
| LLM (cloud)      | Groq llama-3.3-70b                                 | Fast inference, free tier                    |
| LLM (local)      | Ollama llama3.2                                    | No API key, runs on-device                   |
| API framework    | FastAPI                                            | Async, auto-docs, streaming                  |
| UI framework     | Gradio 6.x                                         | Native HF Spaces support                     |
| Persistence      | HF Dataset: irajkoohi/AgenticMultiModalRag_dataset | Survives Space restarts/cold starts          |
| Deploy           | deploy_changes.sh                                  | One command: commit + push to GH + HF Space  |

---

## Potential Improvements

- [ ] Reranking with a cross-encoder for better retrieval precision
- [ ] User feedback buttons (thumbs up/down) to collect ground truth
- [ ] Image captioning / visual Q&A (CLIP or LLaVA)
- [ ] Multi-session support (per-user memory isolation)
- [ ] Authentication on the Gradio UI
- [ ] JavaScript-rendered page support (Playwright)
