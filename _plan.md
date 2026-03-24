# 🗺️ Project Plan — RAG MLOps on HuggingFace Spaces

## Goal
Build a production-ready PDF question-answering system with:
- Strict no-hallucination policy ("I Don't Know" when unsure)
- Conversation memory with summarization
- CI/CD via GitHub Actions → HuggingFace Spaces
- MLOps monitoring loop (drift, accuracy, latency)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     HuggingFace Space                    │
│                                                         │
│   app.py  (entry point)                                 │
│      │                                                  │
│      ├── Thread 1: FastAPI backend  (port 8000)         │
│      │      backend.py                                  │
│      │      • PDF extraction (PyMuPDF)                  │
│      │      • Embedding (sentence-transformers)         │
│      │      • Vector search (cosine similarity)         │
│      │      • LLM answer (Groq LLaMA3)                  │
│      │      • Memory + summarization                    │
│      │      • Prediction logging                        │
│      │                                                  │
│      └── Main: Gradio frontend     (port 7860)          │
│             frontend.py                                 │
│             • Chat tab (default)                        │
│             • Upload tab                                │
│             • TTS, copy, clear buttons                  │
└─────────────────────────────────────────────────────────┘
```

---

## RAG Pipeline

```
User question
     │
     ▼
Embed question (all-MiniLM-L6-v2)
     │
     ▼
Cosine similarity vs all chunks
     │
     ▼
Top-4 chunks (score > 0.25 threshold)
     │
     ├── No chunks above threshold? → "I Don't Know"
     │
     ▼
Build prompt:
  system: "Answer ONLY from CONTEXT"
  context: top-4 chunks
  history: summarized conversation
  question: user question
     │
     ▼
Groq LLaMA3-8b → answer
     │
     ▼
Log (session_id, question, answer, sources, latency)
     │
     ▼
Return to UI + update memory
```

---

## Memory Management

```
Turn 1:  Q1 → A1  →  summary_1 = summarize([Q1,A1])
Turn 2:  Q2 → A2  →  summary_2 = summarize([Q2,A2]) (includes prior summary)
Turn N:  QN → AN  →  summary_N = summarize([QN,AN])
```

Each turn: full history is summarized into ≤200 tokens, passed as context to next turn.
This keeps the LLM context window under control regardless of conversation length.

---

## MLOps Loop

```
Daily (GitHub Actions cron):
  1. drift_check.py   → fetch /logs/stats
                     → alert if IDK rate > 50%
                     → alert if avg latency > 5s

  2. accuracy_check.py → send canary questions
                      → alert if any returns IDK

On push (GitHub Actions):
  1. pip install -r requirements.txt
  2. pytest tests/
  3. git push to HF Space → auto-rebuild
```

---

## File Structure

```
rag_mlops/
├── app.py              Entry point (starts backend + frontend)
├── backend.py          FastAPI RAG engine
├── frontend.py         Gradio UI
├── requirements.txt    Python dependencies
├── Dockerfile          Container definition
├── README.md           HF Space config + description
├── data/               PDFs go here (persisted in Space)
├── logs/               Prediction logs (JSONL)
├── monitor/
│   ├── drift_check.py      Daily MLOps drift monitor
│   └── accuracy_check.py   Daily canary accuracy check
├── tests/              Unit tests (add yours here)
└── .github/
    └── workflows/
        └── deploy.yml  CI/CD pipeline
```

---

## Technology Choices

| Component        | Choice                   | Why                                      |
|------------------|--------------------------|------------------------------------------|
| PDF extraction   | PyMuPDF (fitz)           | Fast, accurate, no external service      |
| Embedding model  | all-MiniLM-L6-v2         | Lightweight, runs on CPU, good quality   |
| Vector search    | cosine similarity + numpy | No vector DB dependency, simple          |
| LLM              | Groq LLaMA3-8b           | Fast inference, free tier, no GPU needed |
| API framework    | FastAPI                  | Async, auto-docs, type-safe              |
| UI framework     | Gradio 4.x               | Native HF Spaces support                 |
| CI/CD            | GitHub Actions           | Free, integrated with GitHub             |
| Monitoring       | Custom JSONL logs        | No external service required             |

---

## Future Improvements

- [ ] Add ChromaDB or FAISS for larger knowledge bases (>10k chunks)
- [ ] Add reranking (cross-encoder) for better retrieval quality
- [ ] Add user feedback buttons (👍 👎) to collect ground truth labels
- [ ] Use labelled feedback to compute real accuracy metrics
- [ ] Add Evidently AI for richer drift reports
- [ ] Add Slack/email alerting to monitor/alerts.py
- [ ] Add authentication to the Gradio UI
