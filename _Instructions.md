# 📋 Instructions — Deploy & Manage

---

## Local Setup

```bash
# 1. Clone your GitHub repo
git clone https://github.com/irajkoohi/rag_mlops
cd rag_mlops

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Ollama (if not already installed)
#    Mac / Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 4. Pull the LLM model
ollama pull llama3

# 5. Start Ollama in a separate terminal (it runs as a local server)
ollama serve
# → Ollama now listening at http://localhost:11434

# 6. (Optional) Put your PDFs in /data before first run
cp your_document.pdf data/

# 7. Run the app
python app.py
# → Frontend: http://localhost:7860
# → Backend:  http://localhost:8000/docs  (FastAPI Swagger)
```

### Changing the Ollama model

```bash
# Pull a different model
ollama pull mistral

# Tell the app to use it (before running app.py)
export OLLAMA_MODEL=mistral

# Or edit the default in backend.py:
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
```

---

## HuggingFace Space Setup

> Ollama cannot run inside an HF Space (no persistent background process).
> The app automatically falls back to Groq when Ollama is unreachable.

### One-time HF setup

1. Go to https://huggingface.co/spaces → **Create new Space**
2. Name: `rag_mlops`, SDK: **Docker**

### Add secrets and variables to the HF Space

Go to: **Space → Settings → Variables and secrets**

**Secrets** (encrypted):

| Secret name    | Value                          |
|----------------|--------------------------------|
| `GROQ_API_KEY` | Your Groq API key              |
| `HF_TOKEN`     | Your HuggingFace write token   |

Get a free Groq key at: https://console.groq.com
Get HF token at: https://huggingface.co/settings/tokens (write scope)

**Variables** (plain text):

| Variable name      | Value                                    |
|--------------------|------------------------------------------|
| `HF_DATASET_REPO`  | e.g. `your-username/my_private_storage`  |

> `HF_DATASET_REPO` enables persistence — indexed documents survive Space restarts/sleeps.
> Create a **private** HF dataset first: https://huggingface.co/new-dataset

### HF Persistence behavior

- If `HF_DATASET_REPO` is set: on startup the app loads previously indexed chunks from the dataset and shows them in the Upload tab. After each upload or delete, data is saved back to the dataset in the background.
- If `HF_DATASET_REPO` is **not** set: the Space wipes all documents on every page load (fresh start each visit).

---

## GitHub → CI/CD → HuggingFace

```bash
# First push
git init
git branch -M main
git remote add origin https://github.com/irajkoohi/rag_mlops.git
git add .
git commit -m "initial commit"
git push origin main
```

Add GitHub secret:

| Secret name | Value                         |
|-------------|-------------------------------|
| `HF_TOKEN`  | Your HuggingFace write token  |

Get token at: https://huggingface.co/settings/tokens (write scope)

After this, every `git push origin main` → tests run → auto-deploys to HF.

---

## Every subsequent deploy

```bash
git add .
git commit -m "describe change"
git push origin main   # CI/CD handles the rest
```

---

## ChromaDB — where is it stored?

**Locally:** `./chroma_db/` folder — persists between runs (PersistentClient).

**On HF Space:** In-memory only (EphemeralClient) — resets on every cold start.
Persistence is handled via HF Dataset (see `HF_DATASET_REPO` above).

---

## Useful API Endpoints

| Endpoint          | Method | What it does                          |
|-------------------|--------|---------------------------------------|
| `/health`         | GET    | App + ChromaDB status                 |
| `/system/info`    | GET    | Env · Device · LLM (powers status bar)|
| `/kb/status`      | GET    | Files and chunk counts in ChromaDB    |
| `/ask`            | POST   | Ask a question (RAG + memory)         |
| `/upload`         | POST   | Upload and index a PDF                |
| `/upload_url`     | POST   | Index a URL (2-level deep crawl)      |
| `/reload`         | POST   | Re-index all PDFs in /data            |
| `/session/clear`  | POST   | Clear conversation memory             |
| `/files/{id}`     | DELETE | Delete a single indexed file          |
| `/files`          | DELETE | Delete all indexed files              |
| `/logs/stats`     | GET    | MLOps metrics (IDK rate, latency)     |

---

## Rollback

```bash
git log --oneline
git revert <commit-hash>
git push origin main
```

---

## Indexing documents

### Upload PDFs

**Via UI:** Upload tab → drop PDF → indexing starts automatically

**Via API:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

**Via git** (only if you want PDFs baked into the Docker image):
```bash
cp new_doc.pdf data/
git add data/new_doc.pdf
git commit -m "add document"
git push origin main
```

### Index a URL

**Via UI:** Upload tab → paste URL → Index URL button

**Via API:**
```bash
curl -X POST http://localhost:8000/upload_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/page"}'
```

- Crawls the page and follows links **2 levels deep** (same domain, up to 30 child pages)
- If the URL points to a **PDF** (by Content-Type or `.pdf` extension), it is downloaded and indexed as a PDF
- JavaScript-rendered pages (e.g. LinkedIn, SPAs) cannot be scraped — only static HTML is supported

---

## Startup hook (SessionStart)

Configured to clear the screen and free ports 8000 and 7860 on startup.
Set in `.claude/settings.json` → `hooks.SessionStart`.
