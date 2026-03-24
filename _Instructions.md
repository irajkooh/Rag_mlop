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

### Add secrets to the HF Space

Go to: **Space → Settings → Variables and secrets → New secret**

| Secret name    | Value                   |
|----------------|-------------------------|
| `GROQ_API_KEY` | Your Groq API key       |

Get a free Groq key at: https://console.groq.com

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

| Secret name | Value                          |
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

Locally: `./chroma_db/` folder — persists between runs.
On HF Space: inside the Docker container — resets on redeploy.

To persist ChromaDB on HF Space, mount a HF Dataset as storage:
```python
# In backend.py, change CHROMA_DIR to a mounted dataset path
CHROMA_DIR = Path("/data/chroma_db")
```

---

## Useful API Endpoints

| Endpoint          | Method | What it does                          |
|-------------------|--------|---------------------------------------|
| `/health`         | GET    | App + ChromaDB status                 |
| `/system/info`    | GET    | Env · Device · LLM (powers status bar)|
| `/kb/status`      | GET    | Files and chunk counts in ChromaDB    |
| `/ask`            | POST   | Ask a question (RAG + memory)         |
| `/upload`         | POST   | Upload and index a PDF                |
| `/reload`         | POST   | Re-index all PDFs in /data            |
| `/session/clear`  | POST   | Clear conversation memory             |
| `/logs/stats`     | GET    | MLOps metrics (IDK rate, latency)     |

---

## Rollback

```bash
git log --oneline
git revert <commit-hash>
git push origin main
```

---

## Add PDFs without redeploying

**Via UI:** Upload tab → drop PDF → Upload & Index

**Via API:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

**Via git** (only needed if you want PDFs baked into the Docker image):
```bash
cp new_doc.pdf data/
git add data/new_doc.pdf
git commit -m "add document"
git push origin main
```
