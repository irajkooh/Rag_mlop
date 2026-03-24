# Deploy to HuggingFace Spaces via GitHub

# Required links:

- GitHub Secrets — add HF_TOKEN, SLACK_WEBHOOK_URL
https://github.com/irajkooh/Rag_mlop/settings/secrets/actions

- HF Tokens — create/renew HF write token
https://huggingface.co/settings/tokens

- GitHub PAT — ensure workflow scope
https://github.com/settings/tokens

- HF Space Settings — add GROQ_API_KEY
https://huggingface.co/spaces/irajkoohi/Rag_mlop/settings

- GitHub Actions — monitor runs
https://github.com/irajkooh/Rag_mlop/actions

- HF Space — live app + logs
https://huggingface.co/spaces/irajkoohi/Rag_mlop?logs=container


###### How it works #####

```
Local machine  →  git push  →  GitHub (main)  →  GitHub Actions CI/CD  →  HuggingFace Space
```

Every push to `main` automatically:
1. Installs dependencies
2. Runs unit tests (`tests/`)
3. Uploads code to HF Spaces via `huggingface_hub` API
4. HF Spaces builds the Docker image and restarts the app

---

## One-time setup (already done)

| What | Where |
|------|-------|
| GitHub repo | [github.com/irajkooh/Rag_mlop](https://github.com/irajkooh/Rag_mlop) |
| HF Space | [huggingface.co/spaces/irajkoohi/Rag_mlop](https://huggingface.co/spaces/irajkoohi/Rag_mlop) |
| Workflow file | `.github/workflows/deploy.yml` |
| GitHub secret | `HF_TOKEN` = your HF user access token |

### GitHub secret required
Go to **[GitHub → Secrets → Actions](https://github.com/irajkooh/Rag_mlop/settings/secrets/actions)** and ensure:
- Name: `HF_TOKEN`
- Value: your HuggingFace token (starts with `hf_`)

Get/renew your token at **[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)** — needs **write** access.

### GitHub PAT required (for pushing workflow files)
Your GitHub Personal Access Token needs the **`workflow`** scope.
Go to **[github.com/settings/tokens](https://github.com/settings/tokens)** → Edit → check ☑️ `workflow` → Update token.

---

## Daily workflow — deploy a change

```bash
# 1. Make your changes locally, then:
git add <files>
git commit -m "your message"
git push

# 2. Watch the pipeline:
#    github.com/irajkooh/Rag_mlop/actions
#    Green ✅ = deployed to HF Spaces
#    Red ❌ = check the failed step

# 3. Once green, HF builds Docker (~3-5 min first time, ~1 min after):
#    huggingface.co/spaces/irajkoohi/Rag_mlop
```

---

## What is NOT committed (`.gitignore`)

| Excluded | Reason |
|----------|--------|
| `.venv/` | Local Python environment |
| `chroma_db/` | Vector store — rebuilt when PDFs are uploaded via UI |
| `data/*.pdf` | Your personal documents — upload through the app UI |
| `logs/` | Runtime logs |
| `.GROQ_API_KEY.txt` | Secret — set in HF Space Secrets instead |
| `.HF_TOKEN.txt` | Secret — never commit tokens |

---

## HF Space environment variables (Secrets)

Set these in **[HF Space Settings](https://huggingface.co/spaces/irajkoohi/Rag_mlop/settings)**:

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key — fallback LLM when Ollama is unavailable |
| `OLLAMA_URL` | Optional — only if you run Ollama externally |

---

## CI/CD pipeline details (`.github/workflows/deploy.yml`)

```
Trigger: push to main  OR  daily cron at 00:00 UTC

Job 1 — deploy (on push):
  ├── Checkout code
  ├── Set up Python 3.11
  ├── Cache pip deps
  ├── Install deps (CPU-only torch + requirements.txt)
  ├── Run unit tests (tests/)
  ├── Push to HF Space via huggingface_hub API
  ├── Notify success (Slack — optional)
  └── Notify failure (Slack — optional)

Job 2 — monitor (daily cron):
  ├── Run drift detection (monitor/drift_check.py)
  └── Run accuracy/canary check (monitor/accuracy_check.py)
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `HF_TOKEN secret is empty` | Re-add at [GitHub Secrets](https://github.com/irajkooh/Rag_mlop/settings/secrets/actions) |
| `workflow` scope error on push | Edit PAT at [github.com/settings/tokens](https://github.com/settings/tokens) → check `workflow` |
| `colorFrom` invalid | README.md `colorFrom` must be: red, yellow, green, blue, indigo, purple, pink, gray |
| `401 Unauthorized` on HF upload | Renew token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Docker build fails on HF | Check [HF Space Logs](https://huggingface.co/spaces/irajkoohi/Rag_mlop) → Logs tab |
| Tests fail in CI | Run `pytest tests/ -v` locally first to reproduce |

---

## Re-trigger deployment without a code change

```bash
git commit --allow-empty -m "ci: trigger deploy" && git push
```

---

## MLOps monitoring — what triggers and what to do

The daily cron job (00:00 UTC) runs two checks against the live HF Space.
Results appear in **[GitHub Actions](https://github.com/irajkooh/Rag_mlop/actions)** under the `monitor` job.

### Check 1 — Drift detection (`monitor/drift_check.py`)

Hits `/logs/stats` on the live Space and checks:

| Metric | Threshold | Alert level |
|--------|-----------|-------------|
| I-Don't-Know rate | > 50% (of last N queries) | 🚨 critical |
| Average latency | > 5000 ms | ⚠️ warning |

**What it means and what to do:**

| Alert | Likely cause | Action |
|-------|-------------|--------|
| High IDK rate | Users are asking about topics not in your documents | Upload more/updated PDFs via the app UI, then re-index |
| High IDK rate | Documents are outdated | Delete old files from the index, upload new versions |
| High latency | Groq API quota hit | Check your Groq dashboard; the free tier has rate limits |
| High latency | HF Space sleeping (free tier sleeps after 48h inactivity) | Open the Space URL to wake it; upgrade to paid HF tier to disable sleeping |

### Check 2 — Accuracy / canary check (`monitor/accuracy_check.py`)

Sends hardcoded questions to `/ask` and fails if any return "I Don't Know".
**These questions must be ones your uploaded PDFs can answer.**

**How to set up canary questions (one-time after first deploy):**

1. Open your live Space, ask 3–5 questions you know your documents answer
2. Edit `monitor/accuracy_check.py` and add them:
   ```python
   CANARY_QUESTIONS: list[str] = [
       "What is Iraj's email address?",
       "How many years of experience does Iraj have?",
       "What LLM frameworks are mentioned?",
   ]
   ```
3. Commit and push — they run every day automatically

**What it means and what to do:**

| Alert | Likely cause | Action |
|-------|-------------|--------|
| Canary question returned IDK | Knowledge base was cleared/reset on HF (ephemeral storage) | Re-upload your PDFs via the app UI |
| Canary question returned IDK | PDF content changed and answer moved | Update the canary question or re-upload the new PDF |
| Network error reaching Space | HF Space is sleeping or crashed | Wake the Space manually; check HF Logs tab |

### Retraining equivalent for a RAG app

This app has no trainable model weights — "retraining" means **refreshing the knowledge base**:

```
Trigger (IDK rate too high or canary fail)
    ↓
1. Go to [huggingface.co/spaces/irajkoohi/Rag_mlop](https://huggingface.co/spaces/irajkoohi/Rag_mlop)
2. Open the Upload Documents tab
3. Delete outdated files (Delete Selected or Delete All)
4. Upload new/updated PDFs → they are re-indexed immediately
5. Ask a test question to verify
6. Monitor job will pass on next daily run (or re-run manually)
```

### Alerts channel — Slack (optional)

Add a Slack webhook to get alerts in a channel instead of only in GitHub Actions logs:

1. Create an incoming webhook at **[api.slack.com/apps](https://api.slack.com/apps)**
2. Add it as a GitHub secret at [GitHub Secrets](https://github.com/irajkooh/Rag_mlop/settings/secrets/actions): name `SLACK_WEBHOOK_URL`, value = the webhook URL
3. Alerts are sent at three levels: `ℹ️ info`, `⚠️ warning`, `🚨 critical`

### Run monitoring manually

```bash
# From GitHub Actions UI:
# Actions → CI/CD → HuggingFace Space → Run workflow → select main

# Or trigger from terminal:
git commit --allow-empty -m "ci: run monitoring" && git push
# (The cron job only runs on schedule, not on push — to test locally:)
SPACE_URL=https://irajkoohi-rag-mlop.hf.space python -m monitor.drift_check
SPACE_URL=https://irajkoohi-rag-mlop.hf.space python -m monitor.accuracy_check
```
