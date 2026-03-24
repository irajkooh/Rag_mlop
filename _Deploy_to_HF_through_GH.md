# Deploy to HuggingFace Spaces via GitHub

## How it works

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

| What | Value |
|------|-------|
| GitHub repo | https://github.com/irajkooh/Rag_mlop |
| HF Space | https://huggingface.co/spaces/irajkoohi/Rag_mlop |
| Workflow file | `.github/workflows/deploy.yml` |
| GitHub secret | `HF_TOKEN` = your HF user access token |

### GitHub secret required
Go to **github.com/irajkooh/Rag_mlop/settings/secrets/actions** and ensure:
- Name: `HF_TOKEN`
- Value: your HuggingFace token (starts with `hf_`)

Get/renew your token at **huggingface.co/settings/tokens** — needs **write** access.

### GitHub PAT required (for pushing workflow files)
Your GitHub Personal Access Token needs the **`workflow`** scope.
Go to **github.com/settings/tokens** → Edit → check ☑️ `workflow` → Update token.

---

## Daily workflow — deploy a change

```bash
# 1. Make your changes locally, then:
git add <files>
git commit -m "your message"
git push

# 2. Watch the pipeline:
#    https://github.com/irajkooh/Rag_mlop/actions
#    Green ✅ = deployed to HF Spaces
#    Red ❌ = check the failed step

# 3. Once green, HF builds Docker (~3-5 min first time, ~1 min after):
#    https://huggingface.co/spaces/irajkoohi/Rag_mlop
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

Set these in **huggingface.co/spaces/irajkoohi/Rag_mlop/settings**:

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
| `HF_TOKEN secret is empty` | Re-add the secret in GitHub repo Settings → Secrets → Actions |
| `workflow` scope error on push | Edit your GitHub PAT and check the `workflow` scope |
| `colorFrom` invalid | README.md `colorFrom` must be: red, yellow, green, blue, indigo, purple, pink, gray |
| `401 Unauthorized` on HF upload | Token expired or wrong — renew at huggingface.co/settings/tokens |
| Docker build fails on HF | Check **huggingface.co/spaces/irajkoohi/Rag_mlop** → Logs tab |
| Tests fail in CI | Run `pytest tests/ -v` locally first to reproduce |

---

## Re-trigger deployment without a code change

```bash
git commit --allow-empty -m "ci: trigger deploy" && git push
```
