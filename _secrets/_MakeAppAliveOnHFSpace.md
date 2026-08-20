# Keep Hugging Face Space Alive for AgenticMultimodalRag

This workflow pings your Hugging Face Space on a schedule to prevent it from sleeping due to inactivity. It also ensures all persistent data is stored in the HF dataset (irajkoohi/AgenticMultiModalRag_dataset), so your app state is always restored on restart or redeploy.

## How it works
- A GitHub Actions workflow runs every 15 minutes and sends a request to your Space URL.
- This keeps the Space active and responsive, even with no users.
- All app data (PDFs, images, vectorstore, tables, etc.) is stored in the HF dataset and restored on Space startup.

## Setup
1. Ensure your app loads and saves all persistent data to the dataset: https://huggingface.co/datasets/irajkoohi/AgenticMultiModalRag_dataset
2. Add the workflow below to `.github/workflows/keep_space_alive.yml` in your repo.
3. Update the `SPACE_URL` if your Space URL changes.

---

```yaml
name: Keep Hugging Face Space Alive

on:
  schedule:
    - cron: '*/15 * * * *'  # every 15 minutes
  workflow_dispatch:

jobs:
  ping-space:
    runs-on: ubuntu-latest
    steps:
      - name: Curl Hugging Face Space
        run: |
          curl -sSf ${{ env.SPACE_URL }} || echo "Space ping failed"
    env:
      SPACE_URL: https://irajkoohi-agenticmultimodalrag.hf.space/
```

---

## Notes
- This workflow does not keep the Space alive 100% of the time (HF may still pause for maintenance), but it prevents idle timeouts.
- All persistent data is always restored from the dataset on startup.
- For best results, keep your data and vectorstore out of the repo and only in the dataset.
