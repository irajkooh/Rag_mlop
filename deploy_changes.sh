"""
<<<<<<< HEAD
# Run:
# chmod +x /Users/ik/UVcodes/Rag_mlop/deploy_changes.sh


#!/usr/bin/env bash
# Usage: ./deploy_changes.sh "your commit message"
set -e

MSG="${1:-update}"

git add -A
git commit -m "$MSG"
git push origin main

echo "✅ Pushed — GitHub Actions will deploy to HF Space automatically."
=======
"""
#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_changes.sh  —  Push local changes to GitHub + HuggingFace Space
#
# USAGE:
#   chmod +x deploy_changes.sh          # one-time: make it executable
#   ./deploy_changes.sh "your message"  # commit + push to both remotes
#   ./deploy_changes.sh                 # uses default commit message
#
# WHAT IT DOES (in order):
#   1. Stages all modified tracked files  (git add -u)
#   2. Commits with your message
#   3. Pushes to GitHub  (origin  → github.com/irajkooh/AgenticMultimodalRag)
#   4. Pushes to HF Space via a clean orphan branch — binary data files
#      (PDF, PNG, DOCX, vectorstore, tables) are excluded from the Space push because HF Space
#      does not support Git LFS; those files live in the HF Dataset repo
#      irajkoohi/AgenticMultiModalRag_dataset and are downloaded at Space startup.
#
# DATA FILES (persistent across Space restarts):
#   - Add/remove files in data/, vectorstore/, or data/tables/ and run:
#       ./deploy_changes.sh "your commit message"
#   - All persistent data is stored in the HF dataset:
#       https://huggingface.co/datasets/irajkoohi/AgenticMultiModalRag_dataset
#   - The Space will always restore data from this dataset on startup.
#
# NOTES:
#   - Known new source files are staged automatically (see staging section below)
#   - If GitHub push fails with "non-fast-forward", run:
#       git pull --rebase origin main && ./deploy_changes.sh "retry"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MSG="${1:-"chore: update app"}"
RESET_DB=false
for arg in "$@"; do [[ "$arg" == "--reset-db" ]] && RESET_DB=true; done

if $RESET_DB; then
    echo "▶ Clearing stale vectorstore from HF Hub dataset..."
    python3 - <<'PYEOF'
import os, sys, re
token = os.environ.get("AgenticMultiModalRag_Token", "").strip()
if not token:
    # Try loading from _secrets/HF_TOKEN.txt — extract the hf_... token line
    try:
        with open("_secrets/HF_TOKEN.txt") as f:
            for line in f:
                line = line.strip()
                if re.match(r'^hf_[A-Za-z0-9]+$', line):
                    token = line
                    break
    except Exception:
        pass
if not token:
    print("⚠  HF token not found — skipping DB reset")
    sys.exit(0)
from huggingface_hub import HfApi
api = HfApi(token=token)
repo = "irajkoohi/MultiModalRag_dataset"
try:
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("vectorstore/")]
    for f in files:
        api.delete_file(path_in_repo=f, repo_id=repo, repo_type="dataset",
                        commit_message="reset vectorstore")
    print(f"✅  Cleared {len(files)} vectorstore file(s) from HF Hub dataset")
except Exception as e:
    print(f"⚠  DB reset failed: {e}")
PYEOF
fi

echo "▶ Staging modified files..."
git add -u
# Stage new untracked source files (skip if already tracked or missing)
git add Agents/rag_workflow.py Agents/workflow_state.py \
        utils/get_workflow_mermaid.py \
        MCPs/claude_sql_mcp.py \
        packages.txt \
        data/txt_1.txt 2>/dev/null || true

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "✅ Nothing to commit — working tree clean."
else
    echo "▶ Committing: \"$MSG\""
    git commit -m "$MSG"
fi

echo "▶ Pushing to GitHub (origin)..."
git push origin main

# ── Upload committed binary data files to HF Hub dataset ─────────────────────
# PDFs/DOCX/PNGs are excluded from the Space rsync (no Git LFS support).
# Uploading them here ensures sync_from_hf_hub() can download them on Space startup.
echo "▶ Syncing data files to HF Hub dataset (upload new + delete removed)..."
python3 - <<'PYEOF'
import os, sys, re, subprocess
from pathlib import Path

token = os.environ.get("AgenticMultiModalRag_Token", "").strip()
if not token:
    try:
        with open("_secrets/HF_TOKEN.txt") as f:
            for line in f:
                line = line.strip()
                if re.match(r'^hf_[A-Za-z0-9]+$', line):
                    token = line
                    break
    except Exception:
        pass
if not token:
    print("⚠  HF token not found — skipping data file sync to HF Hub")
    sys.exit(0)

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
api = HfApi(token=token)
repo = "irajkoohi/MultiModalRag_dataset"

result = subprocess.run(["git", "ls-files", "data/"], capture_output=True, text=True)
committed = result.stdout.splitlines()

# Top-level data files only (no subdirs like images/ or tables/)
sync_exts = {'.pdf', '.png', '.jpg', '.jpeg', '.docx', '.xlsx', '.txt'}
local_files = [
    f for f in committed
    if Path(f).suffix.lower() in sync_exts and '/' not in f[len("data/"):]
]
local_set = set(local_files)

# Files present on HF Hub dataset under data/ (top-level only)
hub_data_files = [
    f for f in api.list_repo_files(repo, repo_type="dataset")
    if f.startswith("data/") and '/' not in f[len("data/"):]
]

upload_ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f) for f in local_files]
delete_ops = [CommitOperationDelete(path_in_repo=f) for f in hub_data_files if f not in local_set]

all_ops = upload_ops + delete_ops
if not all_ops:
    print("  Data files already in sync — nothing to do.")
    sys.exit(0)

try:
    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=all_ops,
        commit_message="deploy: sync data files",
    )
    if upload_ops:
        print(f"✅  Uploaded {len(upload_ops)} file(s): {[Path(f).name for f in local_files]}")
    if delete_ops:
        to_del = [Path(f).name for f in hub_data_files if f not in local_set]
        print(f"🗑️  Deleted {len(delete_ops)} stale file(s) from HF Hub: {to_del}")
except Exception as e:
    print(f"⚠  HF Hub data sync failed: {e}")
PYEOF

# ── Upload data/tables/ (SQLite DBs) to HF Hub dataset ───────────────────────
echo "▶ Syncing data/tables/ to HF Hub dataset..."
python3 - <<'PYEOF'
import os, sys, re
from pathlib import Path

token = os.environ.get("AgenticMultiModalRag_Token", "").strip()
if not token:
    try:
        with open("_secrets/HF_TOKEN.txt") as f:
            for line in f:
                line = line.strip()
                if re.match(r'^hf_[A-Za-z0-9]+$', line):
                    token = line
                    break
    except Exception:
        pass
if not token:
    print("⚠  HF token not found — skipping tables sync to HF Hub")
    sys.exit(0)

tables_dir = Path("data/tables")
if not tables_dir.exists() or not any(tables_dir.iterdir()):
    print("  data/tables/ is empty — skipping.")
    sys.exit(0)

from huggingface_hub import HfApi
api = HfApi(token=token)
repo = "irajkoohi/MultiModalRag_dataset"
try:
    api.upload_folder(
        folder_path=str(tables_dir),
        path_in_repo="tables",
        repo_id=repo,
        repo_type="dataset",
        commit_message="deploy: sync tables",
        ignore_patterns=["*.lock", ".DS_Store"],
    )
    print(f"✅  Uploaded data/tables/ to HF Hub dataset")
except Exception as e:
    print(f"⚠  Tables sync failed: {e}")
PYEOF

# ── HF Space push via a temp directory (never touches working tree) ──────────
echo "▶ Building clean Space deploy branch (binary files excluded)..."

_tmpdir=$(mktemp -d)
# Copy entire working tree to temp dir, excluding what doesn't belong on Space
rsync -a --exclude='.git' \
         --exclude='data/*.pdf' \
         --exclude='data/*.png' \
         --exclude='data/*.jpg' \
         --exclude='data/*.jpeg' \
         --exclude='data/*.docx' \
         --exclude='data/*.xlsx' \
         --exclude='data/images/' \
         --exclude='data/tables/' \
         --exclude='vectorstore/' \
         --exclude='vectorstore_corrupted_backup/' \
         --exclude='_secrets/' \
         --exclude='.venv/' \
         --exclude='__pycache__/' \
         --exclude='*.pyc' \
         . "$_tmpdir/"

# Build an orphan git repo in the temp dir and push it
pushd "$_tmpdir" > /dev/null
git init -q
git checkout -b space-deploy
git add -A
git commit -q -m "$MSG [space deploy]"
echo "▶ Force-pushing to HuggingFace Space..."
git remote add hf "$(cd - > /dev/null && git remote get-url hf)"
git push hf space-deploy:main --force
popd > /dev/null
rm -rf "$_tmpdir"

echo ""
echo "✅ Deployed successfully!"
echo "   GitHub : https://github.com/irajkooh/AgenticMultimodalRag"
echo "   Space  : https://huggingface.co/spaces/irajkoohi/AgenticMultimodalRag"
echo "   Dataset: https://huggingface.co/datasets/irajkoohi/AgenticMultiModalRag_dataset"
>>>>>>> 635d7d0 (fix: update rag_engine.py implementation)
