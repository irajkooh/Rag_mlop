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
