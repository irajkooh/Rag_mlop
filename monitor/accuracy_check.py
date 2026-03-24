"""
monitor/accuracy_check.py
─────────────────────────
Sends known canary questions to the live Space and checks that answers
do NOT return "I Don't Know" for questions the documents CAN answer.

How to use:
  1. After your first successful deploy, open your app and find 3-5
     questions your PDFs clearly answer.
  2. Add them to CANARY_QUESTIONS below.
  3. Push — they will be checked every day automatically.

Exits with code 1 on failure so GitHub Actions marks the job red.
"""

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from monitor.alerts import alert_canary_failure, send_alert

SPACE_URL = os.getenv("SPACE_URL", "http://localhost:8000")

# ── Add your canary questions here ────────────────────────────────────────────
# These must be questions your uploaded PDFs can answer.
# Leave empty to skip this check (safe for first deploy).
CANARY_QUESTIONS: list[str] = [
    # "What is the main topic of the uploaded document?",
    # "Who is the author?",
    # "What year was this published?",
]
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not CANARY_QUESTIONS:
        print("ℹ️  No canary questions configured — skipping accuracy check.")
        print("   Edit CANARY_QUESTIONS in monitor/accuracy_check.py to enable this check.")
        return 0

    print(f"🐦 Running {len(CANARY_QUESTIONS)} canary question(s) against {SPACE_URL} …\n")

    failures: list[str] = []
    errors:   list[str] = []

    for q in CANARY_QUESTIONS:
        try:
            r = requests.post(
                f"{SPACE_URL}/ask",
                json={"question": q, "session_id": "monitor-canary"},
                timeout=30,
            )
            r.raise_for_status()
            answer = r.json().get("answer", "")

            if "I Don't Know" in answer:
                failures.append(q)
                print(f"  ❌ IDK — {q}")
            else:
                short = answer[:80].replace("\n", " ")
                print(f"  ✅ OK  — {q[:60]}…\n       ↳ {short}…")

        except Exception as e:
            errors.append(q)
            print(f"  ⚠️  ERROR — {q}: {e}")

    print()
    total    = len(CANARY_QUESTIONS)
    n_failed = len(failures) + len(errors)

    if failures:
        alert_canary_failure(failures)

    if errors:
        send_alert(
            f"{len(errors)} canary request(s) failed with network/HTTP errors.",
            level="warning",
            context={"errored_questions": len(errors)},
        )

    if n_failed == 0:
        send_alert(
            f"Accuracy check passed. All {total} canary questions answered correctly.",
            level="info",
        )
        print(f"✅ All {total} canary questions passed.")
        return 0

    print(f"🚨 {n_failed}/{total} canary questions failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
