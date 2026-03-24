"""
monitor/drift_check.py
──────────────────────
Fetches /logs/stats from the live Space and checks for:
  • High "I Don't Know" rate  → proxy for data / topic drift
  • High average latency      → proxy for performance degradation

Exits with code 1 on threshold breach so GitHub Actions marks the job red.
Alerts are sent via monitor/alerts.py (Slack + stdout).
"""

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from monitor.alerts import alert_idk_rate, alert_high_latency, send_alert

SPACE_URL         = os.getenv("SPACE_URL", "http://localhost:8000")
IDK_THRESHOLD     = float(os.getenv("IDK_THRESHOLD",     "0.5"))
LATENCY_THRESHOLD = float(os.getenv("LATENCY_THRESHOLD", "5000"))


def main() -> int:
    print(f"🔍 Fetching stats from {SPACE_URL}/logs/stats …")

    try:
        r = requests.get(f"{SPACE_URL}/logs/stats", timeout=20)
        r.raise_for_status()
        stats = r.json()
    except Exception as e:
        print(f"⚠️  Could not reach Space: {e}  (skipping drift check)")
        return 0

    total   = stats.get("total", 0)
    idk     = stats.get("i_dont_know_rate", 0.0)
    latency = stats.get("avg_latency_ms", 0.0)
    last_q  = stats.get("last_query_at", "—")

    print(f"📊  Total queries      : {total}")
    print(f"❓  I-Don't-Know rate  : {idk:.1%}")
    print(f"⏱️   Avg latency        : {latency:.0f} ms")
    print(f"🕐  Last query at      : {last_q}")

    issues = 0

    if total >= 10 and idk > IDK_THRESHOLD:
        alert_idk_rate(idk, IDK_THRESHOLD, total)
        issues += 1

    if latency > LATENCY_THRESHOLD:
        alert_high_latency(latency, LATENCY_THRESHOLD)
        issues += 1

    if issues == 0:
        send_alert(
            f"Drift check passed. IDK={idk:.1%}, latency={latency:.0f}ms, queries={total}",
            level="info",
        )
        print("✅  All metrics within thresholds")

    return 1 if issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
