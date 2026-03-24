"""
monitor/alerts.py
─────────────────
Centralised alerting for the MLOps monitoring loop.
Supports Slack webhooks and plain stdout (always on).

Usage:
    from monitor.alerts import send_alert
    send_alert("IDK rate exceeded 50%", level="critical")

Environment variables:
    SLACK_WEBHOOK_URL   — optional Slack incoming webhook
    ALERT_EMAIL         — placeholder for future email alerting
"""

import os
import json
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Emoji per level for readability in Slack / logs
LEVEL_EMOJI = {
    "info":     "ℹ️",
    "warning":  "⚠️",
    "critical": "🚨",
}


def send_alert(message: str, level: str = "warning", context: dict | None = None) -> bool:
    """
    Send an alert to all configured channels.

    Args:
        message:  Human-readable description of the issue.
        level:    One of "info", "warning", "critical".
        context:  Optional dict of extra key/value pairs to include.

    Returns:
        True if at least one channel succeeded, False otherwise.
    """
    level   = level.lower()
    emoji   = LEVEL_EMOJI.get(level, "❓")
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    app     = "rag_mlops @ irajkoohi"

    # ── Always log to stdout (visible in GitHub Actions logs) ───────────────
    log_line = f"[{level.upper()}] {emoji} {app} | {ts} | {message}"
    if context:
        log_line += " | " + " | ".join(f"{k}={v}" for k, v in context.items())
    print(log_line)

    # ── Slack (optional) ────────────────────────────────────────────────────
    slack_ok = True
    if SLACK_WEBHOOK_URL:
        slack_ok = _send_slack(message, level, emoji, ts, app, context or {})

    return slack_ok


def _send_slack(
    message: str,
    level: str,
    emoji: str,
    ts: str,
    app: str,
    context: dict,
) -> bool:
    """Send a formatted message to a Slack incoming webhook."""
    color_map = {"info": "#36a64f", "warning": "#ff9900", "critical": "#cc0000"}
    color = color_map.get(level, "#888888")

    fields = [{"title": k, "value": str(v), "short": True} for k, v in context.items()]

    payload = {
        "attachments": [
            {
                "color":  color,
                "title":  f"{emoji}  {level.upper()} — {app}",
                "text":   message,
                "fields": fields,
                "footer": ts,
            }
        ]
    }

    try:
        r = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code != 200:
            logger.error(f"Slack webhook returned {r.status_code}: {r.text}")
            return False
        return True
    except Exception as e:
        logger.error(f"Slack webhook failed: {e}")
        return False


def alert_idk_rate(idk_rate: float, threshold: float, total: int):
    send_alert(
        f"I-Don't-Know rate is {idk_rate:.1%}, exceeding threshold of {threshold:.0%}. "
        "Consider uploading more documents or reindexing the knowledge base.",
        level="critical",
        context={"idk_rate": f"{idk_rate:.1%}", "threshold": f"{threshold:.0%}", "total_queries": total},
    )


def alert_high_latency(avg_ms: float, threshold_ms: float):
    send_alert(
        f"Average response latency is {avg_ms:.0f}ms, exceeding threshold of {threshold_ms:.0f}ms. "
        "Check Groq API quota or embedding model performance.",
        level="warning",
        context={"avg_latency_ms": f"{avg_ms:.0f}", "threshold_ms": f"{threshold_ms:.0f}"},
    )


def alert_canary_failure(failed_questions: list[str]):
    questions_str = "\n• ".join(failed_questions)
    send_alert(
        f"{len(failed_questions)} canary question(s) returned 'I Don't Know' — "
        f"these were previously answerable:\n• {questions_str}",
        level="critical",
        context={"failed_count": len(failed_questions)},
    )


def alert_deploy_success(commit_sha: str = ""):
    send_alert(
        "New version successfully deployed to HuggingFace Spaces.",
        level="info",
        context={"commit": commit_sha[:7] if commit_sha else "unknown"},
    )
