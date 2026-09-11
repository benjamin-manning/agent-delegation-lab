"""Simple JSON-lines logger for research data collection."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"


def log_session(
    game_key: str,
    agent_configs: list[dict],
    settings: dict,
    results: dict,
) -> Path:
    """Append one JSON-lines record to logs/sessions.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "sessions.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "game_key": game_key,
        "agent_configs": agent_configs,
        "settings": settings,
        "results": results,
    }

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return path
