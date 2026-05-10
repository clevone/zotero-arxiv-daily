
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .protocol import Paper


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def paper_key(paper: Paper) -> str:
    """
    Stable key used for cross-run deduplication.

    Prefer URL because arXiv/PubMed/bioRxiv URLs are stable. If URL is missing,
    fall back to normalized title.
    """
    if paper.url:
        return f"url::{_normalize_text(paper.url)}"
    return f"title::{_normalize_text(paper.title)}"


def load_history(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"sent": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sent": []}

    if not isinstance(data, dict) or not isinstance(data.get("sent", []), list):
        return {"sent": []}

    return data


def save_history(path: str | Path, history: dict, max_records: int = 2000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sent = history.get("sent", [])
    if max_records > 0:
        sent = sent[-max_records:]

    payload = {"sent": sent}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def seen_keys(history: dict) -> set[str]:
    return {
        item.get("key", "")
        for item in history.get("sent", [])
        if isinstance(item, dict) and item.get("key")
    }


def filter_unseen_papers(
    papers: Iterable[Paper],
    history: dict,
    extra_seen_keys: set[str] | None = None,
) -> list[Paper]:
    seen = seen_keys(history)
    if extra_seen_keys:
        seen |= set(extra_seen_keys)

    unique: list[Paper] = []
    local_seen: set[str] = set()

    for paper in papers:
        key = paper_key(paper)
        if key in seen or key in local_seen:
            continue
        unique.append(paper)
        local_seen.add(key)

    return unique


def mark_papers_sent(history: dict, papers: Iterable[Paper]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    sent = history.setdefault("sent", [])

    existing = seen_keys(history)

    for paper in papers:
        key = paper_key(paper)
        if key in existing:
            continue

        sent.append(
            {
                "key": key,
                "source": paper.source,
                "title": paper.title,
                "url": paper.url,
                "sent_at": now,
            }
        )
        existing.add(key)

    return history
