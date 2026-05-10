from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .history import paper_key
from .protocol import Paper


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _paper_to_dict(paper: Paper) -> dict:
    return {
        "source": paper.source,
        "title": paper.title,
        "authors": list(paper.authors or []),
        "abstract": paper.abstract,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "full_text": paper.full_text,
        "tldr": paper.tldr,
        "affiliations": paper.affiliations,
        "score": paper.score,
    }


def _paper_from_dict(data: dict) -> Paper | None:
    required = ("source", "title", "authors", "abstract", "url")
    if any(key not in data for key in required):
        return None

    return Paper(
        source=data["source"],
        title=data["title"],
        authors=list(data.get("authors", [])),
        abstract=data.get("abstract", ""),
        url=data["url"],
        pdf_url=data.get("pdf_url"),
        full_text=data.get("full_text"),
        tldr=data.get("tldr"),
        affiliations=data.get("affiliations"),
        score=data.get("score"),
    )


def load_preprint_cache(path: str | Path) -> dict:
    """
    Load the persistent cache of strong but unsent preprints.

    File format:
    {
      "items": [
        {
          "key": "...",
          "added_at": "...",
          "last_seen_at": "...",
          "paper": {...}
        }
      ]
    }
    """
    path = Path(path)

    if not path.exists():
        return {"items": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"items": []}

    if not isinstance(data, dict) or not isinstance(data.get("items", []), list):
        return {"items": []}

    return data


def save_preprint_cache(
    path: str | Path,
    cache: dict,
    *,
    max_records: int = 300,
    max_age_days: int = 45,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cache = prune_preprint_cache(
        cache,
        max_records=max_records,
        max_age_days=max_age_days,
    )

    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prune_preprint_cache(
    cache: dict,
    *,
    max_records: int = 300,
    max_age_days: int = 45,
) -> dict:
    now = _now()
    cutoff = now - timedelta(days=max_age_days)

    items = []
    for item in cache.get("items", []):
        if not isinstance(item, dict):
            continue

        added_at = _parse_dt(item.get("added_at"))
        paper_dict = item.get("paper")

        if not isinstance(paper_dict, dict):
            continue

        if added_at is not None and added_at < cutoff:
            continue

        if _paper_from_dict(paper_dict) is None:
            continue

        items.append(item)

    # Keep stronger papers first; for ties, prefer newer papers.
    def _sort_key(item: dict):
        paper_score = item.get("paper", {}).get("score")
        score = float(paper_score) if isinstance(paper_score, (int, float)) else float("-inf")
        added_at = _parse_dt(item.get("added_at")) or datetime.min.replace(tzinfo=timezone.utc)
        return (score, added_at)

    items.sort(key=_sort_key, reverse=True)

    if max_records > 0:
        items = items[:max_records]

    return {"items": items}


def cache_keys(cache: dict) -> set[str]:
    return {
        item.get("key", "")
        for item in cache.get("items", [])
        if isinstance(item, dict) and item.get("key")
    }


def add_papers_to_cache(cache: dict, papers: Iterable[Paper]) -> dict:
    """
    Add strong, unsent preprints to the cache.

    If a paper already exists in the cache, update the stored paper payload and
    `last_seen_at`, but preserve its original `added_at`.
    """
    now = _now_iso()
    items = list(cache.get("items", []))
    index = {
        item.get("key"): item
        for item in items
        if isinstance(item, dict) and item.get("key")
    }

    for paper in papers:
        key = paper_key(paper)
        if not key:
            continue

        if key in index:
            index[key]["last_seen_at"] = now
            index[key]["paper"] = _paper_to_dict(paper)
            continue

        item = {
            "key": key,
            "added_at": now,
            "last_seen_at": now,
            "paper": _paper_to_dict(paper),
        }
        items.append(item)
        index[key] = item

    cache["items"] = items
    return cache


def cached_papers(cache: dict) -> list[Paper]:
    papers: list[Paper] = []

    for item in cache.get("items", []):
        if not isinstance(item, dict):
            continue

        paper_dict = item.get("paper")
        if not isinstance(paper_dict, dict):
            continue

        paper = _paper_from_dict(paper_dict)
        if paper is not None:
            papers.append(paper)

    # Use the already-computed scores stored in the cache. This avoids rerunning
    # the expensive embedding model every day over the whole cache.
    papers.sort(
        key=lambda p: float(p.score) if p.score is not None else float("-inf"),
        reverse=True,
    )
    return papers


def remove_papers_from_cache(cache: dict, papers: Iterable[Paper]) -> dict:
    remove_keys = {paper_key(paper) for paper in papers}
    cache["items"] = [
        item
        for item in cache.get("items", [])
        if isinstance(item, dict) and item.get("key") not in remove_keys
    ]
    return cache
