from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from loguru import logger

from .history import paper_key
from .protocol import CorpusPaper, Paper


def _get_interest_cfg(config):
    return config.get("interests", {}) if config is not None else {}


def augment_corpus_with_interest_profiles(
    corpus: list[CorpusPaper],
    config,
) -> list[CorpusPaper]:
    """
    Add user-defined synthetic interest profiles to the Zotero corpus.

    Zotero reflects what the user has already stored, but not always what they
    are currently beginning to follow. Profiles are injected with the current
    date so the existing recency-weighted reranker treats them as recent
    interests.
    """
    interest_cfg = _get_interest_cfg(config)
    profiles = interest_cfg.get("profiles", [])

    if not profiles:
        return corpus

    augmented = list(corpus)
    now = datetime.utcnow()

    for profile in profiles:
        title = str(profile.get("title", profile.get("name", "Extra interest")))
        abstract = str(profile.get("abstract", "")).strip()
        copies = max(1, int(profile.get("copies", 1)))

        if not abstract:
            continue

        for _ in range(copies):
            augmented.append(
                CorpusPaper(
                    title=title,
                    abstract=abstract,
                    added_date=now,
                    paths=[f"__interest__/{profile.get('name', title)}"],
                )
            )

    logger.info(
        f"Added {len(augmented) - len(corpus)} synthetic interest-profile papers "
        f"to corpus"
    )
    return augmented


def _paper_text(paper: Paper) -> str:
    return "\n".join(
        [
            paper.title or "",
            paper.abstract or "",
            paper.full_text or "",
        ]
    )


def _matches_patterns(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def matches_topic(paper: Paper, topic: dict) -> bool:
    patterns = list(topic.get("patterns", []))
    if not patterns:
        return False

    return _matches_patterns(_paper_text(paper), patterns)


def apply_topic_boosts(papers: list[Paper], config) -> list[Paper]:
    """
    Add a configurable score bonus to papers matching explicitly prioritized topics.
    This is applied after semantic reranking and before score-threshold filtering.
    """
    interest_cfg = _get_interest_cfg(config)
    topics = interest_cfg.get("preferred_topics", [])

    if not topics:
        return papers

    for paper in papers:
        if paper.score is None:
            continue

        for topic in topics:
            if matches_topic(paper, topic):
                paper.score += float(topic.get("bonus", 0.0))

    return sorted(
        papers,
        key=lambda paper: paper.score if paper.score is not None else float("-inf"),
        reverse=True,
    )


def select_with_topic_preferences(
    papers: list[Paper],
    *,
    limit: int,
    config,
    quota_key: str,
    already_selected: list[Paper] | None = None,
) -> list[Paper]:
    """
    Prefer at least N papers from configured topics when such papers exist.

    This is a soft quota:
    - if a wearable-ultrasound paper exists, reserve a slot for it;
    - if none exists, fill with the normal top-ranked papers.
    """
    selected = list(already_selected or [])
    selected_keys = {paper_key(paper) for paper in selected}
    interest_cfg = _get_interest_cfg(config)
    topics = interest_cfg.get("preferred_topics", [])

    for topic in topics:
        target = max(0, int(topic.get(quota_key, 0)))
        if target <= 0:
            continue

        already_have = sum(matches_topic(paper, topic) for paper in selected)
        need = max(0, target - already_have)

        if need == 0:
            continue

        for paper in papers:
            if len(selected) >= limit or need == 0:
                break

            key = paper_key(paper)
            if key in selected_keys:
                continue

            if matches_topic(paper, topic):
                selected.append(paper)
                selected_keys.add(key)
                need -= 1

    for paper in papers:
        if len(selected) >= limit:
            break

        key = paper_key(paper)
        if key in selected_keys:
            continue

        selected.append(paper)
        selected_keys.add(key)

    return selected[:limit]
