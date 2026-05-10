import re
from loguru import logger


DEFAULT_ULTRASOUND_PATTERNS = [
    r"\bultrasound\b",
    r"\bultrasonic\b",
    r"\bultrasonography\b",
    r"\bsonograph\w*\b",
    r"\bechocardiograph\w*\b",
    r"\bechocardiogram\w*\b",
    r"\bHIFU\b",
    r"\bfocused ultrasound\b",
    r"\bhigh[- ]intensity focused ultrasound\b",
    r"\bultrafast ultrasound\b",
    r"\bplane[- ]wave ultrasound\b",
    r"\bultrasound localization microscop\w*\b",
    r"\bsuper[- ]resolution ultrasound\b",
    r"\bcontrast[- ]enhanced ultrasound\b",
    r"\bCEUS\b",
    r"\bintravascular ultrasound\b",
    r"\bIVUS\b",
    r"\bendoscopic ultrasound\b",
    r"\bEUS\b",
    r"\bshear[- ]wave elastograph\w*\b",
    r"\belastograph\w*\b",
    r"\bacoustic radiation force\b",
    r"\bARFI\b",
    r"\bphotoacoustic\w*\b",
    r"\boptoacoustic\w*\b",
    r"\bmicrobubble\w*\b",
    r"\bDoppler ultrasound\b",
    r"\bbeamform\w*\b",

    # Wearable-ultrasound focus terms
    r"\bwearable ultrasound\b",
    r"\bultrasound patch\b",
    r"\bultrasonic patch\b",
    r"\bbioadhesive ultrasound\b",
    r"\bconformal ultrasound\b",
    r"\bflexible ultrasound\b",
    r"\bwearable Doppler ultrasound\b",
    r"\bwearable cardiac ultrasound\b",
    r"\bultrasonic[- ]system[- ]on[- ]patch\b",
    r"\bUSoP\b",
]


def _get_filter_cfg(config):
    return config.get("filter", {}) if config is not None else {}


def _paper_text(paper, include_full_text=True) -> str:
    fields = [
        getattr(paper, "title", "") or "",
        getattr(paper, "abstract", "") or "",
    ]

    if include_full_text:
        fields.append(getattr(paper, "full_text", "") or "")

    return "\n".join(fields)


def is_ultrasound_related(paper, patterns=None, include_full_text=True) -> bool:
    text = _paper_text(paper, include_full_text=include_full_text)

    if not text.strip():
        return False

    patterns = patterns or DEFAULT_ULTRASOUND_PATTERNS

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def filter_ultrasound_papers(papers, config):
    filter_cfg = _get_filter_cfg(config)

    candidate_keyword_filter = bool(
        filter_cfg.get("candidate_keyword_filter", False)
    )

    if not candidate_keyword_filter:
        logger.info(
            "Candidate-stage ultrasound keyword filter is disabled. "
            "All retrieved papers will be sent to reranker."
        )
        return papers

    patterns = filter_cfg.get("ultrasound_patterns", DEFAULT_ULTRASOUND_PATTERNS)
    include_full_text = bool(filter_cfg.get("keyword_include_full_text", True))

    filtered = [
        paper
        for paper in papers
        if is_ultrasound_related(
            paper,
            patterns=patterns,
            include_full_text=include_full_text,
        )
    ]

    logger.info(
        f"Candidate-stage ultrasound hard filter: kept {len(filtered)}/{len(papers)} papers"
    )
    return filtered


def filter_by_score(papers, config):
    filter_cfg = _get_filter_cfg(config)
    min_score = filter_cfg.get("min_score", None)

    if min_score is None:
        return papers

    min_score = float(min_score)

    filtered = [
        paper
        for paper in papers
        if getattr(paper, "score", None) is not None
        and float(paper.score) >= min_score
    ]

    logger.info(
        f"Score filter: kept {len(filtered)}/{len(papers)} papers with score >= {min_score}"
    )
    return filtered


def filter_ultrasound_after_rerank(papers, config):
    filter_cfg = _get_filter_cfg(config)

    require_ultrasound_keyword = bool(
        filter_cfg.get("require_ultrasound_keyword", False)
    )

    if not require_ultrasound_keyword:
        return papers

    patterns = filter_cfg.get("ultrasound_patterns", DEFAULT_ULTRASOUND_PATTERNS)
    include_full_text = bool(filter_cfg.get("keyword_include_full_text", True))

    filtered = [
        paper
        for paper in papers
        if is_ultrasound_related(
            paper,
            patterns=patterns,
            include_full_text=include_full_text,
        )
    ]

    logger.info(
        f"Post-rerank ultrasound relevance filter: kept {len(filtered)}/{len(papers)} papers"
    )
    return filtered
