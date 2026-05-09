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
    r"\bacoustic radiation force\b",
    r"\bARFI\b",
    r"\bphotoacoustic\w*\b",
    r"\boptoacoustic\w*\b",
    r"\bmicrobubble\w*\b",
]


def _paper_text(paper) -> str:
    fields = [
        getattr(paper, "title", "") or "",
        getattr(paper, "abstract", "") or "",
        getattr(paper, "full_text", "") or "",
    ]
    return "\n".join(fields)


def is_ultrasound_related(paper, patterns=None) -> bool:
    text = _paper_text(paper)
    if not text.strip():
        return False

    patterns = patterns or DEFAULT_ULTRASOUND_PATTERNS

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def filter_ultrasound_papers(papers, config):
    filter_cfg = config.get("filter", {})
    require_ultrasound_keyword = filter_cfg.get("require_ultrasound_keyword", True)

    if not require_ultrasound_keyword:
        return papers

    patterns = filter_cfg.get("ultrasound_patterns", DEFAULT_ULTRASOUND_PATTERNS)
    filtered = [p for p in papers if is_ultrasound_related(p, patterns)]

    logger.info(
        f"Ultrasound hard filter: kept {len(filtered)}/{len(papers)} papers"
    )

    if len(papers) > 0 and len(filtered) == 0:
        logger.info(
            "No papers passed the ultrasound hard filter. "
            "This is acceptable if there are no ultrasound-related papers today."
        )

    return filtered


def filter_by_score(papers, config):
    filter_cfg = config.get("filter", {})
    min_score = filter_cfg.get("min_score", None)

    if min_score is None:
        return papers

    min_score = float(min_score)
    filtered = [
        p for p in papers
        if getattr(p, "score", None) is not None and p.score >= min_score
    ]

    logger.info(
        f"Score filter: kept {len(filtered)}/{len(papers)} papers "
        f"with score >= {min_score}"
    )

    return filteredS
