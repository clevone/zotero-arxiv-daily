from datetime import datetime, timezone
from loguru import logger

from .protocol import Paper


CLASSIC_ULTRASOUND_PAPERS = [
    {
        "title": "Ultrafast imaging in biomedical ultrasound",
        "authors": ["Mickael Tanter", "Mathias Fink"],
        "abstract": (
            "A classic review of ultrafast biomedical ultrasound imaging, "
            "including plane-wave insonification, coherent compounding, "
            "ultrafast Doppler, elastography, and functional ultrasound imaging."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/24402899/",
        "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/24402899/",
        "tldr": "经典综述：系统介绍超快超声成像、平面波发射、相干复合、多普勒、弹性成像和功能超声。",
    },
    {
        "title": "Coherent plane-wave compounding for very high frame rate ultrasonography and transient elastography",
        "authors": ["Gabriel Montaldo", "Mickael Tanter", "Jeremy Bercoff", "Nicolas Benech", "Mathias Fink"],
        "abstract": (
            "A foundational paper on coherent plane-wave compounding, "
            "using multiple tilted plane-wave transmissions to improve "
            "ultrasound image quality while maintaining high frame rate."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/19411209/",
        "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/19411209/",
        "tldr": "经典方法论文：提出平面波角度复合，在高帧率条件下提升超声图像质量。",
    },
    {
        "title": "Supersonic shear imaging: a new technique for soft tissue elasticity mapping",
        "authors": ["Jeremy Bercoff", "Mickael Tanter", "Mathias Fink"],
        "abstract": (
            "A foundational paper on supersonic shear imaging for ultrasound "
            "elastography, using focused ultrasound beams to generate shear waves "
            "and ultrafast imaging to map tissue elasticity."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/15139541/",
        "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/15139541/",
        "tldr": "经典弹性成像论文：利用聚焦超声产生剪切波，并用超快成像重建组织弹性。",
    },
    {
        "title": "Ultrafast ultrasound localization microscopy for deep super-resolution vascular imaging",
        "authors": ["Claudia Errico", "Juliette Pierre", "Sophie Pezet", "Yann Desailly", "Zsolt Lenkei", "Olivier Couture", "Mickael Tanter"],
        "abstract": (
            "A landmark paper demonstrating ultrafast ultrasound localization "
            "microscopy for deep super-resolution vascular imaging using "
            "microbubble contrast agents."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/26607546/",
        "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/26607546/",
        "tldr": "经典超分辨超声论文：利用微泡和超快成像实现深部血管超分辨定位显微成像。",
    },
    {
        "title": "FIELD: A program for simulating ultrasound systems",
        "authors": ["Jorgen Arendt Jensen"],
        "abstract": (
            "A classic work introducing the Field II simulation framework "
            "for modeling ultrasound transducer fields and ultrasound imaging systems."
        ),
        "url": "https://field-ii.dk/documents/jaj_nbc_1996.pdf",
        "pdf_url": "https://field-ii.dk/documents/jaj_nbc_1996.pdf",
        "tldr": "经典仿真工具论文：介绍 Field II，用于模拟超声换能器声场和成像系统。",
    },
]


def get_fallback_papers(config, n_needed: int):
    fallback_cfg = config.get("fallback", {})

    if not fallback_cfg.get("enabled", False):
        return []

    max_fallback = int(fallback_cfg.get("max_fallback_papers", n_needed))
    n = max(0, min(n_needed, max_fallback))

    if n == 0:
        return []

    pool = fallback_cfg.get("classic_pool", None)
    if pool is None:
        pool = CLASSIC_ULTRASOUND_PAPERS

    if len(pool) == 0:
        return []

    # Rotate classics by day, so the same fallback papers are not always sent.
    day_index = datetime.now(timezone.utc).timetuple().tm_yday
    start = day_index % len(pool)
    rotated = list(pool[start:]) + list(pool[:start])
    selected = rotated[:n]

    papers = []
    for item in selected:
        papers.append(
            Paper(
                source="classic-ultrasound",
                title=item["title"],
                authors=list(item.get("authors", [])),
                abstract=item.get("abstract", ""),
                url=item["url"],
                pdf_url=item.get("pdf_url", item["url"]),
                full_text=item.get("abstract", ""),
                tldr=item.get("tldr", None),
                affiliations=[],
                score=10.0,
            )
        )

    logger.info(f"Added {len(papers)} fallback classic ultrasound papers")
    return papers
