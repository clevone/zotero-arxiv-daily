import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests
from loguru import logger

from .protocol import Paper
from .paper_filter import is_ultrasound_related


PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


DEFAULT_PUBMED_ULTRASOUND_QUERY = """
(
  ultrasound[Title/Abstract]
  OR ultrasonography[Title/Abstract]
  OR ultrasonic[Title/Abstract]
  OR sonography[Title/Abstract]
  OR echocardiography[Title/Abstract]
  OR "focused ultrasound"[Title/Abstract]
  OR HIFU[Title/Abstract]
  OR "ultrafast ultrasound"[Title/Abstract]
  OR "plane wave ultrasound"[Title/Abstract]
  OR "contrast-enhanced ultrasound"[Title/Abstract]
  OR "ultrasound localization microscopy"[Title/Abstract]
  OR "super-resolution ultrasound"[Title/Abstract]
  OR "shear wave elastography"[Title/Abstract]
  OR photoacoustic[Title/Abstract]
  OR optoacoustic[Title/Abstract]
)
AND
(
  imaging[Title/Abstract]
  OR beamforming[Title/Abstract]
  OR reconstruction[Title/Abstract]
  OR localization[Title/Abstract]
  OR microscopy[Title/Abstract]
  OR elastography[Title/Abstract]
  OR Doppler[Title/Abstract]
  OR transducer[Title/Abstract]
  OR "medical imaging"[Title/Abstract]
)
"""


def _clean_query(q: str) -> str:
    return " ".join(q.split())


def _paper_key(paper: Paper) -> str:
    return (paper.title or "").strip().lower()


def _existing_title_set(existing_papers):
    return {_paper_key(p) for p in existing_papers or [] if _paper_key(p)}


def _pubmed_esearch(query: str, retmax: int, sort: str = "relevance") -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": sort,
    }

    response = requests.get(PUBMED_ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _node_text(node):
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _extract_article_ids(article):
    ids = {}

    for article_id in article.findall(".//ArticleId"):
        id_type = article_id.attrib.get("IdType", "")
        text = _node_text(article_id)
        if id_type and text:
            ids[id_type] = text

    return ids


def _extract_authors(article):
    authors = []

    for author in article.findall(".//AuthorList/Author"):
        last = _node_text(author.find("LastName"))
        fore = _node_text(author.find("ForeName"))
        collective = _node_text(author.find("CollectiveName"))

        if collective:
            authors.append(collective)
        elif last and fore:
            authors.append(f"{fore} {last}")
        elif last:
            authors.append(last)

    return authors


def _extract_abstract(article):
    parts = []

    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label")
        text = _node_text(abstract_text)

        if not text:
            continue

        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)

    return "\n".join(parts)


def _pubmed_efetch(pmids: list[str]) -> list[Paper]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(PUBMED_EFETCH_URL, params=params, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.text)

    papers = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _node_text(article.find(".//PMID"))
        title = _node_text(article.find(".//ArticleTitle"))
        abstract = _extract_abstract(article)
        authors = _extract_authors(article)
        ids = _extract_article_ids(article)

        if not pmid or not title:
            continue

        doi = ids.get("doi")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        pdf_url = f"https://doi.org/{doi}" if doi else url

        journal = _node_text(article.find(".//Journal/Title"))
        pub_year = _node_text(article.find(".//PubDate/Year"))

        extra = []
        if journal:
            extra.append(f"Journal: {journal}")
        if pub_year:
            extra.append(f"Year: {pub_year}")

        full_text_preview = "\n".join(extra + [abstract])

        papers.append(
            Paper(
                source="pubmed-fallback",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                pdf_url=pdf_url,
                full_text=full_text_preview,
                tldr=None,
                affiliations=None,
                score=10.0,
            )
        )

    return papers


def get_pubmed_fallback_papers(config, n_needed: int, existing_papers=None) -> list[Paper]:
    fallback_cfg = config.get("fallback", {})

    query = fallback_cfg.get(
        "pubmed_query",
        DEFAULT_PUBMED_ULTRASOUND_QUERY,
    )
    query = _clean_query(query)

    retmax = int(fallback_cfg.get("pubmed_retmax", 30))
    sort = fallback_cfg.get("pubmed_sort", "relevance")

    logger.info(f"Searching PubMed fallback papers with query: {query}")
    logger.info(f"PubMed fallback sort={sort}, retmax={retmax}")

    pmids = _pubmed_esearch(query=query, retmax=retmax, sort=sort)

    # Be polite to NCBI.
    time.sleep(0.34)

    papers = _pubmed_efetch(pmids)

    existing_titles = _existing_title_set(existing_papers)

    filtered = []
    for paper in papers:
        key = _paper_key(paper)

        if not key or key in existing_titles:
            continue

        # Keep only papers that still match our ultrasound detector.
        if not is_ultrasound_related(paper, include_full_text=True):
            continue

        filtered.append(paper)

        if len(filtered) >= n_needed:
            break

    logger.info(f"PubMed fallback returned {len(filtered)} papers")

    return filtered


def get_fallback_papers(config, n_needed: int, existing_papers=None) -> list[Paper]:
    fallback_cfg = config.get("fallback", {})

    if not fallback_cfg.get("enabled", False):
        return []

    mode = fallback_cfg.get("mode", "pubmed")

    if mode == "none":
        return []

    if mode == "pubmed":
        return get_pubmed_fallback_papers(
            config,
            n_needed=n_needed,
            existing_papers=existing_papers,
        )

    raise ValueError(f"Unknown fallback mode: {mode}")
