
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import arxiv
import requests
from loguru import logger

from .protocol import Paper


PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


DEFAULT_SEARCH_TERMS = [
    "ultrasound",
    "ultrasonography",
    "ultrasonic",
    "sonography",
    '"focused ultrasound"',
    "HIFU",
    '"ultrafast ultrasound"',
    '"plane wave ultrasound"',
    '"contrast-enhanced ultrasound"',
    '"ultrasound localization microscopy"',
    '"super-resolution ultrasound"',
    '"shear wave elastography"',
    "photoacoustic",
    "optoacoustic",
]


DEFAULT_PUBMED_QUERY = """
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


DEFAULT_JOURNAL_WHITELIST = [
    "Nature Biomedical Engineering",
    "Nature Medicine",
    "Nature Communications",
    "Science Advances",
    "Radiology",
    "IEEE Transactions on Medical Imaging",
    "Medical Image Analysis",
    "Ultrasound in Medicine & Biology",
    "IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control",
    "Physics in Medicine & Biology",
    "Medical Physics",
    "Journal of the American College of Cardiology",
    "Circulation",
]


def _clean_query(query: str) -> str:
    return " ".join(query.split())


def _arxiv_term(term: str) -> str:
    # arXiv query strings accept quoted phrases, e.g. all:"focused ultrasound".
    return f"all:{term}"


def _build_arxiv_query(config) -> str:
    preprint_cfg = config.get("preprint", {})
    terms = preprint_cfg.get("search_terms", DEFAULT_SEARCH_TERMS)
    term_query = " OR ".join(_arxiv_term(term) for term in terms)

    categories = list(config.source.arxiv.category)
    category_query = " OR ".join(f"cat:{category}" for category in categories)

    days = int(preprint_cfg.get("backfill_days", 30))
    start = datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc)
    date_query = (
        f"submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    )

    return f"({term_query}) AND ({category_query}) AND {date_query}"


def _convert_arxiv_result(raw_paper: arxiv.Result) -> Paper:
    return Paper(
        source="arxiv",
        title=raw_paper.title,
        authors=[author.name for author in raw_paper.authors],
        abstract=raw_paper.summary,
        url=raw_paper.entry_id,
        pdf_url=raw_paper.pdf_url,
        full_text=None,
    )


def retrieve_recent_arxiv_candidates(config) -> list[Paper]:
    preprint_cfg = config.get("preprint", {})
    retmax = int(preprint_cfg.get("arxiv_backfill_retmax", 80))

    search = arxiv.Search(
        query=_build_arxiv_query(config),
        max_results=retmax,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(num_retries=5, delay_seconds=5)

    logger.info(f"Searching recent arXiv backfill candidates, retmax={retmax}")
    return [_convert_arxiv_result(paper) for paper in client.results(search)]


def _recent_biorxiv_endpoint(server: str, days: int, cursor: int = 0) -> str:
    return f"https://api.biorxiv.org/details/{server}/{days}d/{cursor}"


def _convert_biorxiv_item(item: dict[str, Any], server: str) -> Paper:
    pdf_url = (
        f"https://www.{server}.org/content/"
        f"{item['doi']}v{item['version']}.full.pdf"
    )

    return Paper(
        source=server,
        title=item["title"],
        authors=[a.strip() for a in item["authors"].split(";")],
        abstract=item["abstract"],
        url=pdf_url,
        pdf_url=pdf_url,
        full_text=None,
    )


def retrieve_recent_biorxiv_family_candidates(config, server: str) -> list[Paper]:
    preprint_cfg = config.get("preprint", {})
    days = int(preprint_cfg.get("backfill_days", 30))
    retmax = int(preprint_cfg.get("biorxiv_backfill_retmax", 120))

    source_cfg = config.source.get(server, None)
    if source_cfg is None:
        return []

    allowed_categories = {
        category.lower()
        for category in source_cfg.category
    }

    papers: list[Paper] = []
    cursor = 0

    logger.info(
        f"Searching recent {server} backfill candidates, "
        f"days={days}, retmax={retmax}"
    )

    while len(papers) < retmax:
        response = requests.get(
            _recent_biorxiv_endpoint(server, days, cursor),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        collection = data.get("collection", [])

        if not collection:
            break

        for item in collection:
            if item.get("category", "").lower() not in allowed_categories:
                continue
            papers.append(_convert_biorxiv_item(item, server))
            if len(papers) >= retmax:
                break

        # The API pages in chunks; when the current page is shorter than the
        # typical page size, there is usually nothing more to fetch.
        if len(collection) < 100:
            break

        cursor += len(collection)
        time.sleep(0.34)

    return papers


def retrieve_recent_preprint_candidates(config) -> list[Paper]:
    enabled_sources = set(config.executor.source)
    papers: list[Paper] = []

    if "arxiv" in enabled_sources:
        papers.extend(retrieve_recent_arxiv_candidates(config))

    if "biorxiv" in enabled_sources:
        papers.extend(retrieve_recent_biorxiv_family_candidates(config, "biorxiv"))

    if "medrxiv" in enabled_sources:
        papers.extend(retrieve_recent_biorxiv_family_candidates(config, "medrxiv"))

    return papers


def _node_text(node) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _extract_article_ids(article) -> dict[str, str]:
    ids: dict[str, str] = {}

    for article_id in article.findall(".//ArticleId"):
        id_type = article_id.attrib.get("IdType", "")
        text = _node_text(article_id)
        if id_type and text:
            ids[id_type] = text

    return ids


def _extract_authors(article) -> list[str]:
    authors: list[str] = []

    for author in article.findall(".//AuthorList/Author"):
        last = _node_text(author.find("LastName"))
        fore = _node_text(author.find("ForeName"))
        collective = _node_text(author.find("CollectiveName"))

        if collective:
            authors.append(collective)
        elif fore and last:
            authors.append(f"{fore} {last}")
        elif last:
            authors.append(last)

    return authors


def _extract_abstract(article) -> str:
    parts: list[str] = []

    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label")
        text = _node_text(abstract_text)

        if not text:
            continue

        parts.append(f"{label}: {text}" if label else text)

    return "\n".join(parts)


def _article_year(article) -> int | None:
    year_text = _node_text(article.find(".//PubDate/Year"))
    if year_text.isdigit():
        return int(year_text)
    return None


def _pubmed_esearch(query: str, retmax: int, sort: str) -> list[str]:
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


def _pubmed_efetch(pmids: list[str], config) -> list[Paper]:
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

    published_cfg = config.get("published", {})
    years_back = int(published_cfg.get("years_back", 10))
    min_year = datetime.now(timezone.utc).year - years_back
    journal_whitelist = {
        journal.lower()
        for journal in published_cfg.get(
            "journal_whitelist",
            DEFAULT_JOURNAL_WHITELIST,
        )
    }

    papers: list[Paper] = []

    for article in root.findall(".//PubmedArticle"):
        title = _node_text(article.find(".//ArticleTitle"))
        pmid = _node_text(article.find(".//PMID"))
        abstract = _extract_abstract(article)
        authors = _extract_authors(article)
        ids = _extract_article_ids(article)
        journal = _node_text(article.find(".//Journal/Title"))
        year = _article_year(article)

        if not pmid or not title or not abstract:
            continue
        if year is not None and year < min_year:
            continue
        if journal_whitelist and journal.lower() not in journal_whitelist:
            continue

        doi = ids.get("doi")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        pdf_url = f"https://doi.org/{doi}" if doi else url

        full_text = "\n".join(
            part for part in [f"Journal: {journal}", f"Year: {year}", abstract]
            if part
        )

        papers.append(
            Paper(
                source="pubmed",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                pdf_url=pdf_url,
                full_text=full_text,
            )
        )

    return papers


def retrieve_pubmed_published_candidates(config) -> list[Paper]:
    published_cfg = config.get("published", {})
    if not published_cfg.get("enabled", False):
        return []

    query = _clean_query(
        published_cfg.get("pubmed_query", DEFAULT_PUBMED_QUERY)
    )
    retmax = int(published_cfg.get("pubmed_retmax", 100))
    sort = published_cfg.get("pubmed_sort", "relevance")

    logger.info(
        f"Searching published PubMed candidates, sort={sort}, retmax={retmax}"
    )
    pmids = _pubmed_esearch(query=query, retmax=retmax, sort=sort)

    time.sleep(0.34)

    return _pubmed_efetch(pmids, config)
