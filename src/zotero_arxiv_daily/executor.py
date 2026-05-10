
from __future__ import annotations

import random
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .construct_email import render_email
from .fallback import (
    retrieve_pubmed_published_candidates,
    retrieve_recent_preprint_candidates,
)
from .history import (
    filter_unseen_papers,
    load_history,
    mark_papers_sent,
    paper_key,
    save_history,
)
from .paper_filter import (
    filter_by_score,
    filter_ultrasound_after_rerank,
    filter_ultrasound_papers,
)
from .protocol import CorpusPaper, Paper
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .utils import glob_match, send_email


def normalize_path_patterns(
    patterns: list[str] | ListConfig | None,
    config_key: str,
) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(
            f"config.zotero.{config_key} must contain only glob pattern strings."
        )

    return list(patterns)


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(
            config.zotero.include_path,
            "include_path",
        )
        self.ignore_path_patterns = normalize_path_patterns(
            config.zotero.ignore_path,
            "ignore_path",
        )
        self.retrievers = {
            source: get_retriever_cls(source)(config)
            for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(
            self.config.zotero.user_id,
            "user",
            self.config.zotero.api_key,
        )

        collections = zot.everything(zot.collections())
        collections = {c["key"]: c for c in collections}

        corpus = zot.everything(
            zot.items(itemType="conferencePaper || journalArticle || preprint")
        )
        corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

        def get_collection_path(col_key: str) -> str:
            if p := collections[col_key]["data"]["parentCollection"]:
                return get_collection_path(p) + "/" + collections[col_key]["data"]["name"]
            return collections[col_key]["data"]["name"]

        for c in corpus:
            paths = [get_collection_path(col) for col in c["data"]["collections"]]
            c["paths"] = paths

        logger.info(f"Fetched {len(corpus)} zotero papers")

        return [
            CorpusPaper(
                title=c["data"]["title"],
                abstract=c["data"]["abstractNote"],
                added_date=datetime.strptime(
                    c["data"]["dateAdded"],
                    "%Y-%m-%dT%H:%M:%SZ",
                ),
                paths=c["paths"],
            )
            for c in corpus
        ]

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(
                f"Selecting zotero papers matching include_path: "
                f"{self.include_path_patterns}"
            )
            corpus = [
                c
                for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]

        if self.ignore_path_patterns:
            logger.info(
                f"Excluding zotero papers matching ignore_path: "
                f"{self.ignore_path_patterns}"
            )
            corpus = [
                c
                for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]

        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = "\n".join(
                [c.title + " - " + "\n".join(c.paths) for c in samples]
            )
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")

        return corpus

    def _rerank_and_filter(
        self,
        papers: list[Paper],
        corpus: list[CorpusPaper],
        apply_score_filter: bool = True,
    ) -> list[Paper]:
        if not papers:
            return []

        reranked = self.reranker.rerank(papers, corpus)
        reranked = filter_ultrasound_after_rerank(reranked, self.config)

        if apply_score_filter:
            reranked = filter_by_score(reranked, self.config)

        return reranked

    @staticmethod
    def _top_unseen(
        papers: list[Paper],
        history: dict,
        chosen: list[Paper],
        limit: int,
    ) -> list[Paper]:
        chosen_keys = {paper_key(paper) for paper in chosen}
        unseen = filter_unseen_papers(
            papers,
            history,
            extra_seen_keys=chosen_keys,
        )
        return unseen[:limit]

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)

        if len(corpus) == 0:
            logger.error(
                "No zotero papers found. Please check your zotero settings:\n"
                f"{self.config.zotero}"
            )
            return

        history_cfg = self.config.get("history", {})
        history_path = history_cfg.get("path", "data/sent_history.json")
        max_history_records = int(history_cfg.get("max_records", 2000))
        history = load_history(history_path)

        # ------------------------------------------------------------------
        # 1. Today's preprints from the existing daily retrievers
        # ------------------------------------------------------------------
        daily_preprints: list[Paper] = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue

            logger.info(f"Retrieved {len(papers)} {source} papers")
            daily_preprints.extend(papers)

        logger.info(
            f"Total {len(daily_preprints)} daily preprints retrieved from all sources"
        )

        daily_preprints = filter_ultrasound_papers(
            daily_preprints,
            self.config,
        )
        daily_preprints = self._rerank_and_filter(daily_preprints, corpus)
        daily_preprints = filter_unseen_papers(daily_preprints, history)

        preprint_cfg = self.config.get("preprint", {})
        min_preprints = int(preprint_cfg.get("min_per_email", 0))
        max_preprints = int(preprint_cfg.get("max_per_email", min_preprints or 100))

        selected_preprints = daily_preprints[:max_preprints]

        # ------------------------------------------------------------------
        # 2. If today's strong preprints are fewer than required, backfill with
        #    recent, unseen preprints from the last N days.
        # ------------------------------------------------------------------
        if min_preprints > 0 and len(selected_preprints) < min_preprints:
            n_needed = min_preprints - len(selected_preprints)
            logger.info(
                f"Need {n_needed} additional preprints. "
                "Searching recent unseen preprint backfill candidates..."
            )

            backfill_candidates = retrieve_recent_preprint_candidates(self.config)
            backfill_candidates = filter_ultrasound_papers(
                backfill_candidates,
                self.config,
            )

            strict_backfill = self._rerank_and_filter(
                backfill_candidates,
                corpus,
                apply_score_filter=True,
            )
            selected_preprints.extend(
                self._top_unseen(
                    strict_backfill,
                    history,
                    selected_preprints,
                    n_needed,
                )
            )

            # To satisfy the hard "at least N preprints" quota, relax only the
            # score threshold if the strict pool is still too small. The
            # ultrasound keyword gate is still kept.
            if len(selected_preprints) < min_preprints:
                n_needed = min_preprints - len(selected_preprints)
                logger.info(
                    f"Still need {n_needed} preprints after strict backfill. "
                    "Relaxing score threshold while keeping ultrasound relevance."
                )

                relaxed_backfill = self._rerank_and_filter(
                    backfill_candidates,
                    corpus,
                    apply_score_filter=False,
                )
                selected_preprints.extend(
                    self._top_unseen(
                        relaxed_backfill,
                        history,
                        selected_preprints,
                        n_needed,
                    )
                )

        selected_preprints = selected_preprints[:max_preprints]

        # ------------------------------------------------------------------
        # 3. Published papers from PubMed, reranked against Zotero corpus and
        #    deduplicated across days.
        # ------------------------------------------------------------------
        published_cfg = self.config.get("published", {})
        selected_published: list[Paper] = []

        if published_cfg.get("enabled", False):
            num_published = int(published_cfg.get("num_per_email", 0))

            if num_published > 0:
                published_candidates = retrieve_pubmed_published_candidates(
                    self.config
                )
                published_candidates = self._rerank_and_filter(
                    published_candidates,
                    corpus,
                    apply_score_filter=True,
                )
                selected_published = self._top_unseen(
                    published_candidates,
                    history,
                    selected_preprints,
                    num_published,
                )

        # ------------------------------------------------------------------
        # 4. Final email
        # ------------------------------------------------------------------
        final_papers = selected_preprints + selected_published
        max_total = int(self.config.executor.get("max_paper_num", 100))
        final_papers = final_papers[:max_total]

        logger.info(
            f"Selected {len(selected_preprints)} preprints and "
            f"{len(selected_published)} published papers for email"
        )

        if len(final_papers) == 0 and not self.config.executor.send_empty:
            logger.info("No papers found after filtering. No email will be sent.")
            return

        logger.info("Generating TLDR and affiliations...")

        for paper in tqdm(final_papers):
            if paper.tldr is None:
                paper.generate_tldr(self.openai_client, self.config.llm)

            if paper.affiliations is None:
                paper.generate_affiliations(self.openai_client, self.config.llm)

        logger.info("Sending email...")
        email_content = render_email(final_papers)
        send_email(self.config, email_content)

        # Persist only after the email was sent successfully.
        mark_papers_sent(history, final_papers)
        save_history(history_path, history, max_records=max_history_records)

        logger.info("Email sent successfully")
