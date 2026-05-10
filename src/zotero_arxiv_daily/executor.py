from __future__ import annotations

import random
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .cache import (
    add_papers_to_cache,
    cached_papers,
    load_preprint_cache,
    remove_papers_from_cache,
    save_preprint_cache,
)
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
from .interest import (
    apply_topic_boosts,
    augment_corpus_with_interest_profiles,
    select_with_topic_preferences,
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
        collections = {collection["key"]: collection for collection in collections}

        corpus = zot.everything(
            zot.items(itemType="conferencePaper || journalArticle || preprint")
        )
        corpus = [item for item in corpus if item["data"]["abstractNote"] != ""]

        def get_collection_path(col_key: str) -> str:
            if parent := collections[col_key]["data"]["parentCollection"]:
                return (
                    get_collection_path(parent)
                    + "/"
                    + collections[col_key]["data"]["name"]
                )

            return collections[col_key]["data"]["name"]

        for item in corpus:
            item["paths"] = [
                get_collection_path(collection)
                for collection in item["data"]["collections"]
            ]

        logger.info(f"Fetched {len(corpus)} zotero papers")

        return [
            CorpusPaper(
                title=item["data"]["title"],
                abstract=item["data"]["abstractNote"],
                added_date=datetime.strptime(
                    item["data"]["dateAdded"],
                    "%Y-%m-%dT%H:%M:%SZ",
                ),
                paths=item["paths"],
            )
            for item in corpus
        ]

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(
                f"Selecting zotero papers matching include_path: "
                f"{self.include_path_patterns}"
            )
            corpus = [
                item
                for item in corpus
                if any(
                    glob_match(path, pattern)
                    for path in item.paths
                    for pattern in self.include_path_patterns
                )
            ]

        if self.ignore_path_patterns:
            logger.info(
                f"Excluding zotero papers matching ignore_path: "
                f"{self.ignore_path_patterns}"
            )
            corpus = [
                item
                for item in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in item.paths
                    for pattern in self.ignore_path_patterns
                )
            ]

        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = "\n".join(
                [item.title + " - " + "\n".join(item.paths) for item in samples]
            )
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")

        return corpus

    def _is_advanced_mode(self) -> bool:
        return any(
            section in self.config
            for section in ("preprint", "published", "history", "interests")
        )

    def _rerank_and_filter(
        self,
        papers: list[Paper],
        corpus: list[CorpusPaper],
    ) -> list[Paper]:
        if len(papers) == 0:
            return []

        logger.info(f"Reranking {len(papers)} papers...")
        reranked = self.reranker.rerank(papers, corpus)
        reranked = apply_topic_boosts(reranked, self.config)
        reranked = filter_ultrasound_after_rerank(reranked, self.config)
        reranked = filter_by_score(reranked, self.config)
        return reranked

    def _retrieve_daily_preprints(self) -> list[Paper]:
        all_papers: list[Paper] = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue

            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)

        logger.info(
            f"Total {len(all_papers)} daily preprints retrieved from all sources"
        )
        return all_papers

    def _select_preprints(
        self,
        daily_preprints: list[Paper],
        corpus: list[CorpusPaper],
        history: dict,
        cache: dict,
    ) -> tuple[list[Paper], dict]:
        preprint_cfg = self.config.get("preprint", {})
        min_per_email = int(preprint_cfg.get("min_per_email", 0))
        max_per_email = int(
            preprint_cfg.get(
                "max_per_email",
                self.config.executor.get("max_paper_num", 100),
            )
        )

        daily_candidates = filter_ultrasound_papers(
            daily_preprints,
            self.config,
        )
        daily_ranked = self._rerank_and_filter(daily_candidates, corpus)
        daily_unseen = filter_unseen_papers(daily_ranked, history)

        add_papers_to_cache(cache, daily_unseen)

        selected = select_with_topic_preferences(
            daily_unseen,
            limit=max_per_email,
            config=self.config,
            quota_key="min_preprints_per_email",
        )
        selected_keys = {paper_key(paper) for paper in selected}

        logger.info(
            f"Selected {len(selected)} unseen daily preprints before cache/backfill"
        )

        if len(selected) < min_per_email:
            cache_candidates = filter_unseen_papers(
                cached_papers(cache),
                history,
                extra_seen_keys=selected_keys,
            )
            cache_candidates = filter_ultrasound_after_rerank(
                cache_candidates,
                self.config,
            )
            cache_candidates = filter_by_score(cache_candidates, self.config)

            selected = select_with_topic_preferences(
                cache_candidates,
                limit=min_per_email,
                config=self.config,
                quota_key="min_preprints_per_email",
                already_selected=selected,
            )
            selected_keys = {paper_key(paper) for paper in selected}

            logger.info(
                f"After cache fill: selected {len(selected)} preprints"
            )

        if len(selected) < min_per_email:
            need = min_per_email - len(selected)
            logger.info(
                f"Need {need} additional preprints. "
                "Searching recent unseen preprint backfill candidates..."
            )

            backfill_candidates = retrieve_recent_preprint_candidates(self.config)
            backfill_candidates = filter_ultrasound_papers(
                backfill_candidates,
                self.config,
            )
            backfill_ranked = self._rerank_and_filter(backfill_candidates, corpus)
            backfill_unseen = filter_unseen_papers(
                backfill_ranked,
                history,
                extra_seen_keys=selected_keys,
            )

            add_papers_to_cache(cache, backfill_unseen)

            selected = select_with_topic_preferences(
                backfill_unseen,
                limit=min_per_email,
                config=self.config,
                quota_key="min_preprints_per_email",
                already_selected=selected,
            )

            logger.info(
                f"After recent-backfill fill: selected {len(selected)} preprints"
            )

        return selected[:max_per_email], cache

    def _select_published_papers(
        self,
        corpus: list[CorpusPaper],
        history: dict,
        extra_seen_keys: set[str],
    ) -> list[Paper]:
        published_cfg = self.config.get("published", {})

        if not published_cfg.get("enabled", False):
            return []

        num_per_email = int(published_cfg.get("num_per_email", 0))
        if num_per_email <= 0:
            return []

        published_candidates = retrieve_pubmed_published_candidates(self.config)
        published_candidates = filter_ultrasound_papers(
            published_candidates,
            self.config,
        )
        published_ranked = self._rerank_and_filter(
            published_candidates,
            corpus,
        )
        published_unseen = filter_unseen_papers(
            published_ranked,
            history,
            extra_seen_keys=extra_seen_keys,
        )

        selected = select_with_topic_preferences(
            published_unseen,
            limit=num_per_email,
            config=self.config,
            quota_key="min_published_per_email",
        )
        logger.info(f"Selected {len(selected)} published papers")
        return selected

    def _generate_missing_metadata(self, papers: list[Paper]) -> None:
        logger.info("Generating bilingual summaries and affiliations...")

        for paper in tqdm(papers):
            if (
                paper.title_zh is None
                or paper.tldr_en is None
                or paper.tldr_zh is None
            ):
                paper.generate_bilingual_summary(
                    self.openai_client,
                    self.config.llm,
                )

            if paper.affiliations is None:
                paper.generate_affiliations(
                    self.openai_client,
                    self.config.llm,
                )

    def _run_legacy_mode(self, corpus: list[CorpusPaper]) -> None:
        all_papers = self._retrieve_daily_preprints()
        all_papers = filter_ultrasound_papers(all_papers, self.config)

        reranked_papers = self._rerank_and_filter(all_papers, corpus)

        max_paper_num = int(self.config.executor.get("max_paper_num", 100))
        reranked_papers = reranked_papers[:max_paper_num]

        if len(reranked_papers) == 0 and not self.config.executor.send_empty:
            logger.info("No papers found after filtering. No email will be sent.")
            return

        self._generate_missing_metadata(reranked_papers)

        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")

    def _run_advanced_mode(self, corpus: list[CorpusPaper]) -> None:
        history_cfg = self.config.get("history", {})
        preprint_cfg = self.config.get("preprint", {})

        history_path = history_cfg.get("path", "data/sent_history.json")
        history_max_records = int(history_cfg.get("max_records", 2000))

        cache_path = preprint_cfg.get("cache_path", "data/preprint_cache.json")
        cache_max_records = int(preprint_cfg.get("cache_max_records", 300))
        cache_max_age_days = int(preprint_cfg.get("cache_max_age_days", 45))

        history = load_history(history_path)
        cache = load_preprint_cache(cache_path)

        daily_preprints = self._retrieve_daily_preprints()
        preprints, cache = self._select_preprints(
            daily_preprints,
            corpus,
            history,
            cache,
        )

        selected_preprint_keys = {paper_key(paper) for paper in preprints}
        published_papers = self._select_published_papers(
            corpus,
            history,
            extra_seen_keys=selected_preprint_keys,
        )

        all_selected = preprints + published_papers

        if len(all_selected) == 0 and not self.config.executor.send_empty:
            logger.info("No papers found after filtering. No email will be sent.")
            return

        self._generate_missing_metadata(all_selected)

        logger.info("Sending email...")
        email_content = render_email(preprints, published_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")

        mark_papers_sent(history, all_selected)
        save_history(
            history_path,
            history,
            max_records=history_max_records,
        )

        remove_papers_from_cache(cache, preprints)
        save_preprint_cache(
            cache_path,
            cache,
            max_records=cache_max_records,
            max_age_days=cache_max_age_days,
        )

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        corpus = augment_corpus_with_interest_profiles(corpus, self.config)

        if len(corpus) == 0:
            logger.error(
                f"No zotero papers found. Please check your zotero settings:\n"
                f"{self.config.zotero}"
            )
            return

        if self._is_advanced_mode():
            self._run_advanced_mode(corpus)
        else:
            self._run_legacy_mode(corpus)
