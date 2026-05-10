from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import json
import re

import tiktoken
from loguru import logger
from openai import OpenAI

RawPaperItem = TypeVar("RawPaperItem")


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None

    # Backward-compatible English summary field used by the original project.
    tldr: Optional[str] = None

    # New bilingual fields used by the updated email renderer.
    title_zh: Optional[str] = None
    tldr_en: Optional[str] = None
    tldr_zh: Optional[str] = None

    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _build_summary_prompt(self) -> str:
        prompt = (
            "Given the following information of a scientific paper, return a JSON object "
            "with exactly three keys: "
            '"title_zh", "tldr_en", "tldr_zh".\n'
            "- title_zh: a faithful Chinese translation of the title.\n"
            "- tldr_en: one concise English sentence summarizing the core method and main contribution.\n"
            "- tldr_zh: one concise Chinese sentence summarizing the core method and main contribution.\n"
            "Do not include Markdown. Return valid JSON only.\n\n"
        )

        if self.title:
            prompt += f"Title:\n{self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract:\n{self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n{self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        return enc.decode(prompt_tokens)

    def generate_bilingual_summary(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> tuple[str | None, str | None, str | None]:
        try:
            response = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant who accurately summarizes scientific papers. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": self._build_summary_prompt()},
                ],
                **llm_params.get("generation_kwargs", {}),
            )

            raw = response.choices[0].message.content
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else raw)

            self.title_zh = str(payload.get("title_zh", "")).strip() or None
            self.tldr_en = str(payload.get("tldr_en", "")).strip() or None
            self.tldr_zh = str(payload.get("tldr_zh", "")).strip() or None

            # Preserve the original field for compatibility with existing code/tests.
            self.tldr = self.tldr_en or self.tldr or self.abstract

            return self.title_zh, self.tldr_en, self.tldr_zh

        except Exception as exc:
            logger.warning(
                f"Failed to generate bilingual summary of {self.url}: {exc}"
            )

            # Graceful fallback: keep a usable English summary even if translation fails.
            self.tldr_en = self.tldr_en or self.tldr or self.abstract
            self.tldr = self.tldr_en
            return self.title_zh, self.tldr_en, self.tldr_zh

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        """
        Original single-language method retained for backward compatibility.
        """
        lang = llm_params.get("language", "English")
        prompt = (
            f"Given the following information of a paper, generate a one-sentence "
            f"TLDR summary in {lang}:\n\n"
        )

        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who perfectly summarizes scientific paper, "
                        f"and gives the core idea of the paper to the user. "
                        f"Your answer should be in {lang}."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )
        return response.choices[0].message.content

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as exc:
            logger.warning(f"Failed to generate tldr of {self.url}: {exc}")
            self.tldr = self.abstract
            return self.tldr

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = (
                "Given the beginning of a paper, extract the affiliations of the authors "
                "in a python list format, which is sorted by the author order. "
                "If there is no affiliation found, return an empty list '[]':\n\n"
                f"{self.full_text}"
            )

            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]
            prompt = enc.decode(prompt_tokens)

            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant who perfectly extracts affiliations of authors "
                            "from a paper. You should return a python list of affiliations sorted "
                            'by the author order, like ["TsingHua University","Peking University"]. '
                            "If an affiliation is consisted of multi-level affiliations, like "
                            "'Department of Computer Science, TsingHua University', you should return "
                            "the top-level affiliation 'TsingHua University' only. Do not contain "
                            "duplicated affiliations. If there is no affiliation found, you should "
                            "return an empty list [ ]. You should only return the final list of "
                            "affiliations, and do not return any intermediate results."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get("generation_kwargs", {}),
            )

            affiliations = affiliations.choices[0].message.content
            affiliations = re.search(
                r"\[.*?\]",
                affiliations,
                flags=re.DOTALL,
            ).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            return [str(affiliation) for affiliation in affiliations]

        return None

    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(
                openai_client,
                llm_params,
            )
            self.affiliations = affiliations
            return affiliations
        except Exception as exc:
            logger.warning(
                f"Failed to generate affiliations of {self.url}: {exc}"
            )
            self.affiliations = None
            return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
