from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar
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

    # Backward-compatible field retained from the original project.
    # It stores the English TLDR in the bilingual workflow.
    tldr: Optional[str] = None

    # Bilingual fields used by the updated email renderer.
    title_zh: Optional[str] = None
    tldr_en: Optional[str] = None
    tldr_zh: Optional[str] = None

    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _truncate_prompt(self, prompt: str, max_tokens: int = 4000) -> str:
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        return enc.decode(prompt_tokens[:max_tokens])

    def _paper_context(self) -> str:
        parts = []

        if self.title:
            parts.append(f"Title:\n{self.title}")

        if self.abstract:
            parts.append(f"Abstract:\n{self.abstract}")

        if self.full_text:
            parts.append(f"Preview of main content:\n{self.full_text}")

        if not parts:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")

        return "\n\n".join(parts)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        text = Paper._strip_code_fence(text)

        # First try the whole response.
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

        # Then try the first JSON-looking object inside the response.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload

        raise ValueError("No valid JSON object found in LLM response")

    def _build_bilingual_prompt(self) -> str:
        prompt = (
            "Given the following scientific paper, return a JSON object with exactly "
            'three keys: "title_zh", "tldr_en", "tldr_zh".\n'
            "- title_zh: a faithful Chinese translation of the title.\n"
            "- tldr_en: one concise English sentence summarizing the core method and main contribution.\n"
            "- tldr_zh: one concise Chinese sentence summarizing the core method and main contribution.\n"
            "Do not return Markdown. Return valid JSON only.\n\n"
            f"{self._paper_context()}"
        )
        return self._truncate_prompt(prompt)

    def _build_line_fallback_prompt(self) -> str:
        prompt = (
            "Please summarize and translate the following scientific paper.\n"
            "Return exactly three lines and nothing else:\n"
            "TITLE_ZH: <Chinese title translation>\n"
            "TLDR_EN: <one concise English sentence>\n"
            "TLDR_ZH: <one concise Chinese sentence>\n\n"
            f"{self._paper_context()}"
        )
        return self._truncate_prompt(prompt)

    def _build_chinese_fallback_prompt(self) -> str:
        english_summary = self.tldr_en or self.tldr or self.abstract or ""
        prompt = (
            "Please translate the following scientific paper title into Chinese and "
            "write a concise Chinese one-sentence summary.\n"
            "Return exactly two lines and nothing else:\n"
            "TITLE_ZH: <Chinese title translation>\n"
            "TLDR_ZH: <one concise Chinese sentence>\n\n"
            f"Title:\n{self.title}\n\n"
            f"English summary or abstract:\n{english_summary}"
        )
        return self._truncate_prompt(prompt)

    def _chat(self, openai_client: OpenAI, llm_params: dict, prompt: str) -> str:
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who accurately summarizes scientific papers. "
                        "Follow the requested output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_labeled_lines(text: str) -> dict[str, str]:
        result = {}
        for line in (text or "").splitlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()

            if key in {"TITLE_ZH", "TLDR_EN", "TLDR_ZH"} and value:
                result[key.lower()] = value

        return result

    def generate_bilingual_summary(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> tuple[str | None, str | None, str | None]:
        """
        Generate:
        - Chinese title translation
        - English one-sentence TLDR
        - Chinese one-sentence TLDR

        The function intentionally uses multiple fallback layers because some
        OpenAI-compatible gateways do not reliably return strict JSON.
        """
        # 1) Preferred path: strict JSON response.
        try:
            raw = self._chat(
                openai_client,
                llm_params,
                self._build_bilingual_prompt(),
            )
            payload = self._parse_json_object(raw)

            self.title_zh = str(payload.get("title_zh", "")).strip() or None
            self.tldr_en = str(payload.get("tldr_en", "")).strip() or None
            self.tldr_zh = str(payload.get("tldr_zh", "")).strip() or None
        except Exception as exc:
            logger.warning(
                f"JSON bilingual summary failed for {self.url}: {exc}"
            )

        # 2) Fallback: line-based output, easier for weaker gateways to follow.
        if self.title_zh is None or self.tldr_en is None or self.tldr_zh is None:
            try:
                raw = self._chat(
                    openai_client,
                    llm_params,
                    self._build_line_fallback_prompt(),
                )
                payload = self._parse_labeled_lines(raw)

                self.title_zh = self.title_zh or payload.get("title_zh")
                self.tldr_en = self.tldr_en or payload.get("tldr_en")
                self.tldr_zh = self.tldr_zh or payload.get("tldr_zh")
            except Exception as exc:
                logger.warning(
                    f"Line-format bilingual summary failed for {self.url}: {exc}"
                )

        # 3) Final fallback for Chinese fields only.
        if self.title_zh is None or self.tldr_zh is None:
            try:
                raw = self._chat(
                    openai_client,
                    llm_params,
                    self._build_chinese_fallback_prompt(),
                )
                payload = self._parse_labeled_lines(raw)

                self.title_zh = self.title_zh or payload.get("title_zh")
                self.tldr_zh = self.tldr_zh or payload.get("tldr_zh")
            except Exception as exc:
                logger.warning(
                    f"Chinese fallback translation failed for {self.url}: {exc}"
                )

        # Keep the original field populated for compatibility with old code/tests.
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
            prompt += f"Title:\n{self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract:\n{self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n{self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        prompt = self._truncate_prompt(prompt)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who perfectly summarizes scientific papers "
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
            logger.warning(f"Failed to generate TLDR of {self.url}: {exc}")
            self.tldr = self.abstract
            return self.tldr

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        if self.full_text is None:
            return None

        prompt = (
            "Given the beginning of a paper, extract the affiliations of the authors "
            "in a Python list format, sorted by author order. "
            "If no affiliation is found, return an empty list []:\n\n"
            f"{self.full_text}"
        )
        prompt = self._truncate_prompt(prompt, max_tokens=2000)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract author affiliations from scientific papers. "
                        'Return only a Python/JSON list, e.g. ["Tsinghua University", '
                        '"Peking University"]. If an affiliation has multiple levels, '
                        "return only the top-level institution. Remove duplicates. "
                        "If none are found, return []."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )

        affiliations_text = response.choices[0].message.content or ""
        match = re.search(r"\[.*?\]", affiliations_text, flags=re.DOTALL)
        if match is None:
            raise ValueError("No list found in affiliation response")

        affiliations = json.loads(match.group(0))
        return list(dict.fromkeys(str(item) for item in affiliations))

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
