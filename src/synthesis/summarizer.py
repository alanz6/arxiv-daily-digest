"""LLM summarization of top-scored papers.

Public API:
  - `summarize(paper, prior_titles)` — single paper, sync.
  - `summarize_many(papers, prior_titles, concurrency, on_each_complete)` —
    multiple papers in parallel via asyncio + semaphore. This is what the
    pipeline uses; cuts wall-clock time on a 10-paper shortlist from
    ~80s sequential to ~16s at concurrency=5.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Optional

from src.llm import LLMClient
from src.models import DigestSummary, Paper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant writing daily briefings of new arXiv papers for a researcher.

For each paper, produce:
- plain_language: A 2-3 sentence summary in plain language (no jargon) that captures what the paper does and why it matters.
- key_contributions: 2-4 short bullet points of the paper's main contributions, as the authors would describe them.
- methodology_notes: 1-2 sentences about how they did it (datasets, model sizes, evaluation setup, anything notable).
- connections: 1-2 sentences relating this work to the researcher's previously-surfaced papers (provided below). If none of the prior papers connect, say "No direct connection to prior surfaced work."

Be accurate. Do not invent results that aren't in the abstract.
"""

SUMMARY_SCHEMA_HINT = """{
  "plain_language": "<string>",
  "key_contributions": ["<string>", "<string>"],
  "methodology_notes": "<string>",
  "connections": "<string>"
}"""


def _default_concurrency() -> int:
    raw = os.environ.get("SUMMARIZE_CONCURRENCY", "5")
    try:
        n = int(raw)
    except ValueError:
        n = 5
    return max(1, min(10, n))


def _build_user_content(paper: Paper, prior_titles: list[str]) -> str:
    prior_block = (
        "Previously surfaced papers (for connection-finding):\n"
        + "\n".join(f"- {t}" for t in prior_titles[:30])
        if prior_titles
        else "Previously surfaced papers: (none yet)"
    )
    tldr_line = f"Semantic Scholar TLDR: {paper.tldr}\n" if paper.tldr else ""
    return (
        f"{prior_block}\n\n"
        f"---\n"
        f"Paper to summarize:\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors[:8])}{' et al.' if len(paper.authors) > 8 else ''}\n"
        f"Categories: {', '.join(paper.categories)}\n"
        f"{tldr_line}"
        f"Abstract: {paper.abstract}\n"
    )


def _build_summary_from_response(paper: Paper, data: dict) -> DigestSummary:
    contributions = data.get("key_contributions", [])
    if not isinstance(contributions, list):
        contributions = [str(contributions)]
    return DigestSummary(
        paper=paper,
        plain_language=data.get("plain_language", ""),
        key_contributions=[str(c) for c in contributions],
        methodology_notes=data.get("methodology_notes", ""),
        connections=data.get("connections", ""),
    )


class Summarizer:
    def __init__(self, llm: Optional[LLMClient] = None, concurrency: Optional[int] = None):
        self.llm = llm or LLMClient()
        self.concurrency = concurrency if concurrency is not None else _default_concurrency()

    def summarize(self, paper: Paper, prior_titles: list[str]) -> DigestSummary:
        """Sync single-paper summarization. Kept for CLI and tests."""
        user_content = _build_user_content(paper, prior_titles)
        data = self.llm.chat_json(
            system=SYSTEM_PROMPT,
            user=user_content,
            schema_hint=SUMMARY_SCHEMA_HINT,
            max_tokens=4096,
        )
        return _build_summary_from_response(paper, data)

    async def summarize_async(self, paper: Paper, prior_titles: list[str]) -> DigestSummary:
        user_content = _build_user_content(paper, prior_titles)
        # Smaller open-source models (e.g. gpt-oss-20b) occasionally truncate
        # mid-response — not a max_tokens issue, just a model quirk. Retry once
        # on parse failure before giving up.
        try:
            data = await self.llm.chat_json_async(
                system=SYSTEM_PROMPT,
                user=user_content,
                schema_hint=SUMMARY_SCHEMA_HINT,
                max_tokens=4096,
            )
        except ValueError as e:
            logger.warning("Summarize parse failed for %s on attempt 1: %s; retrying once", paper.arxiv_id, e)
            data = await self.llm.chat_json_async(
                system=SYSTEM_PROMPT,
                user=user_content,
                schema_hint=SUMMARY_SCHEMA_HINT,
                max_tokens=4096,
            )
        return _build_summary_from_response(paper, data)

    def summarize_many(
        self,
        papers: list[Paper],
        prior_titles: list[str],
        on_each_complete: Optional[Callable[[Paper, Optional[DigestSummary], Optional[Exception]], None]] = None,
    ) -> list[DigestSummary]:
        """Summarize multiple papers in parallel.

        Returns successful summaries in input order. Failures are logged and
        passed to `on_each_complete` as the third arg; they do not appear in
        the returned list.
        """
        if not papers:
            return []
        return asyncio.run(self._summarize_many_async(papers, prior_titles, on_each_complete))

    async def _summarize_many_async(
        self,
        papers: list[Paper],
        prior_titles: list[str],
        on_each_complete: Optional[Callable[[Paper, Optional[DigestSummary], Optional[Exception]], None]],
    ) -> list[DigestSummary]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(paper: Paper) -> Optional[DigestSummary]:
            async with semaphore:
                try:
                    summary = await self.summarize_async(paper, prior_titles)
                except Exception as e:
                    logger.warning("Summarize failed for %s: %s", paper.arxiv_id, e)
                    if on_each_complete:
                        on_each_complete(paper, None, e)
                    return None
                if on_each_complete:
                    on_each_complete(paper, summary, None)
                return summary

        results = await asyncio.gather(*(one(p) for p in papers))
        return [r for r in results if r is not None]
