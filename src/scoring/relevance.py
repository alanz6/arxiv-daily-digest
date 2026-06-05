"""LLM-based relevance scoring against a user interest profile.

Each paper is scored 0-1 against the profile. Batches run concurrently via
asyncio + a semaphore so wall-clock time is bounded by the slowest batch
× ceil(num_batches / concurrency), not the sum of batch times.

The public `score()` method is sync — it dispatches to `_score_async` internally
via `asyncio.run`. Callers (the pipeline, the CLI) don't need to know.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Optional

from src.llm import LLMClient
from src.models import Paper

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are a research librarian helping a researcher triage arXiv papers.

The researcher's profile:
- Name: {name}
- Research interests:
{interests}
- Methodological preferences:
{methods}
- Topics to deprioritize:
{avoid}
- Past ratings (papers they liked or disliked):
{ratings}

For each paper you are shown, output a relevance score between 0.0 and 1.0 and a one-sentence rationale.

Scoring guidance:
- 0.9-1.0: Directly addresses a stated research interest with a methodological approach the researcher prefers
- 0.7-0.9: Strongly related to stated interests; the researcher should read this
- 0.5-0.7: Adjacent to interests; worth a skim
- 0.3-0.5: Tangentially related; mention only as background
- 0.0-0.3: Off-topic or in the deprioritized list

Be honest. Most papers should score below 0.5. Reserve high scores for genuine matches.
"""

SCORING_SCHEMA_HINT = """{
  "scores": [
    {"arxiv_id": "<string matching the input>", "score": <number 0-1>, "rationale": "<one sentence>"}
  ]
}"""


def _format_profile(profile: dict) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"  * {x}" for x in items) if items else "  (none)"

    ratings = profile.get("rating_history", [])
    rating_lines = []
    for r in ratings[-20:]:
        verdict = r.get("verdict", "?")
        title = r.get("title", "")
        rating_lines.append(f"  * [{verdict}] {title}")
    ratings_text = "\n".join(rating_lines) if rating_lines else "  (no ratings yet)"

    return SYSTEM_TEMPLATE.format(
        name=profile.get("name", "Researcher"),
        interests=bullets(profile.get("research_interests", [])),
        methods=bullets(profile.get("methodological_preferences", [])),
        avoid=bullets(profile.get("avoid", [])),
        ratings=ratings_text,
    )


def _format_papers(papers: list[Paper]) -> str:
    blocks = []
    for p in papers:
        block = (
            f"arxiv_id: {p.arxiv_id}\n"
            f"title: {p.title}\n"
            f"categories: {', '.join(p.categories)}\n"
            f"abstract: {p.abstract}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def _default_concurrency() -> int:
    raw = os.environ.get("SCORING_CONCURRENCY", "3")
    try:
        n = int(raw)
    except ValueError:
        n = 3
    # Hard upper bound: DigitalOcean publishes ~250 req/min for Agent Inference
    # (no published number for Serverless Inference, but real cap is account-
    # tier dependent). 60 is conservative headroom; benchmark up to confirm.
    return max(1, min(60, n))


def _default_batch_size() -> int:
    raw = os.environ.get("SCORING_BATCH_SIZE", "10")
    try:
        n = int(raw)
    except ValueError:
        n = 10
    return max(1, min(50, n))


class RelevanceScorer:
    def __init__(
        self,
        profile: dict,
        llm: Optional[LLMClient] = None,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        self.profile = profile
        self.llm = llm or LLMClient()
        self.batch_size = batch_size if batch_size is not None else _default_batch_size()
        # Cap concurrency: too high → DO rate-limits or 5xxs;
        # too low → wastes the parallelism win. Default 3, tunable via env.
        self.concurrency = concurrency if concurrency is not None else _default_concurrency()
        self.system_text = _format_profile(profile)

    def score(
        self,
        papers: list[Paper],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Paper]:
        """Mutate papers in-place with relevance_score and relevance_rationale.

        Public sync API. Dispatches to the async implementation internally so
        batches run concurrently bounded by `self.concurrency`.

        `on_progress`, if given, is called as on_progress(done, total) after
        each batch completes. Note: with concurrency > 1, batches may finish
        out of order — `done` is monotonic but the order of fired progress
        callbacks doesn't correspond to batch index.
        """
        if not papers:
            return papers
        asyncio.run(self._score_async(papers, on_progress))
        return papers

    async def _score_async(
        self,
        papers: list[Paper],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> None:
        by_id = {p.arxiv_id: p for p in papers}
        batches: list[tuple[int, list[Paper]]] = []
        for i in range(0, len(papers), self.batch_size):
            batches.append((i, papers[i : i + self.batch_size]))

        semaphore = asyncio.Semaphore(self.concurrency)
        completed = 0
        total = len(papers)

        async def process_batch(idx: int, batch: list[Paper]) -> None:
            nonlocal completed
            async with semaphore:
                logger.info(
                    "Scoring batch %d-%d of %d (concurrency=%d)",
                    idx, idx + len(batch), total, self.concurrency,
                )
                try:
                    scores = await self._score_batch_async(batch)
                except Exception as e:
                    logger.warning("Scoring batch failed: %s; defaulting batch to 0", e)
                    for p in batch:
                        p.relevance_score = 0.0
                        p.relevance_rationale = f"scoring failed: {e}"
                else:
                    for entry in scores:
                        paper = by_id.get(entry.get("arxiv_id"))
                        if paper is None:
                            continue
                        try:
                            paper.relevance_score = float(entry["score"])
                        except (TypeError, ValueError):
                            paper.relevance_score = 0.0
                        paper.relevance_rationale = entry.get("rationale", "")
                    for p in batch:
                        if p.relevance_score is None:
                            p.relevance_score = 0.0
                            p.relevance_rationale = "no score returned"

                # Single-threaded event loop — no lock needed
                completed += len(batch)
                if on_progress:
                    on_progress(completed, total)

        await asyncio.gather(*(process_batch(i, b) for i, b in batches))

    async def _score_batch_async(self, batch: list[Paper]) -> list[dict]:
        user_content = (
            "Score these papers. Return one entry per paper with the exact arxiv_id given.\n\n"
            + _format_papers(batch)
        )
        # temperature=0 for scoring: relevance is a ranking task where we want
        # deterministic, stable scores across reruns. Verified empirically —
        # at temperature=0.2 the self-consistency top-10 Jaccard was 0.43;
        # at 0 it should approach 1.0 (see docs/evaluation.md).
        data = await self.llm.chat_json_async(
            system=self.system_text,
            user=user_content,
            schema_hint=SCORING_SCHEMA_HINT,
            max_tokens=4096,
            temperature=0.0,
        )
        scores = data.get("scores", [])
        if not isinstance(scores, list):
            raise ValueError(f"expected 'scores' array, got {type(scores).__name__}")
        return scores
