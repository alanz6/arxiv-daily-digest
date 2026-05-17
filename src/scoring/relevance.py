"""LLM-based relevance scoring against a user interest profile.

Each paper is scored 0-1 against the profile. The profile sits in the cached
system prompt so repeated requests amortize its cost.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

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

SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "score": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["arxiv_id", "score", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


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


class RelevanceScorer:
    def __init__(
        self,
        profile: dict,
        client: Optional[anthropic.Anthropic] = None,
        model: str = "claude-opus-4-7",
        batch_size: int = 10,
    ):
        self.profile = profile
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.batch_size = batch_size
        self.system_text = _format_profile(profile)

    def score(self, papers: list[Paper]) -> list[Paper]:
        """Mutate papers in-place with relevance_score and relevance_rationale."""
        if not papers:
            return papers

        by_id = {p.arxiv_id: p for p in papers}
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i : i + self.batch_size]
            logger.info("Scoring batch %d-%d of %d", i, i + len(batch), len(papers))
            try:
                scores = self._score_batch(batch)
            except Exception as e:
                logger.warning("Scoring batch failed: %s; defaulting batch to 0", e)
                for p in batch:
                    p.relevance_score = 0.0
                    p.relevance_rationale = f"scoring failed: {e}"
                continue

            for entry in scores:
                paper = by_id.get(entry["arxiv_id"])
                if paper is None:
                    continue
                paper.relevance_score = float(entry["score"])
                paper.relevance_rationale = entry["rationale"]

            # Backfill any paper the model dropped
            for p in batch:
                if p.relevance_score is None:
                    p.relevance_score = 0.0
                    p.relevance_rationale = "no score returned"

        return papers

    def _score_batch(self, batch: list[Paper]) -> list[dict]:
        user_content = (
            "Score these papers. Return one entry per paper with the exact arxiv_id given.\n\n"
            + _format_papers(batch)
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": self.system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": SCORING_SCHEMA,
                }
            },
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            raise ValueError("model returned no text")
        data = json.loads(text)
        return data["scores"]
