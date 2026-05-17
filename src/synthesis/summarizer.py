"""LLM summarization of top-scored papers.

For each shortlisted paper produces a plain-language summary, key contributions,
methodology notes, and connections to previously surfaced work.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

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

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "plain_language": {"type": "string"},
        "key_contributions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "methodology_notes": {"type": "string"},
        "connections": {"type": "string"},
    },
    "required": ["plain_language", "key_contributions", "methodology_notes", "connections"],
    "additionalProperties": False,
}


class Summarizer:
    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        model: str = "claude-opus-4-7",
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def summarize(self, paper: Paper, prior_titles: list[str]) -> DigestSummary:
        prior_block = (
            "Previously surfaced papers (for connection-finding):\n"
            + "\n".join(f"- {t}" for t in prior_titles[:30])
            if prior_titles
            else "Previously surfaced papers: (none yet)"
        )
        tldr_line = f"Semantic Scholar TLDR: {paper.tldr}\n" if paper.tldr else ""

        user_content = (
            f"{prior_block}\n\n"
            f"---\n"
            f"Paper to summarize:\n"
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors[:8])}{' et al.' if len(paper.authors) > 8 else ''}\n"
            f"Categories: {', '.join(paper.categories)}\n"
            f"{tldr_line}"
            f"Abstract: {paper.abstract}\n"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": SUMMARY_SCHEMA,
                }
            },
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        return DigestSummary(
            paper=paper,
            plain_language=data["plain_language"],
            key_contributions=data["key_contributions"],
            methodology_notes=data["methodology_notes"],
            connections=data["connections"],
        )
