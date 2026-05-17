"""Trend radar: cluster the shortlisted papers into themes."""
from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

from src.models import Paper, TrendCluster

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research analyst spotting emerging themes in newly published arXiv papers.

Given a list of papers, identify 2-5 thematic clusters where multiple papers share a research direction, technique, or problem. A cluster needs at least 2 papers.

For each cluster:
- theme: A short noun-phrase naming the trend (e.g., "verifier-guided agent training")
- paper_ids: arXiv IDs of the papers in the cluster
- why_it_matters: 1-2 sentences explaining the broader implication

Be selective. Only surface clusters that represent a real common direction, not loose topical groupings. If fewer than 2 clusters meet the bar, return fewer.
"""

RADAR_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                    "why_it_matters": {"type": "string"},
                },
                "required": ["theme", "paper_ids", "why_it_matters"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clusters"],
    "additionalProperties": False,
}


class TrendRadar:
    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        model: str = "claude-opus-4-7",
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def cluster(self, papers: list[Paper]) -> list[TrendCluster]:
        if len(papers) < 2:
            return []

        listing = "\n---\n".join(
            f"arxiv_id: {p.arxiv_id}\ntitle: {p.title}\nabstract: {p.abstract}"
            for p in papers
        )
        user_content = f"Find thematic clusters in these papers:\n\n{listing}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": RADAR_SCHEMA,
                }
            },
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        return [
            TrendCluster(
                theme=c["theme"],
                paper_ids=c["paper_ids"],
                why_it_matters=c["why_it_matters"],
            )
            for c in data["clusters"]
        ]
