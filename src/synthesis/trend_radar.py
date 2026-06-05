"""Trend radar: cluster the shortlisted papers into themes."""
from __future__ import annotations

import logging
from typing import Optional

from src.llm import LLMClient
from src.models import Paper, TrendCluster

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research analyst spotting emerging themes in newly published arXiv papers.

Given a list of papers, identify 2-5 thematic clusters where multiple papers share a research direction, technique, or problem. A cluster needs at least 2 papers.

For each cluster:
- theme: A short noun-phrase naming the trend (e.g., "verifier-guided agent training")
- paper_ids: arXiv IDs of the papers in the cluster
- why_it_matters: 1-2 sentences explaining the broader implication

Be selective. Only surface clusters that represent a real common direction, not loose topical groupings. If fewer than 2 clusters meet the bar, return fewer (an empty array is fine).
"""

RADAR_SCHEMA_HINT = """{
  "clusters": [
    {"theme": "<string>", "paper_ids": ["<arxiv_id>"], "why_it_matters": "<string>"}
  ]
}"""


class TrendRadar:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def cluster(self, papers: list[Paper]) -> list[TrendCluster]:
        if len(papers) < 2:
            return []

        listing = "\n---\n".join(
            f"arxiv_id: {p.arxiv_id}\ntitle: {p.title}\nabstract: {p.abstract}"
            for p in papers
        )
        user_content = f"Find thematic clusters in these papers:\n\n{listing}"

        data = self.llm.chat_json(
            system=SYSTEM_PROMPT,
            user=user_content,
            schema_hint=RADAR_SCHEMA_HINT,
            max_tokens=2048,
        )

        clusters = data.get("clusters", [])
        if not isinstance(clusters, list):
            return []
        return [
            TrendCluster(
                theme=str(c.get("theme", "")),
                paper_ids=[str(x) for x in c.get("paper_ids", [])],
                why_it_matters=str(c.get("why_it_matters", "")),
            )
            for c in clusters
            if c.get("theme")
        ]
