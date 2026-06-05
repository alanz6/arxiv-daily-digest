"""Semantic Scholar enrichment.

Used to enrich arXiv papers with citation counts and TLDRs when available.
For brand-new papers, citation data will be sparse/zero — that's expected; we
mostly use this to flag "sleeper" papers later in the pipeline.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

import requests

from src.models import Paper

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,citationCount,influentialCitationCount,tldr"


def enrich_papers(papers: Iterable[Paper], api_key: Optional[str] = None) -> None:
    """Mutate papers in-place with Semantic Scholar metadata.

    Uses the arXiv ID lookup endpoint. Silently skips papers S2 hasn't indexed yet.
    Bails out of rate-limit retries quickly so we don't block the pipeline.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    consecutive_rate_limits = 0
    for paper in papers:
        if consecutive_rate_limits >= 5:
            logger.warning("S2 rate-limit persists; skipping remaining enrichment")
            return
        url = f"{S2_BASE}/paper/arXiv:{paper.arxiv_id}"
        try:
            resp = requests.get(url, params={"fields": FIELDS}, headers=headers, timeout=10)
            if resp.status_code == 404:
                consecutive_rate_limits = 0
                continue
            if resp.status_code == 429:
                consecutive_rate_limits += 1
                logger.warning("S2 rate-limited (%d in a row)", consecutive_rate_limits)
                time.sleep(2)
                continue
            resp.raise_for_status()
            consecutive_rate_limits = 0
            data = resp.json()
            paper.semantic_scholar_id = data.get("paperId")
            paper.citation_count = data.get("citationCount")
            paper.influential_citation_count = data.get("influentialCitationCount")
            tldr = data.get("tldr")
            if tldr and isinstance(tldr, dict):
                paper.tldr = tldr.get("text")
        except requests.RequestException as e:
            logger.warning("S2 fetch failed for %s: %s", paper.arxiv_id, e)
        time.sleep(1.0 if not api_key else 0.1)
