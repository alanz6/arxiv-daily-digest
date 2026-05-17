"""End-to-end pipeline: ingest -> score -> shortlist -> summarize -> trend radar."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.ingestion.arxiv_client import fetch_recent_papers
from src.ingestion.semantic_scholar_client import enrich_papers
from src.ingestion.storage import PaperStore
from src.models import DigestSummary, Paper, TrendCluster
from src.scoring.relevance import RelevanceScorer
from src.synthesis.summarizer import Summarizer
from src.synthesis.trend_radar import TrendRadar

logger = logging.getLogger(__name__)


@dataclass
class Digest:
    shortlist: list[DigestSummary] = field(default_factory=list)
    trends: list[TrendCluster] = field(default_factory=list)
    skipped: int = 0
    new_papers: int = 0

    def to_dict(self) -> dict:
        return {
            "shortlist": [
                {
                    "arxiv_id": s.paper.arxiv_id,
                    "title": s.paper.title,
                    "authors": s.paper.authors,
                    "abs_url": s.paper.abs_url,
                    "relevance_score": s.paper.relevance_score,
                    "relevance_rationale": s.paper.relevance_rationale,
                    "citation_count": s.paper.citation_count,
                    "tldr": s.paper.tldr,
                    "plain_language": s.plain_language,
                    "key_contributions": s.key_contributions,
                    "methodology_notes": s.methodology_notes,
                    "connections": s.connections,
                }
                for s in self.shortlist
            ],
            "trends": [
                {
                    "theme": t.theme,
                    "paper_ids": t.paper_ids,
                    "why_it_matters": t.why_it_matters,
                }
                for t in self.trends
            ],
            "stats": {"new_papers": self.new_papers, "skipped": self.skipped},
        }


def run_pipeline(
    profile_path: Path,
    db_path: Path,
    lookback_days: int = 1,
    max_results_per_category: int = 100,
    shortlist_threshold: float = 0.6,
    shortlist_max: int = 10,
    semantic_scholar_api_key: Optional[str] = None,
    enable_enrichment: bool = True,
    model: str = "claude-opus-4-7",
) -> Digest:
    profile = json.loads(profile_path.read_text())
    store = PaperStore(db_path)

    # 1. Ingest
    categories = profile.get("arxiv_categories", ["cs.CL", "cs.LG", "cs.AI"])
    raw = fetch_recent_papers(
        categories=categories,
        lookback_days=lookback_days,
        max_results_per_category=max_results_per_category,
    )
    fresh = store.filter_new(raw)
    logger.info("Ingest: %d raw, %d new", len(raw), len(fresh))

    if not fresh:
        return Digest()

    if enable_enrichment:
        enrich_papers(fresh, api_key=semantic_scholar_api_key)

    # 2. Score
    scorer = RelevanceScorer(profile=profile, model=model)
    scorer.score(fresh)

    # 3. Persist all scored papers
    store.upsert_many(fresh)

    # 4. Shortlist
    fresh.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
    shortlist: list[Paper] = [
        p for p in fresh if (p.relevance_score or 0.0) >= shortlist_threshold
    ][:shortlist_max]
    logger.info("Shortlisted %d papers (threshold=%.2f)", len(shortlist), shortlist_threshold)

    # 5. Summarize each shortlisted paper
    summarizer = Summarizer(model=model)
    prior_titles = store.previously_surfaced_titles()
    summaries: list[DigestSummary] = []
    for p in shortlist:
        try:
            summary = summarizer.summarize(p, prior_titles=prior_titles)
            summaries.append(summary)
            store.update_summary(p.arxiv_id, summary.plain_language)
        except Exception as e:
            logger.warning("Summarize failed for %s: %s", p.arxiv_id, e)

    # 6. Trend radar
    trends: list[TrendCluster] = []
    if len(shortlist) >= 2:
        try:
            radar = TrendRadar(model=model)
            trends = radar.cluster(shortlist)
        except Exception as e:
            logger.warning("Trend radar failed: %s", e)

    return Digest(
        shortlist=summaries,
        trends=trends,
        new_papers=len(fresh),
        skipped=len(fresh) - len(shortlist),
    )
