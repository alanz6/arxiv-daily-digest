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
from src.llm import LLMClient
from src.models import DigestSummary, Paper, TrendCluster
from src.scoring.embedding_filter import EmbeddingFilter
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
    prefilter_kept: int = 0
    prefilter_dropped: int = 0

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
            "stats": {
                "new_papers": self.new_papers,
                "skipped": self.skipped,
                "prefilter_kept": self.prefilter_kept,
                "prefilter_dropped": self.prefilter_dropped,
            },
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
    llm: Optional[LLMClient] = None,
    on_progress=None,
) -> Digest:
    """Run the end-to-end digest pipeline.

    `on_progress`, if given, is called with `on_progress(event_type, message, **data)`
    at each stage and sub-step. Used by the webapp to stream live progress.
    """
    def emit(event_type: str, message: str, **data) -> None:
        logger.info("[%s] %s", event_type, message)
        if on_progress:
            on_progress(event_type, message, **data)

    profile = json.loads(profile_path.read_text())
    store = PaperStore(db_path)
    llm = llm or LLMClient()

    # 1. Ingest
    categories = profile.get("arxiv_categories", ["cs.CL", "cs.LG", "cs.AI"])
    emit("ingest_start", f"Fetching new papers from arXiv ({', '.join(categories)})...")
    raw = fetch_recent_papers(
        categories=categories,
        lookback_days=lookback_days,
        max_results_per_category=max_results_per_category,
    )
    fresh = store.filter_new(raw)
    emit("ingest_done", f"Fetched {len(raw)} papers, {len(fresh)} are new.", new=len(fresh), raw=len(raw))

    if not fresh:
        emit("complete", "No new papers to score.", new_papers=0, shortlisted=0)
        return Digest()

    # 2. Embedding pre-filter — keep only the top-K papers by semantic
    # similarity to the profile before paying the LLM cost. K is generous
    # (default 75) to preserve recall on the eventual shortlist.
    prefilter = EmbeddingFilter(profile=profile)
    prefilter_dropped: list[Paper] = []
    effective_k = prefilter.effective_k(len(fresh))
    if prefilter.enabled and len(fresh) > effective_k:
        emit(
            "prefilter_start",
            f"Embedding-ranking {len(fresh)} papers to keep top {effective_k}...",
            total=len(fresh),
            top_k=effective_k,
        )
        kept, prefilter_dropped = prefilter.filter(fresh)
        # Mark dropped papers so they show up in the DB (audit trail + so we
        # never re-score them on the next run via filter_new).
        for p in prefilter_dropped:
            p.relevance_score = 0.0
            p.relevance_rationale = "below embedding pre-filter cutoff (not LLM-scored)"
        emit(
            "prefilter_done",
            f"Kept {len(kept)} most relevant, dropped {len(prefilter_dropped)}.",
            kept=len(kept),
            dropped=len(prefilter_dropped),
        )
        scoring_set = kept
    else:
        scoring_set = fresh

    # 3. Score (no enrichment yet — Semantic Scholar is slow per-paper, so
    # we only enrich the shortlist below)
    import os as _os
    model = _os.environ.get("DO_INFERENCE_MODEL", "configured model")
    emit("score_start", f"Scoring {len(scoring_set)} papers against your profile using {model}...", total=len(scoring_set))
    scorer = RelevanceScorer(profile=profile, llm=llm)

    def _score_progress(done: int, total: int):
        emit("score_progress", f"Scored {done}/{total} papers", done=done, total=total)

    scorer.score(scoring_set, on_progress=_score_progress)
    emit("score_done", "Scoring complete.")

    # 4. Persist all papers (scored kept-set + audit-tagged drops)
    store.upsert_many(scoring_set)
    if prefilter_dropped:
        store.upsert_many(prefilter_dropped)

    # 5. Shortlist (only from the LLM-scored set — dropped papers can't be shortlisted)
    scoring_set.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
    shortlist: list[Paper] = [
        p for p in scoring_set if (p.relevance_score or 0.0) >= shortlist_threshold
    ][:shortlist_max]
    emit(
        "shortlist_done",
        f"Shortlisted {len(shortlist)} papers above relevance threshold {shortlist_threshold:.2f}.",
        shortlisted=len(shortlist),
    )

    # 5. Enrich just the shortlist (~10 papers), so S2 rate limits don't blow up.
    if enable_enrichment and shortlist:
        emit("enrich_start", "Enriching shortlist with Semantic Scholar (citations + TLDR)...")
        enrich_papers(shortlist, api_key=semantic_scholar_api_key)
        emit("enrich_done", "Enrichment complete.")

    # 6. Summarize each shortlisted paper — in parallel
    summarizer = Summarizer(llm=llm)
    prior_titles = store.previously_surfaced_titles()
    emit(
        "summarize_start",
        f"Summarizing {len(shortlist)} papers in parallel (concurrency={summarizer.concurrency})...",
        total=len(shortlist),
    )
    completed_count = [0]

    def _on_summary_complete(paper: Paper, summary, error) -> None:
        completed_count[0] += 1
        done = completed_count[0]
        if error is None and summary is not None:
            store.update_summary(paper.arxiv_id, summary.plain_language)
            emit(
                "summarize_progress",
                f"Completed {done}/{len(shortlist)}: {paper.title[:60]}{'...' if len(paper.title) > 60 else ''}",
                done=done,
                total=len(shortlist),
            )
        else:
            emit(
                "summarize_progress",
                f"Failed {done}/{len(shortlist)}: {paper.title[:60]}",
                done=done,
                total=len(shortlist),
            )

    summaries = summarizer.summarize_many(
        shortlist,
        prior_titles=prior_titles,
        on_each_complete=_on_summary_complete,
    )

    # 7. Trend radar
    trends: list[TrendCluster] = []
    if len(shortlist) >= 2:
        emit("trend_start", "Clustering papers into emerging themes...")
        try:
            radar = TrendRadar(llm=llm)
            trends = radar.cluster(shortlist)
            emit(
                "trend_done",
                f"Identified {len(trends)} thematic cluster{'s' if len(trends) != 1 else ''}.",
                clusters=len(trends),
            )
        except Exception as e:
            logger.warning("Trend radar failed: %s", e)
            emit("trend_done", "Trend radar skipped (error).", clusters=0)

    emit(
        "complete",
        f"Done. {len(fresh)} new papers, {len(summaries)} shortlisted, {len(trends)} trends.",
        new_papers=len(fresh),
        shortlisted=len(summaries),
        trends=len(trends),
    )
    return Digest(
        shortlist=summaries,
        trends=trends,
        new_papers=len(fresh),
        skipped=len(fresh) - len(shortlist),
        prefilter_kept=len(scoring_set),
        prefilter_dropped=len(prefilter_dropped),
    )
