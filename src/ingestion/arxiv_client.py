"""arXiv API client. Uses the `arxiv` library which wraps the public Atom feed."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import arxiv

from src.models import Paper

logger = logging.getLogger(__name__)


def _to_paper(result: arxiv.Result) -> Paper:
    arxiv_id = result.entry_id.split("/abs/")[-1]
    return Paper(
        arxiv_id=arxiv_id,
        title=result.title.strip().replace("\n", " "),
        abstract=result.summary.strip().replace("\n", " "),
        authors=[a.name for a in result.authors],
        categories=list(result.categories),
        published=result.published,
        updated=result.updated,
        pdf_url=result.pdf_url,
        abs_url=result.entry_id,
    )


def fetch_recent_papers(
    categories: list[str],
    lookback_days: int = 1,
    max_results_per_category: int = 100,
) -> list[Paper]:
    """Fetch papers submitted in the last `lookback_days` for the given categories.

    arXiv's API doesn't support a clean date-range query for "new submissions",
    so we sort by submittedDate and filter client-side.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

    papers: dict[str, Paper] = {}
    for category in categories:
        query = f"cat:{category}"
        search = arxiv.Search(
            query=query,
            max_results=max_results_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        logger.info("Fetching arXiv category %s", category)
        for result in client.results(search):
            if result.published < cutoff:
                break
            paper = _to_paper(result)
            papers[paper.arxiv_id] = paper

    logger.info("Fetched %d unique papers across %d categories", len(papers), len(categories))
    return list(papers.values())
