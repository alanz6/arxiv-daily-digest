"""Dumb keyword-substring baseline scorer used for evaluation.

Scores a paper as the fraction of profile keywords that appear in title+abstract,
case-insensitively. This is the "what if we didn't use an LLM at all" reference.
"""
from __future__ import annotations

from typing import Iterable

from src.models import Paper


def score_paper(paper: Paper, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    haystack = (paper.title + " " + paper.abstract).lower()
    hits = sum(1 for kw in keywords if kw.lower() in haystack)
    return hits / len(keywords)


def score_papers(papers: Iterable[Paper], profile: dict) -> list[tuple[Paper, float]]:
    """Score every paper with the keyword baseline. Returns (paper, score) pairs."""
    keywords = profile.get("keywords", [])
    return [(p, score_paper(p, keywords)) for p in papers]
