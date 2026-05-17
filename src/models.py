from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    abs_url: str

    semantic_scholar_id: Optional[str] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    tldr: Optional[str] = None

    relevance_score: Optional[float] = None
    relevance_rationale: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat()
        d["updated"] = self.updated.isoformat()
        return d


@dataclass
class ScoredPaper:
    paper: Paper
    score: float
    rationale: str


@dataclass
class DigestSummary:
    paper: Paper
    plain_language: str
    key_contributions: list[str]
    methodology_notes: str
    connections: str


@dataclass
class TrendCluster:
    theme: str
    paper_ids: list[str]
    why_it_matters: str
