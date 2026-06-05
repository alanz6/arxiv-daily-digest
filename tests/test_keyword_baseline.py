"""Tests for the keyword baseline scorer."""
from datetime import datetime

from src.models import Paper
from src.scoring.keyword_baseline import score_paper, score_papers


def _make(title: str, abstract: str) -> Paper:
    return Paper(
        arxiv_id="x",
        title=title,
        abstract=abstract,
        authors=[],
        categories=[],
        published=datetime.now(),
        updated=datetime.now(),
        pdf_url="",
        abs_url="",
    )


def test_no_keywords_returns_zero():
    p = _make("anything", "anything")
    assert score_paper(p, []) == 0.0


def test_all_keywords_match_returns_one():
    p = _make("agents and RAG", "uses RLHF and tool use")
    keywords = ["agent", "RAG", "RLHF", "tool use"]
    assert score_paper(p, keywords) == 1.0


def test_partial_match():
    p = _make("Agentic RAG systems", "abstract")
    keywords = ["agent", "RAG", "RLHF", "tool use"]
    # "agent" matches via "Agentic", "RAG" matches; RLHF and tool use don't
    assert score_paper(p, keywords) == 0.5


def test_case_insensitive():
    p = _make("AGENT-based AGENTIC system", "abstract")
    assert score_paper(p, ["agent"]) == 1.0


def test_score_papers_returns_pairs_in_input_order():
    papers = [_make("a", "agent"), _make("b", "nothing")]
    pairs = score_papers(papers, {"keywords": ["agent"]})
    assert len(pairs) == 2
    assert pairs[0][0].title == "a"
    assert pairs[0][1] == 1.0
    assert pairs[1][1] == 0.0
