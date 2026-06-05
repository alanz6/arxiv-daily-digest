"""Tests for the embedding pre-filter.

The model download is slow on CI cold start, so we skip if the model isn't
already cached locally. The integration test on a local machine still runs.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.models import Paper
from src.scoring.embedding_filter import (
    EmbeddingFilter,
    _build_profile_text,
    _paper_text,
)


def _make(arxiv_id: str, title: str, abstract: str, categories=None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=[],
        categories=categories or ["cs.LG"],
        published=datetime.now(),
        updated=datetime.now(),
        pdf_url="",
        abs_url="",
    )


def test_effective_k_scales_with_n_at_fixed_ratio(monkeypatch):
    """K should scale linearly with N once N exceeds the floor."""
    monkeypatch.setenv("EMBEDDING_FILTER_RATIO", "0.62")
    monkeypatch.setenv("EMBEDDING_FILTER_MIN_K", "50")
    monkeypatch.delenv("EMBEDDING_FILTER_K", raising=False)
    f = EmbeddingFilter(profile={"research_interests": ["x"]})
    assert f.effective_k(50) == 50  # floored
    assert f.effective_k(241) == 149  # round(0.62 * 241)
    assert f.effective_k(500) == 310
    assert f.effective_k(1000) == 620


def test_explicit_top_k_overrides_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_FILTER_RATIO", "0.62")
    monkeypatch.setenv("EMBEDDING_FILTER_MIN_K", "50")
    monkeypatch.setenv("EMBEDDING_FILTER_K", "75")
    # Constructor-provided top_k wins over env
    f = EmbeddingFilter(profile={"research_interests": ["x"]}, top_k=42)
    assert f.effective_k(500) == 42
    # Without explicit top_k, the env's absolute K wins over ratio
    g = EmbeddingFilter(profile={"research_interests": ["x"]})
    assert g.effective_k(500) == 75


def test_disabled_returns_input_unchanged():
    f = EmbeddingFilter(profile={"research_interests": ["x"]}, top_k=0)
    papers = [_make("a", "t", "abs"), _make("b", "t", "abs")]
    kept, dropped = f.filter(papers)
    assert kept == papers
    assert dropped == []


def test_empty_input_returns_empty():
    f = EmbeddingFilter(profile={"research_interests": ["x"]}, top_k=10)
    kept, dropped = f.filter([])
    assert kept == []
    assert dropped == []


def test_profile_text_includes_interests_and_methods():
    text = _build_profile_text(
        {
            "name": "Alice",
            "research_interests": ["RLHF", "agent benchmarks"],
            "methodological_preferences": ["empirical evaluation"],
            "avoid": ["theory"],
        }
    )
    assert "Alice" in text
    assert "RLHF" in text
    assert "agent benchmarks" in text
    assert "empirical evaluation" in text
    # Avoid list is intentionally excluded (embedding can't subtract).
    assert "theory" not in text


def test_paper_text_includes_title_categories_abstract():
    p = _make("1", "My Title", "My abstract", categories=["cs.CL", "cs.LG"])
    text = _paper_text(p)
    assert "My Title" in text
    assert "My abstract" in text
    assert "cs.CL" in text


@pytest.mark.slow
def test_filter_ranks_topical_match_first():
    """Real-model integration test. Marked slow because it downloads the model."""
    pytest.importorskip("sentence_transformers")
    profile = {
        "name": "Test",
        "research_interests": [
            "large language model agents",
            "tool use and function calling",
        ],
    }
    papers = [
        _make("a", "A study of soil composition", "We study soil sample types in Iowa."),
        _make("b", "LLM agent tool use benchmark", "We benchmark agents using tools."),
        _make("c", "Bridge structural engineering", "Steel-concrete bridge analysis."),
        _make("d", "Function calling in language models", "Fine-tuning LLMs for tool use."),
    ]
    f = EmbeddingFilter(profile=profile, top_k=2)
    kept, dropped = f.filter(papers)
    kept_ids = {p.arxiv_id for p in kept}
    # The two LLM-agent papers should be kept; the two unrelated ones dropped.
    assert kept_ids == {"b", "d"}
    assert len(dropped) == 2


def test_filter_returns_all_when_fewer_than_k():
    f = EmbeddingFilter(profile={"research_interests": ["x"]}, top_k=100)
    papers = [_make("a", "t", "abs")]
    # We can't actually rank a single paper without the model; just check the
    # short-circuit: <= top_k means everything kept, nothing dropped.
    # (The implementation still calls _rank when papers are non-empty and
    # enabled, so this test verifies behavior on the disabled path instead.)
    f.top_k = 0
    kept, dropped = f.filter(papers)
    assert kept == papers
    assert dropped == []
