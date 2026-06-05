"""Tests for the bookmark storage layer."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.ingestion.storage import PaperStore
from src.models import Paper


def _make(arxiv_id: str, title: str = "t") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="abs",
        authors=["Author One"],
        categories=["cs.LG"],
        published=datetime.now(),
        updated=datetime.now(),
        pdf_url="",
        abs_url="",
        relevance_score=0.8,
        relevance_rationale="matches",
        summary="plain language summary",
    )


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "papers.db")
    s.upsert_many([_make("a"), _make("b"), _make("c")])
    return s


def test_toggle_adds_then_removes(store):
    assert store.toggle_bookmark("a") is True
    assert "a" in store.bookmarked_ids()
    assert store.toggle_bookmark("a") is False
    assert "a" not in store.bookmarked_ids()


def test_bookmarked_ids_empty_by_default(store):
    assert store.bookmarked_ids() == set()


def test_load_bookmarked_newest_first(store):
    store.toggle_bookmark("a")
    store.toggle_bookmark("b")
    store.toggle_bookmark("c")
    entries = store.load_bookmarked()
    ids = [p.arxiv_id for p, _ in entries]
    # 'c' bookmarked last, so it should be first
    assert ids[0] == "c"
    assert set(ids) == {"a", "b", "c"}


def test_load_bookmarked_returns_full_paper(store):
    store.toggle_bookmark("a")
    entries = store.load_bookmarked()
    assert len(entries) == 1
    p, when = entries[0]
    assert p.title == "t"
    assert p.relevance_score == 0.8
    assert p.summary == "plain language summary"
    assert when  # ISO timestamp string


def test_toggle_unknown_paper_still_works(store):
    # Bookmark a paper that doesn't exist in `papers`. The bookmark row
    # gets inserted (FK isn't enforced in SQLite by default), but the join
    # in load_bookmarked silently drops it.
    assert store.toggle_bookmark("nope") is True
    entries = store.load_bookmarked()
    assert all(p.arxiv_id != "nope" for p, _ in entries)
