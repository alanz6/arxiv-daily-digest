"""Tests for the SQLite storage layer."""
import tempfile
from datetime import datetime
from pathlib import Path

from src.ingestion.storage import PaperStore
from src.models import Paper


def make_paper(arxiv_id: str = "2606.00001v1", **overrides) -> Paper:
    defaults = dict(
        arxiv_id=arxiv_id,
        title="Test Title",
        abstract="Test abstract.",
        authors=["Author One", "Author Two"],
        categories=["cs.CL"],
        published=datetime(2026, 6, 1, 12, 0, 0),
        updated=datetime(2026, 6, 1, 12, 0, 0),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )
    defaults.update(overrides)
    return Paper(**defaults)


def test_upsert_returns_new_and_existing_counts():
    with tempfile.TemporaryDirectory() as tmp:
        store = PaperStore(Path(tmp) / "test.db")
        p1 = make_paper("2606.00001v1")
        p2 = make_paper("2606.00002v1")

        new, existing = store.upsert_many([p1, p2])
        assert new == 2
        assert existing == 0

        # Second insert of the same papers: no new
        new, existing = store.upsert_many([p1, p2])
        assert new == 0
        assert existing == 2


def test_filter_new_removes_seen_papers():
    with tempfile.TemporaryDirectory() as tmp:
        store = PaperStore(Path(tmp) / "test.db")
        p1 = make_paper("2606.00001v1")
        store.upsert_many([p1])

        p2 = make_paper("2606.00002v1")
        fresh = store.filter_new([p1, p2])
        assert len(fresh) == 1
        assert fresh[0].arxiv_id == "2606.00002v1"


def test_load_all_round_trips_fields():
    with tempfile.TemporaryDirectory() as tmp:
        store = PaperStore(Path(tmp) / "test.db")
        p = make_paper(
            "2606.00099v1",
            relevance_score=0.85,
            relevance_rationale="strong fit",
            citation_count=5,
            tldr="A tldr from semantic scholar",
        )
        store.upsert_many([p])

        loaded = store.load_all()
        assert len(loaded) == 1
        got = loaded[0]
        assert got.arxiv_id == "2606.00099v1"
        assert got.title == "Test Title"
        assert got.authors == ["Author One", "Author Two"]
        assert got.categories == ["cs.CL"]
        assert got.relevance_score == 0.85
        assert got.citation_count == 5
        assert got.tldr == "A tldr from semantic scholar"


def test_previously_surfaced_titles_filters_low_scores():
    with tempfile.TemporaryDirectory() as tmp:
        store = PaperStore(Path(tmp) / "test.db")
        store.upsert_many([
            make_paper("a1", title="high score paper", relevance_score=0.9),
            make_paper("a2", title="medium paper", relevance_score=0.5),
            make_paper("a3", title="another high paper", relevance_score=0.7),
        ])
        titles = store.previously_surfaced_titles()
        # Only papers >= 0.6 should appear
        assert "high score paper" in titles
        assert "another high paper" in titles
        assert "medium paper" not in titles
