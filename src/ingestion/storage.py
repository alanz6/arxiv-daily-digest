"""SQLite storage with arxiv_id as the dedup key."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from src.models import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    published TEXT NOT NULL,
    updated TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    abs_url TEXT NOT NULL,
    semantic_scholar_id TEXT,
    citation_count INTEGER,
    influential_citation_count INTEGER,
    tldr TEXT,
    relevance_score REAL,
    relevance_rationale TEXT,
    summary TEXT,
    first_seen TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_relevance ON papers(relevance_score);
CREATE INDEX IF NOT EXISTS idx_papers_first_seen ON papers(first_seen);

CREATE TABLE IF NOT EXISTS bookmarks (
    arxiv_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
);

CREATE INDEX IF NOT EXISTS idx_bookmarks_created ON bookmarks(created_at);
"""


class PaperStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_many(self, papers: Iterable[Paper]) -> tuple[int, int]:
        """Insert new papers, leave existing ones untouched. Returns (new, seen_existing)."""
        new_count = 0
        existing_count = 0
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for p in papers:
                row = conn.execute(
                    "SELECT 1 FROM papers WHERE arxiv_id = ?", (p.arxiv_id,)
                ).fetchone()
                if row:
                    existing_count += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO papers (
                        arxiv_id, title, abstract, authors, categories,
                        published, updated, pdf_url, abs_url,
                        semantic_scholar_id, citation_count, influential_citation_count, tldr,
                        relevance_score, relevance_rationale, summary, first_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p.arxiv_id, p.title, p.abstract,
                        json.dumps(p.authors), json.dumps(p.categories),
                        p.published.isoformat(), p.updated.isoformat(),
                        p.pdf_url, p.abs_url,
                        p.semantic_scholar_id, p.citation_count,
                        p.influential_citation_count, p.tldr,
                        p.relevance_score, p.relevance_rationale, p.summary,
                        now,
                    ),
                )
                new_count += 1
        return new_count, existing_count

    def filter_new(self, papers: Iterable[Paper]) -> list[Paper]:
        """Return only papers we haven't seen before."""
        with self._connect() as conn:
            fresh = []
            for p in papers:
                row = conn.execute(
                    "SELECT 1 FROM papers WHERE arxiv_id = ?", (p.arxiv_id,)
                ).fetchone()
                if not row:
                    fresh.append(p)
            return fresh

    def update_scoring(self, arxiv_id: str, score: float, rationale: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET relevance_score = ?, relevance_rationale = ? WHERE arxiv_id = ?",
                (score, rationale, arxiv_id),
            )

    def update_summary(self, arxiv_id: str, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET summary = ? WHERE arxiv_id = ?",
                (summary, arxiv_id),
            )

    def load_all(self) -> list[Paper]:
        """Load every paper in the DB back into Paper objects.

        Used by the offline evaluation script so we don't re-fetch arXiv.
        """
        import json as _json
        from datetime import datetime as _dt

        out: list[Paper] = []
        with self._connect() as conn:
            for r in conn.execute("SELECT * FROM papers"):
                p = Paper(
                    arxiv_id=r["arxiv_id"],
                    title=r["title"],
                    abstract=r["abstract"],
                    authors=_json.loads(r["authors"]),
                    categories=_json.loads(r["categories"]),
                    published=_dt.fromisoformat(r["published"]),
                    updated=_dt.fromisoformat(r["updated"]),
                    pdf_url=r["pdf_url"],
                    abs_url=r["abs_url"],
                    semantic_scholar_id=r["semantic_scholar_id"],
                    citation_count=r["citation_count"],
                    influential_citation_count=r["influential_citation_count"],
                    tldr=r["tldr"],
                    relevance_score=r["relevance_score"],
                    relevance_rationale=r["relevance_rationale"],
                    summary=r["summary"],
                )
                out.append(p)
        return out

    # --- Bookmarks ----------------------------------------------------------

    def toggle_bookmark(self, arxiv_id: str) -> bool:
        """Add the bookmark if missing, remove it if present. Returns the new state."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM bookmarks WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
            if row:
                conn.execute("DELETE FROM bookmarks WHERE arxiv_id = ?", (arxiv_id,))
                return False
            conn.execute(
                "INSERT INTO bookmarks (arxiv_id, created_at) VALUES (?, ?)",
                (arxiv_id, now),
            )
            return True

    def bookmarked_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT arxiv_id FROM bookmarks").fetchall()
            return {r["arxiv_id"] for r in rows}

    def load_bookmarked(self) -> list[tuple[Paper, str]]:
        """Return (paper, bookmarked_at) pairs, newest bookmark first.

        Joins bookmarks against papers so we get the full paper record back.
        Bookmarks whose paper row was somehow purged are silently skipped.
        """
        import json as _json
        from datetime import datetime as _dt

        out: list[tuple[Paper, str]] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, b.created_at AS bookmarked_at
                FROM bookmarks b
                JOIN papers p ON p.arxiv_id = b.arxiv_id
                ORDER BY b.created_at DESC
                """
            ).fetchall()
            for r in rows:
                p = Paper(
                    arxiv_id=r["arxiv_id"],
                    title=r["title"],
                    abstract=r["abstract"],
                    authors=_json.loads(r["authors"]),
                    categories=_json.loads(r["categories"]),
                    published=_dt.fromisoformat(r["published"]),
                    updated=_dt.fromisoformat(r["updated"]),
                    pdf_url=r["pdf_url"],
                    abs_url=r["abs_url"],
                    semantic_scholar_id=r["semantic_scholar_id"],
                    citation_count=r["citation_count"],
                    influential_citation_count=r["influential_citation_count"],
                    tldr=r["tldr"],
                    relevance_score=r["relevance_score"],
                    relevance_rationale=r["relevance_rationale"],
                    summary=r["summary"],
                )
                out.append((p, r["bookmarked_at"]))
        return out

    def previously_surfaced_titles(self, limit: int = 50) -> list[str]:
        """Recent high-scoring titles, used as 'connections' context for the summarizer."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT title FROM papers
                WHERE relevance_score IS NOT NULL AND relevance_score >= 0.6
                ORDER BY first_seen DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [r["title"] for r in rows]
