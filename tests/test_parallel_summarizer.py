"""Tests for the parallel summarizer."""
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.models import Paper
from src.synthesis.summarizer import Summarizer


def _make(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="Abstract.",
        authors=["Author"],
        categories=["cs.CL"],
        published=datetime(2026, 6, 1, 12, 0),
        updated=datetime(2026, 6, 1, 12, 0),
        pdf_url="",
        abs_url="",
    )


@pytest.fixture
def summarizer(monkeypatch):
    monkeypatch.setenv("DO_INFERENCE_API_KEY", "test-key")
    return Summarizer(concurrency=5)


def test_returns_summaries_in_input_order(summarizer):
    papers = [_make(f"id{i}") for i in range(6)]

    async def fake_chat(system, user, **kwargs):
        # Echo arxiv_id from the user prompt back as the plain_language
        for line in user.splitlines():
            if line.startswith("Title:"):
                aid = line.split(" ")[2]
                return {
                    "plain_language": f"summary-{aid}",
                    "key_contributions": ["c1"],
                    "methodology_notes": "m",
                    "connections": "n",
                }
        return {"plain_language": "?", "key_contributions": [], "methodology_notes": "", "connections": ""}

    summarizer.llm.chat_json_async = AsyncMock(side_effect=fake_chat)
    summaries = summarizer.summarize_many(papers, prior_titles=[])

    assert len(summaries) == 6
    # Must be in input order even though they may have completed concurrently
    for i, s in enumerate(summaries):
        assert s.paper.arxiv_id == f"id{i}"
        assert s.plain_language == f"summary-id{i}"


def test_on_each_complete_fires_for_every_paper(summarizer):
    papers = [_make(f"id{i}") for i in range(5)]

    async def fake_chat(system, user, **kwargs):
        return {"plain_language": "x", "key_contributions": [], "methodology_notes": "", "connections": ""}

    summarizer.llm.chat_json_async = AsyncMock(side_effect=fake_chat)

    calls: list = []
    summarizer.summarize_many(
        papers,
        prior_titles=[],
        on_each_complete=lambda p, s, e: calls.append((p.arxiv_id, s is not None, e)),
    )
    assert len(calls) == 5
    assert all(c[1] is True for c in calls)  # all succeeded
    assert all(c[2] is None for c in calls)  # no errors


def test_one_failure_does_not_kill_the_rest(summarizer):
    papers = [_make(f"id{i}") for i in range(4)]
    call = [0]

    async def fake_chat(system, user, **kwargs):
        call[0] += 1
        if call[0] == 2:
            raise RuntimeError("boom")
        return {"plain_language": "ok", "key_contributions": [], "methodology_notes": "", "connections": ""}

    summarizer.llm.chat_json_async = AsyncMock(side_effect=fake_chat)
    results: list = []
    out = summarizer.summarize_many(
        papers,
        prior_titles=[],
        on_each_complete=lambda p, s, e: results.append((p.arxiv_id, s is not None, e)),
    )
    # 3 successes, 1 failure
    assert len(out) == 3
    assert sum(1 for r in results if r[1]) == 3
    assert sum(1 for r in results if r[2] is not None) == 1


def test_concurrency_actually_parallelizes(summarizer):
    """With concurrency=5 and 5 papers of 200ms latency,
    sequential = 1.0s, parallel = ~0.2s."""
    papers = [_make(f"id{i}") for i in range(5)]

    async def slow_chat(system, user, **kwargs):
        await asyncio.sleep(0.2)
        return {"plain_language": "x", "key_contributions": [], "methodology_notes": "", "connections": ""}

    summarizer.llm.chat_json_async = AsyncMock(side_effect=slow_chat)

    t0 = time.time()
    summarizer.summarize_many(papers, prior_titles=[])
    elapsed = time.time() - t0
    assert elapsed < 0.5, f"expected concurrent run < 0.5s, got {elapsed:.2f}s"


def test_empty_input_returns_empty(summarizer):
    summarizer.llm.chat_json_async = AsyncMock(side_effect=AssertionError("should not be called"))
    assert summarizer.summarize_many([], prior_titles=[]) == []
