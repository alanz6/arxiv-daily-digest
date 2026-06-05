"""Tests for the async-concurrent batched relevance scoring.

Mocks the LLM client's async method — no real network calls.
"""
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.models import Paper
from src.scoring.relevance import RelevanceScorer


def _make(arxiv_id: str, title: str = "T", abstract: str = "A") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=[],
        categories=["cs.CL"],
        published=datetime(2026, 6, 1, 12, 0),
        updated=datetime(2026, 6, 1, 12, 0),
        pdf_url="",
        abs_url="",
    )


PROFILE = {
    "name": "Test",
    "research_interests": ["agents"],
    "methodological_preferences": [],
    "avoid": [],
}


def _mock_response(papers):
    """Build the JSON shape the scorer expects."""
    return {
        "scores": [
            {"arxiv_id": p.arxiv_id, "score": 0.5, "rationale": f"r-{p.arxiv_id}"}
            for p in papers
        ]
    }


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.setenv("DO_INFERENCE_API_KEY", "test-key")
    return RelevanceScorer(profile=PROFILE, batch_size=10, concurrency=3)


def test_score_assigns_results_to_correct_papers(scorer):
    papers = [_make(f"id{i}") for i in range(25)]

    async def fake_chat(system, user, schema_hint=None, max_tokens=4096, temperature=0.2):
        # Recover the batch from the user prompt
        ids = [line.split(": ", 1)[1] for line in user.splitlines() if line.startswith("arxiv_id: ")]
        return {
            "scores": [
                {"arxiv_id": i, "score": 0.5, "rationale": f"r-{i}"} for i in ids
            ]
        }

    scorer.llm.chat_json_async = AsyncMock(side_effect=fake_chat)
    scorer.score(papers)

    assert all(p.relevance_score == 0.5 for p in papers)
    assert papers[0].relevance_rationale == "r-id0"
    assert papers[24].relevance_rationale == "r-id24"


def test_progress_callback_fires_with_monotonic_counts(scorer):
    papers = [_make(f"id{i}") for i in range(25)]

    async def fake_chat(system, user, **kwargs):
        ids = [line.split(": ", 1)[1] for line in user.splitlines() if line.startswith("arxiv_id: ")]
        return {"scores": [{"arxiv_id": i, "score": 0.4, "rationale": ""} for i in ids]}

    scorer.llm.chat_json_async = AsyncMock(side_effect=fake_chat)

    progress_calls: list[tuple[int, int]] = []
    scorer.score(papers, on_progress=lambda done, total: progress_calls.append((done, total)))

    # 3 batches of 10/10/5
    assert len(progress_calls) == 3
    dones = [c[0] for c in progress_calls]
    # Monotonic
    assert dones == sorted(dones)
    # Last call hits the total
    assert progress_calls[-1] == (25, 25)
    # Totals are correct
    assert all(c[1] == 25 for c in progress_calls)


def test_failed_batch_falls_back_to_zero_without_killing_others(scorer):
    papers = [_make(f"id{i}") for i in range(20)]
    call_count = [0]

    async def fake_chat(system, user, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated DO outage")
        ids = [line.split(": ", 1)[1] for line in user.splitlines() if line.startswith("arxiv_id: ")]
        return {"scores": [{"arxiv_id": i, "score": 0.7, "rationale": "ok"} for i in ids]}

    scorer.llm.chat_json_async = AsyncMock(side_effect=fake_chat)
    scorer.score(papers)

    # All papers got a score (0 for the failed batch, 0.7 for the rest)
    assert all(p.relevance_score is not None for p in papers)
    zeros = [p for p in papers if p.relevance_score == 0.0]
    sevens = [p for p in papers if p.relevance_score == 0.7]
    assert len(zeros) == 10  # one batch failed
    assert len(sevens) == 10
    # Failed batch carries the error message
    assert "simulated DO outage" in zeros[0].relevance_rationale


def test_concurrency_actually_parallelizes_batches(scorer):
    """With concurrency=3 and 6 batches of fake 200ms latency,
    sequential = 1.2s, parallel-3 = ~0.4s."""
    scorer.concurrency = 3
    papers = [_make(f"id{i}") for i in range(60)]

    async def slow_chat(system, user, **kwargs):
        await asyncio.sleep(0.2)
        ids = [line.split(": ", 1)[1] for line in user.splitlines() if line.startswith("arxiv_id: ")]
        return {"scores": [{"arxiv_id": i, "score": 0.5, "rationale": ""} for i in ids]}

    scorer.llm.chat_json_async = AsyncMock(side_effect=slow_chat)

    t0 = time.time()
    scorer.score(papers)
    elapsed = time.time() - t0

    # 6 batches × 200ms / 3 = 400ms sequential portion + overhead.
    # Sequential would be 1.2s. Assert we beat 0.8s (generous margin).
    assert elapsed < 0.8, f"expected concurrent run < 0.8s, got {elapsed:.2f}s"


def test_concurrency_one_runs_sequentially(scorer):
    """With concurrency=1, 6 batches of 100ms ≈ 0.6s."""
    scorer.concurrency = 1
    papers = [_make(f"id{i}") for i in range(60)]

    async def slow_chat(system, user, **kwargs):
        await asyncio.sleep(0.1)
        ids = [line.split(": ", 1)[1] for line in user.splitlines() if line.startswith("arxiv_id: ")]
        return {"scores": [{"arxiv_id": i, "score": 0.5, "rationale": ""} for i in ids]}

    scorer.llm.chat_json_async = AsyncMock(side_effect=slow_chat)

    t0 = time.time()
    scorer.score(papers)
    elapsed = time.time() - t0

    # Expected ≥ 0.5s (6 × 0.1 ≈ 0.6, allow some scheduling slack)
    assert elapsed >= 0.5, f"expected sequential run ≥ 0.5s, got {elapsed:.2f}s"
