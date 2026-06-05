"""Benchmark how concurrency affects wall-clock time and reliability.

Tests the two pipeline stages with tunable concurrency:
  1. Scoring  (50 papers, batched 10 at a time → 5 batches; concurrency = 3,5,8,10,15,20)
  2. Summarization (10 papers, 1 LLM call each; concurrency = 1,3,5,8,10)

Reports wall-clock time, success rate, and any failures so you can pick the
sweet spot for each stage.

The other two pipeline stages aren't included because:
  - Enrichment (Semantic Scholar): per-paper sequential, capped by S2 rate
    limits. Higher concurrency just triggers the rate limiter faster.
  - Trend radar: single LLM call across the shortlist. No concurrency knob.

Usage:
    python scripts/benchmark_concurrency.py
"""
from __future__ import annotations

import copy
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.ingestion.storage import PaperStore
from src.llm import LLMClient
from src.scoring.relevance import RelevanceScorer
from src.synthesis.summarizer import Summarizer

ROOT = Path(__file__).resolve().parent.parent

SCORING_SAMPLE = 241  # Use the full DB so num_batches > 20
SUMMARIZE_SAMPLE = 10

# Order: highest first. With SCORING_SAMPLE=241 and batch_size=10 (= 25 batches),
# concurrency > 25 is the same as concurrency=25 (can't dispatch more batches
# than exist). We test up to 30 anyway to confirm DO doesn't penalize the burst.
SCORING_CONCURRENCIES = [30, 25, 20]
SUMMARIZE_CONCURRENCIES = [10, 8, 5, 3, 1]

PAUSE_BETWEEN_RUNS = 3  # seconds — lets DO rate-limit windows reset


def benchmark_scoring(profile: dict, sample, llm: LLMClient) -> int | None:
    """Walk concurrency high → low. Stop at the first level with zero failures.
    Returns the highest safe concurrency, or None if all levels failed.
    """
    print(f"=== SCORING ({len(sample)} papers, batch_size=10) ===")
    print(f"{'conc':>6} {'time':>8} {'failed':>8} {'success':>9}")
    print("-" * 40)

    highest_clean = None
    for c in SCORING_CONCURRENCIES:
        fresh = [copy.deepcopy(p) for p in sample]
        for p in fresh:
            p.relevance_score = None
            p.relevance_rationale = None

        scorer = RelevanceScorer(profile=profile, llm=llm, concurrency=c)
        t0 = time.time()
        scorer.score(fresh)
        elapsed = time.time() - t0

        failed = sum(1 for p in fresh if (p.relevance_rationale or "").startswith("scoring failed"))
        success = len(sample) - failed
        marker = "  ← clean" if failed == 0 else ""
        print(f"{c:>6} {elapsed:>7.1f}s {failed:>8d} {success:>9d}{marker}")

        if failed == 0:
            highest_clean = c
            print(f"\nHighest safe scoring concurrency: {c} (stopping; lower levels would also be clean)\n")
            return c
        time.sleep(PAUSE_BETWEEN_RUNS)
    print()
    return highest_clean


def benchmark_summarization(papers, llm: LLMClient) -> int | None:
    print(f"=== SUMMARIZATION ({len(papers)} papers, 1 LLM call each) ===")
    print(f"{'conc':>6} {'time':>8} {'failed':>8} {'success':>9}")
    print("-" * 40)

    for c in SUMMARIZE_CONCURRENCIES:
        summarizer = Summarizer(llm=llm, concurrency=c)
        failed_count = [0]

        def _on_complete(p, s, e):
            if e is not None or s is None:
                failed_count[0] += 1

        t0 = time.time()
        summaries = summarizer.summarize_many(
            list(papers),
            prior_titles=[],
            on_each_complete=_on_complete,
        )
        elapsed = time.time() - t0

        success = len(summaries)
        failed = failed_count[0]
        marker = "  ← clean" if failed == 0 else ""
        print(f"{c:>6} {elapsed:>7.1f}s {failed:>8d} {success:>9d}{marker}")

        if failed == 0:
            print(f"\nHighest safe summarize concurrency: {c} (stopping; lower levels would also be clean)\n")
            return c
        time.sleep(PAUSE_BETWEEN_RUNS)
    print()
    return None


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    profile = json.loads((ROOT / "config" / "user_profile.example.json").read_text())
    store = PaperStore(ROOT / "data" / "papers.db")
    papers_all = store.load_all()
    if len(papers_all) < SCORING_SAMPLE:
        print(f"Need ≥ {SCORING_SAMPLE} papers in the DB. Run the pipeline first.")
        return 1

    llm = LLMClient()
    print(f"Model: {llm.model}")
    print(f"Pause between runs: {PAUSE_BETWEEN_RUNS}s (lets rate-limit windows reset)")
    print()

    random.seed(42)
    # Cap scoring sample to whatever's actually in the DB
    scoring_sample = random.sample(papers_all, min(SCORING_SAMPLE, len(papers_all)))
    summarize_sample = random.sample(papers_all, SUMMARIZE_SAMPLE)

    print(f"Scoring sample size: {len(scoring_sample)} papers")
    print(f"Number of batches at batch_size=10: {(len(scoring_sample) + 9) // 10}")
    print()

    benchmark_scoring(profile, scoring_sample, llm)
    # Summarization already benchmarked — skip if you only care about scoring
    # benchmark_summarization(summarize_sample, llm)

    print("Notes:")
    print("- 'failed' counts LLM calls that errored entirely (rate limit, 5xx, JSON parse).")
    print("- For scoring, failure means 10 papers default to score=0 — quality hit.")
    print("- For summarization, failure means that paper is dropped from the digest.")
    print("- The floor on either stage is ~1 batch latency (the first call to come back).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
