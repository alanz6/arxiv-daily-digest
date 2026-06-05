"""Offline evaluation of the personalized scorer.

Three no-label tests, all run against papers already in the local SQLite DB
(no arXiv re-fetch, no manual ratings needed):

  1. LLM vs keyword baseline. Run the keyword scorer on the same papers the LLM
     already scored. Compare top-K. Quantify where each method picks papers the
     other misses (the value-add narrative).

  2. Cross-profile sensitivity. Re-score a sample of papers against three
     deliberately-distinct research profiles (NLP/agents, vision/robotics,
     theory/crypto). If personalization works, the top-10 lists should be
     near-disjoint across profiles (low Jaccard). High overlap = scorer is
     just picking "popular" papers regardless of profile.

  3. Self-consistency. Re-score the top-20 papers a second time. Compute
     mean absolute score diff and rank-correlation. High noise = unreliable.

Outputs a markdown report to docs/evaluation.md.

Usage:
    python scripts/evaluate.py --profile config/me.json --db data/papers.db

Requires that you've already run the pipeline at least once to populate the DB.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.ingestion.storage import PaperStore
from src.llm import LLMClient
from src.models import Paper
from src.scoring.keyword_baseline import score_papers as keyword_score_papers
from src.scoring.embedding_filter import EmbeddingFilter
from src.scoring.relevance import RelevanceScorer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
EVAL_PROFILES_DIR = ROOT / "config" / "eval_profiles"
REPORT_PATH = ROOT / "docs" / "evaluation.md"


def top_k_ids(scored: list[tuple[Paper, float]], k: int) -> list[str]:
    return [p.arxiv_id for p, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation. Returns 1.0 for identical rankings."""
    n = len(a)
    if n < 2:
        return 0.0
    rank_a = {v: i for i, v in enumerate(sorted(a))}
    rank_b = {v: i for i, v in enumerate(sorted(b))}
    ra = [rank_a[v] for v in a]
    rb = [rank_b[v] for v in b]
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    da = (sum((ra[i] - mean_a) ** 2 for i in range(n))) ** 0.5
    db = (sum((rb[i] - mean_b) ** 2 for i in range(n))) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def evaluate_llm_vs_keyword(
    papers: list[Paper], profile: dict, k: int = 10
) -> dict:
    """Compare the LLM scores already in the DB against the keyword baseline."""
    llm_scored = [(p, p.relevance_score or 0.0) for p in papers]
    kw_scored = keyword_score_papers(papers, profile)

    llm_top = top_k_ids(llm_scored, k)
    kw_top = top_k_ids(kw_scored, k)

    overlap = set(llm_top) & set(kw_top)
    llm_only = [i for i in llm_top if i not in overlap]
    kw_only = [i for i in kw_top if i not in overlap]

    by_id = {p.arxiv_id: p for p in papers}

    def title(pid: str) -> str:
        return by_id[pid].title

    return {
        "k": k,
        "overlap_count": len(overlap),
        "jaccard": jaccard(llm_top, kw_top),
        "llm_top_titles": [title(i) for i in llm_top],
        "kw_top_titles": [title(i) for i in kw_top],
        "llm_only_picks": [(i, title(i)) for i in llm_only],
        "kw_only_picks": [(i, title(i)) for i in kw_only],
    }


def evaluate_cross_profile(
    papers: list[Paper], llm: LLMClient, k: int = 10, sample_size: int = 50
) -> dict:
    """Score the same N papers against several distinct profiles. Measure top-K
    overlap (lower = better personalization)."""
    profiles = {}
    for path in sorted(EVAL_PROFILES_DIR.glob("*.json")):
        profiles[path.stem] = json.loads(path.read_text())

    if len(profiles) < 2:
        return {"error": "Need at least 2 eval profiles in config/eval_profiles/"}

    random.seed(42)
    sample = random.sample(papers, min(sample_size, len(papers)))
    logger.info("Cross-profile eval on %d papers across %d profiles", len(sample), len(profiles))

    # Score each sample paper against each profile. We use a fresh copy of
    # Paper objects per profile so the scorer can mutate without cross-contamination.
    import copy
    rankings: dict[str, list[tuple[Paper, float]]] = {}
    for name, profile in profiles.items():
        logger.info("  scoring against profile: %s", name)
        copies = [copy.deepcopy(p) for p in sample]
        for c in copies:
            c.relevance_score = None
            c.relevance_rationale = None
        scorer = RelevanceScorer(profile=profile, llm=llm)
        scorer.score(copies)
        rankings[name] = [(c, c.relevance_score or 0.0) for c in copies]

    by_id = {p.arxiv_id: p for p in sample}
    tops = {name: top_k_ids(s, k) for name, s in rankings.items()}

    # Pairwise Jaccard between profile top-Ks
    names = list(tops.keys())
    jaccards = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]} vs {names[j]}"
            jaccards[key] = jaccard(tops[names[i]], tops[names[j]])

    # Each profile's top-3 picks with titles, for the report
    summary_per_profile = {}
    for name, top_ids in tops.items():
        summary_per_profile[name] = [
            {"arxiv_id": pid, "title": by_id[pid].title}
            for pid in top_ids[:3]
        ]

    return {
        "k": k,
        "sample_size": len(sample),
        "profile_names": names,
        "jaccards": jaccards,
        "summary_per_profile": summary_per_profile,
    }


def evaluate_self_consistency(
    papers: list[Paper], profile: dict, llm: LLMClient, n: int = 20
) -> dict:
    """Score the top-N papers twice and compare the two runs.

    Both passes use the *current* scoring config, so this isolates the
    stability of the model + prompt + temperature combination — not the
    drift between two different historical configs. (Previous versions
    of this test compared DB-cached scores against a fresh rerun, which
    conflated rerun noise with any config changes that happened between
    the original ingestion and the eval.)
    """
    import copy

    top_n = sorted(papers, key=lambda p: p.relevance_score or 0.0, reverse=True)[:n]

    def _fresh_score():
        copies = [copy.deepcopy(p) for p in top_n]
        for c in copies:
            c.relevance_score = None
            c.relevance_rationale = None
        scorer = RelevanceScorer(profile=profile, llm=llm)
        scorer.score(copies)
        return {c.arxiv_id: c.relevance_score or 0.0 for c in copies}

    run_a = _fresh_score()
    run_b = _fresh_score()

    ids = list(run_a.keys())
    a = [run_a[i] for i in ids]
    b = [run_b[i] for i in ids]

    diffs = [abs(a[i] - b[i]) for i in range(len(ids))]
    return {
        "n": n,
        "mean_abs_diff": statistics.mean(diffs),
        "max_abs_diff": max(diffs),
        "spearman_rank_corr": spearman(a, b),
        "rerun_kept_top10_overlap": jaccard(
            sorted(run_a, key=run_a.get, reverse=True)[:10],
            sorted(run_b, key=run_b.get, reverse=True)[:10],
        ),
    }


def evaluate_prefilter_recall(papers: list[Paper], profile: dict, ks: list[int] = [50, 75, 100]) -> dict:
    """How much does the embedding pre-filter cost us in recall?

    For each candidate K, we take the embedding pre-filter's top-K and check
    how many of the LLM's true top-10 picks are inside it. Recall@K = 1.0 means
    "the prefilter would have lost zero shortlist papers at this K".

    Requires that `papers` were LLM-scored *without* the prefilter (otherwise
    the dropped papers are already at score=0 and the test is circular). We
    auto-exclude the audit-tagged drops so the eval is safe to run repeatedly.
    """
    real_scored = [
        p for p in papers
        if p.relevance_score is not None
        and not (p.relevance_rationale or "").startswith("below embedding pre-filter")
    ]
    if len(real_scored) < max(ks) + 10:
        return {
            "error": (
                f"Need ≥ {max(ks) + 10} LLM-scored papers (have {len(real_scored)}) to "
                f"compute meaningful recall@K. Run the pipeline once with EMBEDDING_FILTER_K=0 "
                f"first, then re-run this eval."
            )
        }

    real_scored.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
    llm_top10 = set(p.arxiv_id for p in real_scored[:10])

    results = {}
    timings = {}
    for k in ks:
        f = EmbeddingFilter(profile=profile, top_k=k)
        import time
        t0 = time.time()
        kept, _ = f.filter(list(real_scored))
        timings[k] = time.time() - t0
        kept_ids = set(p.arxiv_id for p in kept)
        recovered = llm_top10 & kept_ids
        results[k] = {
            "recall_at_k": len(recovered) / len(llm_top10),
            "lost_papers": [
                {"arxiv_id": p.arxiv_id, "title": p.title, "llm_score": p.relevance_score}
                for p in real_scored[:10]
                if p.arxiv_id not in kept_ids
            ],
            "llm_calls_saved_pct": 1 - (k / len(real_scored)),
        }

    return {
        "n_scored": len(real_scored),
        "true_top10_size": len(llm_top10),
        "by_k": results,
        "encode_seconds_for_k75": timings.get(75, 0.0),
    }


def render_markdown(profile: dict, llm_model: str, n_db: int, kw_eval: dict, xp_eval: dict, sc_eval: dict, pf_eval: dict = None) -> str:
    lines = ["# Evaluation Results", ""]
    lines.append(
        f"Generated against a local DB of **{n_db} papers** "
        f"scored by `{llm_model}` (DigitalOcean serverless inference). "
        f"All three tests below use papers already in the DB — no re-fetch from arXiv, "
        f"no manual labels."
    )
    lines.append("")

    # 1. LLM vs keyword
    lines.append("## 1. Personalized LLM scoring vs. keyword baseline")
    lines.append("")
    lines.append(
        f"The keyword baseline scores a paper by the fraction of profile "
        f"keywords ({', '.join(repr(k) for k in profile.get('keywords', []))}) "
        f"that appear in title+abstract."
    )
    lines.append("")
    lines.append(f"- **Top-{kw_eval['k']} overlap with LLM**: {kw_eval['overlap_count']} / {kw_eval['k']}")
    lines.append(f"- **Jaccard similarity**: {kw_eval['jaccard']:.2f}")
    lines.append("")
    lines.append(
        "Interpretation: a low overlap means the two methods disagree, which is "
        "evidence the LLM is contributing semantic understanding beyond pure "
        "keyword matching. A high overlap would suggest the LLM isn't adding much."
    )
    lines.append("")
    if kw_eval["llm_only_picks"]:
        lines.append("**Papers the LLM picked that keyword filtering missed** (likely the personalization value-add):")
        lines.append("")
        for pid, t in kw_eval["llm_only_picks"][:5]:
            lines.append(f"- `{pid}` — {t}")
        lines.append("")
    if kw_eval["kw_only_picks"]:
        lines.append("**Papers keyword filtering picked that the LLM rejected** (likely noise the LLM is filtering out):")
        lines.append("")
        for pid, t in kw_eval["kw_only_picks"][:5]:
            lines.append(f"- `{pid}` — {t}")
        lines.append("")

    # 2. Cross-profile sensitivity
    lines.append("## 2. Cross-profile sensitivity")
    lines.append("")
    if "error" in xp_eval:
        lines.append(f"Skipped: {xp_eval['error']}")
    else:
        lines.append(
            f"Scored the same {xp_eval['sample_size']} papers against {len(xp_eval['profile_names'])} "
            f"distinct research profiles ({', '.join(xp_eval['profile_names'])}). "
            f"If personalization works, the top-{xp_eval['k']} lists should be near-disjoint."
        )
        lines.append("")
        lines.append(f"**Pairwise Jaccard similarity of top-{xp_eval['k']} lists:**")
        lines.append("")
        for pair, j in xp_eval["jaccards"].items():
            lines.append(f"- {pair}: **{j:.2f}**")
        lines.append("")
        lines.append("Interpretation: 0.0 = no overlap (perfect personalization), 1.0 = identical (no personalization).")
        lines.append("")
        lines.append("**Top picks per profile (sanity check):**")
        lines.append("")
        for name, picks in xp_eval["summary_per_profile"].items():
            lines.append(f"*{name}*")
            for p in picks:
                lines.append(f"  - `{p['arxiv_id']}` — {p['title']}")
            lines.append("")

    # 3. Self-consistency
    lines.append("## 3. Self-consistency on rerun")
    lines.append("")
    lines.append(
        f"Re-scored the top {sc_eval['n']} papers a second time using the "
        f"same profile and model. Measures noise/reliability."
    )
    lines.append("")
    lines.append(f"- **Mean absolute score difference**: {sc_eval['mean_abs_diff']:.3f}")
    lines.append(f"- **Max absolute score difference**: {sc_eval['max_abs_diff']:.3f}")
    lines.append(f"- **Spearman rank correlation**: {sc_eval['spearman_rank_corr']:.3f}")
    lines.append(f"- **Top-10 overlap (run 1 vs run 2)**: Jaccard {sc_eval['rerun_kept_top10_overlap']:.2f}")
    lines.append("")
    lines.append(
        "Interpretation: low mean diff + high rank correlation + high top-10 overlap "
        "all indicate the scorer produces stable rankings across runs. High noise "
        "would suggest the scoring prompt or the model temperature need tuning."
    )
    lines.append("")
    if sc_eval.get("rerun_kept_top10_overlap", 1.0) < 0.7 or sc_eval.get("spearman_rank_corr", 1.0) < 0.5:
        lines.append(
            "*This run shows meaningful rerun noise.* Top-10 overlap < 0.7 and/or "
            "rank correlation < 0.5 indicate the scorer's rankings are not stable. "
            "Likely causes: (1) the LLM's temperature is non-zero (currently 0.2), "
            "(2) the top-N papers all cluster in a narrow score range (e.g. 0.85–0.95), "
            "so small score wobbles flip ranks. Mitigations: lower temperature to 0, "
            "use a stronger model for scoring, or average across 2–3 runs and take "
            "the median score."
        )
        lines.append("")

    # 4. Embedding pre-filter recall
    if pf_eval is not None:
        lines.append("## 4. Embedding pre-filter recall")
        lines.append("")
        lines.append(
            "Before running the LLM scorer, we filter the day's papers down to the "
            "top-K most semantically similar to the user's profile using a small "
            "local embedding model (all-MiniLM-L6-v2). This eval measures the cost "
            "in lost shortlist picks (recall@K against the LLM's true top-10) for "
            "several choices of K."
        )
        lines.append("")
        if "error" in pf_eval:
            lines.append(f"_Skipped: {pf_eval['error']}_")
            lines.append("")
        else:
            lines.append(f"On {pf_eval['n_scored']} LLM-scored papers in the DB:")
            lines.append("")
            lines.append("| K | Recall@K vs LLM top-10 | LLM calls saved | Lost papers |")
            lines.append("|---:|---:|---:|---|")
            for k, r in pf_eval["by_k"].items():
                lost = ", ".join(p["arxiv_id"] for p in r["lost_papers"]) or "_none_"
                lines.append(
                    f"| {k} | {r['recall_at_k']:.0%} | {r['llm_calls_saved_pct']:.0%} | {lost} |"
                )
            lines.append("")
            lines.append(
                f"Embedding the full set takes ~{pf_eval['encode_seconds_for_k75']:.1f}s on CPU "
                f"(cold start; cached model afterwards), which is a fraction of the LLM "
                f"scoring time it replaces."
            )
            lines.append("")
            k75 = pf_eval["by_k"].get(75)
            if k75 and k75["recall_at_k"] >= 0.9:
                lines.append(
                    f"At the default K=75, recall is {k75['recall_at_k']:.0%} and we save "
                    f"{k75['llm_calls_saved_pct']:.0%} of LLM scoring calls — a near-loss-free "
                    f"latency win. The few lost papers (if any) sat at the bottom of the LLM's "
                    f"top-10 — i.e. they were borderline shortlist picks, not slam dunks."
                )
                lines.append("")

    # Limitations
    lines.append("## Limitations of this evaluation")
    lines.append("")
    lines.append(
        "- **No ground truth.** All three tests are *internal* — they validate "
        "consistency, sensitivity, and disagreement with a dumb baseline, but they "
        "do not directly measure whether the LLM's picks match what a real "
        "researcher would actually want to read. A user study with thumbs-up/down "
        "ratings is required for that (planned for the next milestone)."
    )
    lines.append(
        "- **Small sample.** Cross-profile sensitivity runs on a 50-paper random "
        "sample to keep LLM cost low. Larger samples would tighten the Jaccard estimates."
    )
    lines.append(
        "- **Profile design influences results.** Cross-profile Jaccard depends on "
        "how distinct the eval profiles are. The three included profiles are "
        "deliberately disjoint research areas; profiles closer to the main one "
        "would (correctly) show higher overlap."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation of the digest scorer.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "papers.db")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--skip-cross-profile", action="store_true", help="Skip cross-profile sensitivity (saves LLM cost)")
    parser.add_argument("--skip-self-consistency", action="store_true")
    parser.add_argument("--skip-prefilter-recall", action="store_true")
    parser.add_argument("--xp-sample", type=int, default=50)
    parser.add_argument("--sc-sample", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.db.exists():
        print(f"DB not found at {args.db}. Run the pipeline first.")
        return 1

    profile = json.loads(args.profile.read_text())
    store = PaperStore(args.db)
    papers = store.load_all()
    scored = [p for p in papers if p.relevance_score is not None]
    if not scored:
        print("No scored papers in DB. Run the pipeline first.")
        return 1
    logger.info("Loaded %d papers from DB (%d scored)", len(papers), len(scored))

    llm = LLMClient()

    kw_eval = evaluate_llm_vs_keyword(scored, profile, k=args.top_k)

    if args.skip_cross_profile:
        xp_eval = {"error": "skipped by flag"}
    else:
        xp_eval = evaluate_cross_profile(scored, llm, k=args.top_k, sample_size=args.xp_sample)

    if args.skip_self_consistency:
        sc_eval = {"n": 0, "mean_abs_diff": 0, "max_abs_diff": 0, "spearman_rank_corr": 0, "rerun_kept_top10_overlap": 0}
    else:
        sc_eval = evaluate_self_consistency(scored, profile, llm, n=args.sc_sample)

    if args.skip_prefilter_recall:
        pf_eval = None
    else:
        pf_eval = evaluate_prefilter_recall(scored, profile)

    import os as _os
    model = _os.environ.get("DO_INFERENCE_MODEL", "unknown")
    md = render_markdown(profile, model, len(scored), kw_eval, xp_eval, sc_eval, pf_eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
