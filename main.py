"""CLI entrypoint for the arXiv digest pipeline.

Usage:
  python main.py --profile config/user_profile.example.json
  python main.py --profile config/me.json --lookback 2 --threshold 0.7 --format markdown
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.pipeline import Digest, run_pipeline


def render_markdown(digest: Digest) -> str:
    if not digest.shortlist and not digest.trends:
        return f"# arXiv Digest\n\nNo relevant new papers found. ({digest.new_papers} new total)\n"

    lines = ["# arXiv Daily Digest", ""]
    lines.append(
        f"_{digest.new_papers} new papers ingested, "
        f"{len(digest.shortlist)} shortlisted, {digest.skipped} skipped._"
    )
    lines.append("")

    if digest.trends:
        lines.append("## Trend Radar")
        for t in digest.trends:
            lines.append(f"### {t.theme}")
            lines.append(t.why_it_matters)
            lines.append("")
            lines.append("Papers: " + ", ".join(f"`{pid}`" for pid in t.paper_ids))
            lines.append("")

    lines.append("## Shortlist")
    for s in digest.shortlist:
        p = s.paper
        score = f"{p.relevance_score:.2f}" if p.relevance_score is not None else "?"
        lines.append(f"### [{p.title}]({p.abs_url})")
        authors = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors += f" + {len(p.authors) - 3} more"
        lines.append(f"_{authors}_ · `{p.arxiv_id}` · relevance **{score}**")
        if p.citation_count is not None:
            lines.append(f"_Citations: {p.citation_count}_")
        lines.append("")
        lines.append(f"**Why it matters for you:** {p.relevance_rationale}")
        lines.append("")
        lines.append(s.plain_language)
        lines.append("")
        lines.append("**Key contributions:**")
        for c in s.key_contributions:
            lines.append(f"- {c}")
        lines.append("")
        lines.append(f"**Methodology:** {s.methodology_notes}")
        lines.append("")
        lines.append(f"**Connections:** {s.connections}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the arXiv daily digest pipeline.")
    parser.add_argument("--profile", type=Path, required=True, help="Path to user profile JSON")
    parser.add_argument("--db", type=Path, default=Path("data/papers.db"), help="SQLite DB path")
    parser.add_argument("--lookback", type=int, default=1, help="Days of arXiv submissions to fetch")
    parser.add_argument("--max-per-cat", type=int, default=100, help="Max papers per category")
    parser.add_argument("--threshold", type=float, default=0.6, help="Min relevance score to shortlist")
    parser.add_argument("--shortlist-max", type=int, default=10, help="Max papers in the shortlist")
    parser.add_argument("--no-enrichment", action="store_true", help="Skip Semantic Scholar enrichment")
    parser.add_argument("--s2-api-key", type=str, default=None, help="Semantic Scholar API key (optional)")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--out", type=Path, default=None, help="Write output to a file instead of stdout")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    digest = run_pipeline(
        profile_path=args.profile,
        db_path=args.db,
        lookback_days=args.lookback,
        max_results_per_category=args.max_per_cat,
        shortlist_threshold=args.threshold,
        shortlist_max=args.shortlist_max,
        semantic_scholar_api_key=args.s2_api_key,
        enable_enrichment=not args.no_enrichment,
    )

    if args.format == "json":
        output = json.dumps(digest.to_dict(), indent=2)
    else:
        output = render_markdown(digest)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
