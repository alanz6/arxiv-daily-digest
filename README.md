# ArXiv Daily Digest

A personalized academic-research briefing agent. Pulls fresh arXiv submissions, scores them against a user-defined research profile using Claude, summarizes the top picks, and surfaces emerging trends across them.

This milestone covers the **core pipeline**: ingestion → relevance scoring → summarization + trend radar. Delivery (email/Slack) and the feedback loop are not yet implemented.

## What works

- **Ingestion** — arXiv API polling by category with date filtering, Semantic Scholar enrichment (citation counts + TLDR), and SQLite-backed deduplication so papers are scored exactly once.
- **Relevance scoring** — Claude scores each new paper 0–1 against your interest profile. The profile is sent as a cached system prompt so repeated batches amortize cost.
- **Summarization** — Top-scored papers get a plain-language summary, key contributions, methodology notes, and connections to your previously surfaced papers (pulled from the local DB).
- **Trend radar** — Claude clusters the shortlist into emerging themes when it spots 2+ papers in a shared direction.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

Copy the example profile and edit it:

```bash
cp config/user_profile.example.json config/me.json
$EDITOR config/me.json
```

## Run

```bash
python main.py --profile config/me.json --verbose
```

Useful flags:

- `--lookback 2` — fetch papers from the last 2 days instead of 1
- `--threshold 0.7` — raise the relevance bar (default 0.6)
- `--shortlist-max 5` — cap the shortlist size
- `--no-enrichment` — skip Semantic Scholar (faster, no citation data)
- `--format json` — machine-readable output
- `--out digest.md` — write to a file instead of stdout
- `--model claude-sonnet-4-6` — use a cheaper model for runs at scale

## Layout

```
arxiv_digest/
├── main.py                      # CLI + markdown renderer
├── src/
│   ├── models.py                # Paper, DigestSummary, TrendCluster dataclasses
│   ├── pipeline.py              # End-to-end orchestrator
│   ├── ingestion/
│   │   ├── arxiv_client.py      # arXiv API polling
│   │   ├── semantic_scholar_client.py  # citation + TLDR enrichment
│   │   └── storage.py           # SQLite + dedup
│   ├── scoring/
│   │   └── relevance.py         # LLM relevance scoring (batched, cached profile)
│   └── synthesis/
│       ├── summarizer.py        # Per-paper digest entries
│       └── trend_radar.py       # Cross-paper theme clustering
├── config/
│   └── user_profile.example.json
└── data/
    └── papers.db                # created on first run
```

## Roadmap (next milestones)

- **Delivery** — email (SendGrid) and Slack webhook rendering. The markdown renderer in `main.py` is the bones of an email template.
- **Feedback loop** — thumbs-up/down links, web UI, and the wiring back into `profile.rating_history`. The profile schema already has a `rating_history` slot; the scorer reads it but nothing writes to it yet.
- **Scheduling** — meant to run from cron or a cloud function once the delivery layer exists.
- **Citation velocity / sleeper-paper detection** — Semantic Scholar data is captured in the DB but not yet trended over time.

## Notes on the Anthropic integration

- Uses structured outputs (`output_config.format` with a JSON schema) end-to-end, so scoring/summarization/clustering all return validated JSON. Less brittle than parsing free-form responses.
- The user profile is placed in the system prompt with `cache_control: ephemeral`. With batches of 10 papers per scoring request, the profile-prefix cache hits across batches in the same run.
- Default model is `claude-opus-4-7`. Switch to `claude-sonnet-4-6` (cheaper) or `claude-haiku-4-5` (cheapest, fine for triage) via `--model`.
