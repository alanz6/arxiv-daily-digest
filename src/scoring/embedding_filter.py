"""Embedding-based pre-filter for the scoring stage.

Why this exists: scoring ~241 papers/day with an LLM is the slowest stage in
the pipeline. Most of those papers are not even remotely related to the user's
interests — the LLM is doing expensive work to assign them a 0.0-0.1 score
that we will throw away anyway.

This module embeds the user's profile and every fresh paper with a small
local model, then keeps only the top-K most semantically similar papers
for the LLM scorer downstream. K is generous (default 75) so the recall on
the true top-10 stays near 100% — the goal is to cut latency, not to do
the ranking itself.

The model (sentence-transformers/all-MiniLM-L6-v2) is ~80MB, downloads on
first run, and encodes 241 short texts in <10s on CPU.

Disable by setting EMBEDDING_FILTER_K=0 in the environment.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.models import Paper

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The variable that drives recall is K/N (fraction kept), not K (absolute count).
# At fixed K, recall falls as N grows because more distractor papers compete for
# the top-K slots. So we scale K with N. The sweep in docs/evaluation.md §4 on
# N=241 found K/N=0.62 gives 90% recall on the LLM's true top-10. MIN_K is a
# floor for tiny days so we don't pre-filter a 30-paper run down to ~18.
DEFAULT_RATIO = 0.62
DEFAULT_MIN_K = 50


def _compute_k_for_n(n: int) -> int:
    """Resolve K from env vars and the day's paper count N.

    Resolution order:
      1. EMBEDDING_FILTER_K (absolute count). If set, used verbatim. 0 disables.
      2. EMBEDDING_FILTER_RATIO * N, floored at EMBEDDING_FILTER_MIN_K.
    """
    explicit = os.environ.get("EMBEDDING_FILTER_K")
    if explicit is not None and explicit.strip() != "":
        try:
            return int(explicit)
        except ValueError:
            pass  # fall through to ratio path

    try:
        ratio = float(os.environ.get("EMBEDDING_FILTER_RATIO", DEFAULT_RATIO))
    except ValueError:
        ratio = DEFAULT_RATIO
    try:
        min_k = int(os.environ.get("EMBEDDING_FILTER_MIN_K", DEFAULT_MIN_K))
    except ValueError:
        min_k = DEFAULT_MIN_K

    return max(min_k, round(ratio * n))


def _build_profile_text(profile: dict) -> str:
    """Flatten the profile into a single text blob for embedding.

    We use the same fields the LLM scorer sees so the embedding similarity
    is measuring "topical match to what the LLM cares about", not just a
    keyword overlap.
    """
    parts: list[str] = []
    name = profile.get("name")
    if name:
        parts.append(f"Researcher: {name}")
    interests = profile.get("research_interests") or []
    if interests:
        parts.append("Research interests: " + "; ".join(interests))
    methods = profile.get("methodological_preferences") or []
    if methods:
        parts.append("Methodological preferences: " + "; ".join(methods))
    # Note: we intentionally do NOT include `avoid` here. Embedding similarity
    # cannot subtract — including "avoid" terms would pull matching papers
    # closer, not push them away. The LLM downstream handles the avoid list.
    return "\n".join(parts) if parts else "machine learning research"


def _paper_text(paper: Paper) -> str:
    cats = ", ".join(paper.categories) if paper.categories else ""
    return f"{paper.title}\n[{cats}]\n{paper.abstract}"


class EmbeddingFilter:
    """Pre-filter papers by embedding cosine similarity to the profile.

    Lazy-loads the sentence-transformer model on first call so that the
    cold-import path stays cheap (Flask boot, CLI --help, etc.).
    """

    def __init__(
        self,
        profile: dict,
        top_k: Optional[int] = None,
        model_name: str = DEFAULT_MODEL,
    ):
        """If `top_k` is None, K is resolved per-call as a fraction of the
        input size (see `_compute_k_for_n`). If `top_k` is an int >= 0, that
        value is used verbatim and 0 disables the filter entirely.
        """
        self.profile = profile
        self.top_k = top_k  # may be None → resolve at filter time
        self.model_name = model_name
        self._model = None  # lazy
        self._profile_text = _build_profile_text(profile)

    @property
    def enabled(self) -> bool:
        # Disabled iff the user explicitly pinned top_k=0 (or env said so).
        if self.top_k is not None:
            return self.top_k > 0
        explicit = os.environ.get("EMBEDDING_FILTER_K", "").strip()
        if explicit == "0":
            return False
        return True

    def effective_k(self, n: int) -> int:
        if self.top_k is not None:
            return self.top_k
        return _compute_k_for_n(n)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` or set EMBEDDING_FILTER_K=0 to disable."
            ) from e
        logger.info("Loading embedding model %s (first run downloads ~80MB)", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def filter(self, papers: list[Paper]) -> tuple[list[Paper], list[Paper]]:
        """Return (kept, dropped). `kept` is at most K papers, ranked.

        If embedding is disabled, returns (papers, []). If there are fewer
        than K papers, returns them all (still ranked, no drops).
        """
        if not self.enabled or not papers:
            return papers, []
        k = self.effective_k(len(papers))
        if len(papers) <= k:
            # Nothing to filter — but we still rank by similarity so the
            # downstream LLM batches see related papers grouped together,
            # which marginally helps batch coherence.
            ranked = self._rank(papers)
            return ranked, []

        ranked = self._rank(papers)
        return ranked[:k], ranked[k:]

    def _rank(self, papers: list[Paper]) -> list[Paper]:
        model = self._load_model()
        import numpy as np  # local import; numpy comes in via sentence-transformers

        profile_vec = model.encode(
            [self._profile_text], normalize_embeddings=True, show_progress_bar=False
        )
        paper_texts = [_paper_text(p) for p in papers]
        paper_vecs = model.encode(
            paper_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        # Cosine similarity == dot product on normalized vectors.
        sims = (paper_vecs @ profile_vec.T).ravel()
        order = np.argsort(-sims)
        return [papers[i] for i in order]
