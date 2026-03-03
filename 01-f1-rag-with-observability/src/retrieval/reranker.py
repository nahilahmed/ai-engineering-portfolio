"""
Reranker for Phase 2 — F1 Strategy RAG.

Responsibility: take a small candidate set from the hybrid retriever and
re-score each chunk against the query using a cross-encoder model.

Bi-encoder vs cross-encoder:
  Bi-encoder (used in Embedder): query and chunk are encoded SEPARATELY into
  vectors, then compared with cosine similarity. Fast — embeddings are
  precomputed. But the model never sees query and chunk together, so it
  misses subtle relevance signals.

  Cross-encoder (used here): [query + chunk] is passed as a SINGLE input.
  Full transformer attention runs across both at once — every query token
  can attend to every chunk token. More accurate, but must run inference
  per (query, chunk) pair. Only feasible on the small candidate set after
  retrieval, not on the full corpus.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 6-layer MiniLM, trained on MS MARCO passage ranking
  - Outputs raw logits: positive = relevant, negative = not relevant
  - Fast enough for 5-10 pairs on M1 CPU (~50ms total)
  - Load once at startup — ~90MB, reused across all requests
"""

from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

log = structlog.get_logger()

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Re-scores a small candidate set using a cross-encoder model.

    Loaded once at init — CrossEncoder takes ~2s and ~90MB on first load.
    Subsequent rerank() calls are fast: inference on 5-10 pairs is ~50ms.

    Input: chunks from HybridRetriever (or any retriever) as list[dict]
    Output: same dicts re-ordered by cross-encoder score, top_k returned
    """

    def __init__(self) -> None:
        log.info("reranker.loading", model=MODEL_NAME)
        self._model = CrossEncoder(MODEL_NAME)
        log.info("reranker.ready", model=MODEL_NAME)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """Re-score and re-order chunks by cross-encoder relevance.

        Args:
            query:  the original user question
            chunks: candidate chunks from upstream retrieval (hybrid or vector)
            top_k:  how many to return after reranking

        Returns:
            Top-k chunks ordered by descending cross-encoder score.
            Each dict retains original keys and gains a 'rerank_score' field.
        """
        if not chunks:
            return []

        # Build (query, chunk_text) pairs — cross-encoder scores each pair
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self._model.predict(pairs)  # numpy array, one score per pair

        # Attach scores and sort descending — higher logit = more relevant
        ranked = sorted(
            zip(chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        results = [
            {**chunk, "rerank_score": round(float(score), 4)}
            for chunk, score in ranked
        ]

        log.info(
            "reranker.done",
            query=query,
            input_chunks=len(chunks),
            output_chunks=len(results),
            top_score=results[0]["rerank_score"] if results else None,
            bottom_score=results[-1]["rerank_score"] if results else None,
        )
        return results
