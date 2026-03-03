"""
Hybrid Retriever for Phase 2 — F1 Strategy RAG.

Responsibility: combine BM25 keyword ranking and ChromaDB semantic ranking
into a single merged ranking using Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Semantic search (vector) finds conceptually similar chunks even when exact
    words don't match. Misses: specific terms, acronyms, proper nouns.
  - BM25 (keyword) finds chunks with exact term overlap. Misses: paraphrasing,
    synonyms, conceptual questions.
  - RRF merges both ranked lists by position, not raw score. A chunk that ranks
    highly in both lists beats one that dominates only one.

RRF formula:
  score(chunk) = Σ 1 / (k + rank_in_list)  for each ranked list
  k=60 is standard — dampens the rank-1 advantage without flattening the curve.
"""

from __future__ import annotations

import structlog

from retrieval.bm25_retriever import BM25Retriever
from retrieval.retriever import Retriever

log = structlog.get_logger()

RRF_K = 60  # standard constant — higher k = flatter curve, less weight on top ranks


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    top_k: int,
) -> list[dict]:
    """Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Args:
        ranked_lists: each inner list is a ranked sequence of chunk dicts,
                      ordered best-first. Each dict must have a 'chunk_id' key.
        top_k:        number of results to return

    Returns:
        Merged list of chunk dicts, ordered by descending RRF score.
        Each dict retains original keys and gains an 'rrf_score' field.
    """
    rrf_scores: dict[str, float] = {}
    # Keep one copy of each chunk's data for the final output
    chunk_by_id: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            chunk_by_id[cid] = chunk  # last write wins — data is identical across lists

    merged = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    return [
        {**chunk_by_id[cid], "rrf_score": round(score, 6)}
        for cid, score in merged
    ]


class HybridRetriever:
    """Merges semantic and keyword retrieval via Reciprocal Rank Fusion.

    Fetches 2 * top_k candidates from each retriever to give RRF enough
    material — a chunk ranked 6th by BM25 and 7th by vector might be the
    best combined result, but would be lost if we only fetched top_k from each.

    Dependencies are injected — Retriever and BM25Retriever are shared
    singletons from pipeline startup.
    """

    def __init__(self, retriever: Retriever, bm25_retriever: BM25Retriever) -> None:
        self._retriever = retriever
        self._bm25 = bm25_retriever
        log.info("hybrid_retriever.ready")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve and merge results from semantic and keyword search.

        Args:
            query:  raw user question
            top_k:  number of chunks to return after fusion

        Returns:
            List of chunk dicts ordered by descending RRF score.
            Keys: chunk_id, text, metadata, rrf_score.
        """
        candidates = top_k * 2  # fetch wider so fusion has enough to work with

        semantic_results = self._retriever.retrieve(query, top_k=candidates)
        bm25_results = self._bm25.retrieve(query, top_k=candidates)

        merged = _reciprocal_rank_fusion(
            ranked_lists=[semantic_results, bm25_results],
            top_k=top_k,
        )

        log.info(
            "hybrid_retriever.done",
            query=query,
            semantic_candidates=len(semantic_results),
            bm25_candidates=len(bm25_results),
            merged_results=len(merged),
            top_rrf_score=merged[0]["rrf_score"] if merged else 0,
        )
        return merged
