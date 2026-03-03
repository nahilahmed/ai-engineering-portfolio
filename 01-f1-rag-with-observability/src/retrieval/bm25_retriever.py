"""
BM25 Retriever for Phase 2 — F1 Strategy RAG.

Responsibility: rank chunks by keyword relevance using the BM25Okapi algorithm.
Complements ChromaDB's semantic search — catches exact term matches that vector
similarity misses (driver names, race names, acronyms like VSC/DRS/DNF).

BM25 (Best Matching 25) scores each document by:
  1. Term frequency      — how often the query term appears in the document
  2. Inverse doc freq    — rare terms across the corpus score higher
  3. Length normalisation — short documents aren't penalised vs long ones

The index is built once at startup from all chunks in the vector store.
Each retrieve() call is a pure in-memory operation — no API calls, no disk I/O.
"""

from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi

from retrieval.vector_store import VectorStore

log = structlog.get_logger()


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace.

    Simple but sufficient for F1 text — domain terms like 'verstappen',
    'undercut', 'vsc', 'drs' are already distinctive without stemming.
    """
    return text.lower().split()


class BM25Retriever:
    """Keyword retriever backed by BM25Okapi.

    Loads the full corpus from ChromaDB once at init and builds an in-memory
    BM25 index. Subsequent retrieve() calls are fast — no DB reads.

    Returns the same dict shape as Retriever (chunk_id, text, metadata) plus
    a 'score' field (higher = more relevant, opposite convention to 'distance').
    The HybridRetriever uses rank position, not raw score values, so the
    different conventions don't matter there.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        chunks = vector_store.get_all()

        if not chunks:
            log.warning("bm25_retriever.empty_corpus")

        # Preserve chunk order — BM25 scores are index-aligned to this list
        self._chunks = chunks
        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self._index = BM25Okapi(tokenized_corpus)

        log.info("bm25_retriever.ready", corpus_size=len(chunks))

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Return the top-k chunks ranked by BM25 keyword relevance.

        Args:
            query:  raw user question
            top_k:  number of chunks to return

        Returns:
            List of dicts with keys: chunk_id, text, metadata, score
            Ordered by descending BM25 score (highest relevance first).
        """
        if not self._chunks:
            return []

        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)  # numpy array, one score per chunk

        # Pair each chunk with its score, sort descending, take top_k
        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        results = [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": round(float(score), 4),
            }
            for chunk, score in ranked
        ]

        log.info(
            "bm25_retriever.done",
            query=query,
            top_k=top_k,
            top_score=results[0]["score"] if results else 0,
        )
        return results
