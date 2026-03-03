"""
Retriever for Phase 1 — F1 Strategy RAG.

Responsibility: accept a raw text query, embed it, and return the top-k
most semantically similar chunks from the vector store.

This is the glue layer between user input and ChromaDB — the Embedder and
VectorStore are injected so they can be shared across the pipeline without
being re-instantiated per request.
"""

from __future__ import annotations

import structlog

from embeddings.embedder import Embedder
from ingestion.chunker import Chunk
from retrieval.vector_store import VectorStore

log = structlog.get_logger()


class Retriever:
    """Converts a text query into a vector and fetches top-k matching chunks.

    Dependencies are injected at init — never instantiated internally.
    This keeps the Embedder and VectorStore as singletons in the pipeline.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Embed a query string and return the top-k most similar chunks.

        Args:
            query:  raw user question or search string
            top_k:  number of chunks to return
            where:  optional metadata filter, e.g. {"year": "2024"}

        Returns:
            List of dicts with keys: chunk_id, text, metadata, distance
        """
        log.info("retriever.query", query=query, top_k=top_k)

        # Wrap query text in a minimal Chunk so Embedder.embed() accepts it.
        # chunk_id and metadata are placeholders — only text is used for embedding.
        query_chunk = Chunk(
            chunk_id="query",
            text=query,
            metadata={"source": "query"},
        )

        [embedded_query] = self._embedder.embed([query_chunk])
        results = self._vector_store.query(
            embedding=embedded_query.embedding,
            top_k=top_k,
            where=where,
        )

        log.info("retriever.done", num_results=len(results))
        return results
