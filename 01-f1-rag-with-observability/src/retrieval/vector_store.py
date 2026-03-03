"""
VectorStore for Phase 1 — F1 Strategy RAG.

Responsibility: persist EmbeddedChunks to ChromaDB and expose a query
interface for top-k semantic retrieval.

Storage layout:
  Collection : f1_chunks (single collection, metadata-filtered at query time)
  Persist dir: ../../data/chroma_db/ (relative to phase-1-fundamentals/)

ChromaDB stores per chunk:
  - embedding  : 384-float vector used for similarity search
  - document   : raw chunk text returned alongside results
  - metadata   : source, race, year, chunk_index — used for filtering
  - id         : deterministic chunk_id from MD5(source:index)
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import structlog

from embeddings.embedder import EmbeddedChunk

log = structlog.get_logger()

COLLECTION_NAME = "f1_chunks"
PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"


class VectorStore:
    """Wraps ChromaDB for storing and querying EmbeddedChunks.

    The collection is created on first use and reloaded on subsequent runs —
    ChromaDB handles this transparently via get_or_create_collection().
    """

    def __init__(self) -> None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "vector_store.ready",
            collection=COLLECTION_NAME,
            persist_dir=str(PERSIST_DIR),
            existing_chunks=self._collection.count(),
        )

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Persist EmbeddedChunks to ChromaDB.

        Upserts by chunk_id — safe to call multiple times with the same chunks,
        no duplicates will be created.
        """
        if not chunks:
            log.warning("vector_store.add.empty_input")
            return

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

        log.info("vector_store.add.done", num_chunks=len(chunks))

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Return the top-k most similar chunks to the given embedding.

        Args:
            embedding: query vector (384 floats) from the same model used at ingest
            top_k:     number of results to return
            where:     optional ChromaDB metadata filter, e.g. {"year": "2024"}

        Returns:
            List of dicts with keys: chunk_id, text, metadata, distance
            Ordered by ascending cosine distance (0 = identical, 2 = opposite).
        """
        kwargs: dict = {"query_embeddings": [embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where

        results = self._collection.query(
            **kwargs,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": round(distance, 4),
                }
            )

        log.info("vector_store.query.done", top_k=top_k, num_hits=len(hits))
        return hits

    def get_all(self) -> list[dict]:
        """Return every chunk in the collection.

        Used by BM25Retriever to build its keyword index at startup.
        Returns the same dict shape as query() — chunk_id, text, metadata —
        but without a distance field (no query was made).
        """
        results = self._collection.get(include=["documents", "metadatas"])
        return [
            {"chunk_id": cid, "text": text, "metadata": meta}
            for cid, text, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]

    def count(self) -> int:
        """Return total number of chunks currently stored."""
        return self._collection.count()
