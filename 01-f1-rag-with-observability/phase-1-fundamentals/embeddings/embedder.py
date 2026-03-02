"""
Embedder for Phase 1 — F1 Strategy RAG.

Responsibility: convert Chunk objects into EmbeddedChunk objects by running
each chunk's text through a local sentence-transformers model.

Model: all-MiniLM-L6-v2
  - 384-dimensional output
  - 256-token hard limit (chunks are sized to 180 words to stay safe)
  - Runs on M1 CPU, ~90MB RAM for the model weights
"""

from __future__ import annotations

import structlog
from pydantic import ConfigDict
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk

log = structlog.get_logger()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


class EmbeddedChunk(Chunk):
    """A Chunk with its embedding vector attached.

    Inherits chunk_id, text, and metadata from Chunk.
    Adds embedding — a 384-float dense vector from all-MiniLM-L6-v2.
    """

    model_config = ConfigDict(extra="ignore")

    embedding: list[float]


class Embedder:
    """Converts Chunks to EmbeddedChunks using a local sentence-transformers model.

    The model is loaded once at instantiation and reused across all calls.
    Never reload per request — model loading takes ~1-2s and wastes RAM.
    """

    def __init__(self) -> None:
        log.info("embedder.loading", model=MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        log.info("embedder.ready", model=MODEL_NAME)

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of Chunks and return EmbeddedChunks.

        All chunk texts are encoded in a single batched call — sentence-transformers
        processes BATCH_SIZE chunks at a time internally, which is faster than
        calling encode() once per chunk.
        """
        if not chunks:
            log.warning("embedder.empty_input")
            return []

        texts = [chunk.text for chunk in chunks]

        log.info("embedder.encoding", num_chunks=len(texts), batch_size=BATCH_SIZE)
        vectors = self._model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embedded = [
            EmbeddedChunk(
                **chunk.model_dump(),
                embedding=vector.tolist(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        log.info("embedder.done", num_embedded=len(embedded))
        return embedded
