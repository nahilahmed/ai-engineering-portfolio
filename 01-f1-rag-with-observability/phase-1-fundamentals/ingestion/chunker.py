"""
Chunker for Phase 1 — F1 Strategy RAG.

Responsibility: split a Document into overlapping word-window Chunks
that fit within the embedding model's 256-token limit.

Sizing rationale:
  all-MiniLM-L6-v2 hard limit = 256 tokens
  English prose average       ≈ 1.3 tokens/word
  Safe word budget            = 256 / 1.3 ≈ 197 words → use 180 for headroom
  Overlap                     = 20 words (keeps context across boundaries)
"""

from __future__ import annotations

import hashlib

import structlog
from pydantic import BaseModel, ConfigDict

from ingestion.document_loader import Document

log = structlog.get_logger()

CHUNK_SIZE = 180   # words per chunk
CHUNK_OVERLAP = 20  # words of overlap between consecutive chunks


class Chunk(BaseModel):
    """A single embeddable unit produced by the chunker.

    chunk_id — deterministic ID for ChromaDB (hash of source + index)
    text     — chunk content, guaranteed to fit within 256 tokens
    metadata — parent document metadata + chunk_index for traceability
    """

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str
    metadata: dict[str, str]


class Chunker:
    """Splits Documents into overlapping word-window Chunks.

    Stateless — no __init__ needed. All config lives at module level
    so it's visible in one place and easy to tune.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a Document into Chunks and return them all."""
        words = document.text.split()
        step = CHUNK_SIZE - CHUNK_OVERLAP
        chunks: list[Chunk] = []

        for i in range(0, len(words), step):
            window = words[i : i + CHUNK_SIZE]
            if not window:
                break
            chunk_index = len(chunks)
            chunk_id = self._make_id(document.metadata["source"], chunk_index)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=" ".join(window),
                    metadata={
                        **document.metadata,
                        "chunk_index": str(chunk_index),
                    },
                )
            )

        log.info(
            "chunker.done",
            source=document.metadata.get("source"),
            total_words=len(words),
            num_chunks=len(chunks),
        )
        return chunks

    def _make_id(self, source: str, index: int) -> str:
        """Deterministic 12-char ID from source URL/path + chunk index."""
        raw = f"{source}:{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
