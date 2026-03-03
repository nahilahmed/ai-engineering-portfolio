"""
Answer Generator for Phase 1 — F1 Strategy RAG.

Responsibility: take a user query + pre-retrieved chunks, build a prompt,
call the Groq LLM, and return a structured answer with citations.

Design decisions:
- Accepts pre-retrieved chunks rather than owning retrieval internally.
  This keeps retrieval and generation decoupled so Phase 2's reranker
  can be inserted between them without touching this class.
- Citations are LLM-generated: the prompt labels each chunk with its ID
  and instructs the model to cite inline using [chunk_id] notation.
  Phase 2 will enforce citation presence programmatically.
- Groq model: llama-3.3-70b-versatile (free tier).
"""

from __future__ import annotations

import os
import re

import structlog
from groq import Groq
from pydantic import BaseModel

log = structlog.get_logger()

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_EXCERPT_WORDS = 30

# Stored as a module-level constant so it's easy to diff in git.
# Phase 2 will migrate this to configs/prompts.yaml with versioning.
SYSTEM_PROMPT = """\
You are an F1 strategy and technical expert.
Answer the user's question using ONLY the context chunks provided below.
For every claim you make, cite the source inline using the chunk ID in square \
brackets, e.g. [chunk_id].
If the provided context does not contain enough information to answer, say so \
explicitly — do not speculate.
Keep your answer concise and factual.\
"""


class Citation(BaseModel):
    """A single source reference grounding a claim in the answer.

    chunk_id: ties back to the ChromaDB document ID
    source:   human-readable label from chunk metadata (e.g. "fia_report_2024")
    excerpt:  first MAX_EXCERPT_WORDS words of the chunk — lets you verify
              the citation without re-fetching from the vector store
    """

    chunk_id: str
    source: str
    excerpt: str


class GeneratedAnswer(BaseModel):
    """The full output of one RAG query.

    query:     the original user question (preserved for tracing in Phase 4)
    answer:    the LLM's synthesized response, may contain [chunk_id] inline cites
    citations: structured list of sources the LLM referenced
    model:     which Groq model produced this answer
    """

    query: str
    answer: str
    citations: list[Citation]
    model: str


class AnswerGenerator:
    """Generates a grounded answer from pre-retrieved chunks via Groq.

    Dependencies are injected/loaded at init — never re-instantiated per request.
    Accepts chunks from the Retriever (or eventually the Reranker in Phase 2)
    so retrieval and generation remain decoupled.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
        self._client = Groq(api_key=api_key)
        log.info("answer_generator.ready", model=GROQ_MODEL)

    def _build_user_message(self, query: str, chunks: list[dict]) -> str:
        """Format retrieved chunks into a numbered context block for the LLM.

        Each chunk is labelled with its chunk_id so the LLM can cite it inline.
        The question follows after all chunks.
        """
        lines: list[str] = ["Context chunks:\n"]
        for chunk in chunks:
            lines.append(f"[{chunk['chunk_id']}] source={chunk['metadata'].get('source', 'unknown')}")
            lines.append(chunk["text"])
            lines.append("")  # blank line between chunks

        lines.append(f"Question: {query}")
        return "\n".join(lines)

    def _parse_citations(self, answer: str, chunks: list[dict]) -> list[Citation]:
        """Extract cited chunk IDs from the LLM response and build Citation objects.

        The LLM is instructed to write [chunk_id] inline. We regex-scan the
        answer for any [...] pattern, then look up each ID in the retrieved
        chunks. Unrecognised IDs are logged and skipped.
        """
        # Build a lookup so we can find chunk data by ID in O(1)
        chunk_by_id = {c["chunk_id"]: c for c in chunks}

        # Extract every [something] in the answer, then split on commas
        # The LLM sometimes groups multiple IDs: [id1, id2, id3]
        raw_brackets = re.findall(r"\[([^\]]+)\]", answer)
        raw_ids = [cid.strip() for bracket in raw_brackets for cid in bracket.split(",")]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids = [cid for cid in raw_ids if not (cid in seen or seen.add(cid))]  # type: ignore[func-returns-value]

        citations: list[Citation] = []
        for cid in unique_ids:
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                log.warning("answer_generator.unknown_citation", chunk_id=cid)
                continue
            excerpt = " ".join(chunk["text"].split()[:MAX_EXCERPT_WORDS])
            citations.append(
                Citation(
                    chunk_id=cid,
                    source=chunk["metadata"].get("source", "unknown"),
                    excerpt=excerpt,
                )
            )

        return citations

    def generate(self, query: str, chunks: list[dict]) -> GeneratedAnswer:
        """Generate a cited answer from a query and pre-retrieved chunks.

        Args:
            query:  the original user question
            chunks: list of dicts from Retriever.retrieve() —
                    keys: chunk_id, text, metadata, distance

        Returns:
            GeneratedAnswer with answer text, citations, and provenance fields.
        """
        if not chunks:
            log.warning("answer_generator.no_chunks", query=query)
            return GeneratedAnswer(
                query=query,
                answer="No relevant context was found to answer this question.",
                citations=[],
                model=GROQ_MODEL,
            )

        log.info("answer_generator.generating", query=query, num_chunks=len(chunks))

        user_message = self._build_user_message(query, chunks)

        response = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # low temp for factual, grounded answers
        )

        answer_text = response.choices[0].message.content or ""
        citations = self._parse_citations(answer_text, chunks)

        log.info(
            "answer_generator.done",
            num_citations=len(citations),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            model=GROQ_MODEL,
        )
