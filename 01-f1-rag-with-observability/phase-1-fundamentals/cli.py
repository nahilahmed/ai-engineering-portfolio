"""
CLI for Phase 1 — F1 Strategy RAG.

Usage:
    conda run -n f1-rag python cli.py

Wires together the full Phase 1 pipeline:
    Embedder → VectorStore → Retriever → AnswerGenerator

All components are initialised once at startup and reused across queries.
Type a question at the prompt, or 'quit' / 'exit' to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any component imports so GROQ_API_KEY is available
load_dotenv(Path(__file__).parent / ".env")

# Add phase root to sys.path so sibling packages (embeddings, retrieval, etc.) resolve
sys.path.insert(0, str(Path(__file__).parent))

from embeddings.embedder import Embedder
from generation.generator import AnswerGenerator
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore


def print_answer(result) -> None:  # type: ignore[no-untyped-def]
    """Pretty-print a GeneratedAnswer to the terminal."""
    print("\n" + "=" * 60)
    print("Answer:")
    print(result.answer)

    if result.citations:
        print("\nCitations:")
        for i, citation in enumerate(result.citations, start=1):
            print(f"  [{i}] {citation.chunk_id} | {citation.source}")
            print(f"      \"{citation.excerpt}...\"")
    else:
        print("\nCitations: none (LLM did not cite any chunks)")

    print("=" * 60 + "\n")


def main() -> None:
    print("Initialising RAG pipeline...")

    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    generator = AnswerGenerator()

    chunk_count = vector_store.count()
    if chunk_count == 0:
        print("\nWarning: ChromaDB is empty — no chunks to retrieve from.")
        print("Run the ingestion pipeline first (see rag_test.ipynb).\n")

    print(f"\nReady. {chunk_count} chunks in vector store.")
    print("Type a question, or 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        chunks = retriever.retrieve(query, top_k=5)
        result = generator.generate(query=query, chunks=chunks)
        print_answer(result)


if __name__ == "__main__":
    main()
