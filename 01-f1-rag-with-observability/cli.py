"""
CLI for F1 Strategy RAG — Project 01.

Usage:
    conda run -n f1-rag python cli.py

Runs the Phase 2 pipeline end-to-end:
    Classifier → [Structured | Hybrid+Reranker | Both] → AnswerGenerator
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any component imports so GROQ_API_KEY is available
load_dotenv(Path(__file__).parent / ".env")

# Add src/ to sys.path so packages resolve
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import PipelineResult, RAGPipeline


def print_result(result: PipelineResult) -> None:
    """Pretty-print a PipelineResult to the terminal."""
    print("\n" + "=" * 60)
    print(f"Route : {result.route.value.upper()}  (confidence={result.confidence:.2f})")
    print(f"Reason: {result.reasoning}")
    print()
    print("Answer:")
    print(result.answer.answer)

    if result.answer.citations:
        print("\nCitations:")
        for i, c in enumerate(result.answer.citations, start=1):
            print(f"  [{i}] {c.chunk_id} | {c.source}")
            print(f"      \"{c.excerpt}...\"")
    else:
        print("\nCitations: none")

    print("=" * 60 + "\n")


def main() -> None:
    print("Initialising Phase 2 RAG pipeline...")
    pipeline = RAGPipeline()

    chunk_count = pipeline._vector_store.count()
    if chunk_count == 0:
        print("\nWarning: ChromaDB is empty — run ingestion first.\n")

    print(f"\nReady. {chunk_count} chunks in vector store.")
    print("Type a question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        result = pipeline.query(question)
        print_result(result)


if __name__ == "__main__":
    main()
