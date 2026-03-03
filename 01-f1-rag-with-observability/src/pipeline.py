"""
RAG Pipeline for Phase 2 — F1 Strategy RAG.

Responsibility: orchestrate all components end-to-end for a single query.
Owns component initialization (all singletons) and routing logic.

Query flow:
  1. Classifier  → STRUCTURED / UNSTRUCTURED / HYBRID
  2a. STRUCTURED  → StructuredRetriever (tool-calling loop over live APIs)
  2b. UNSTRUCTURED → HybridRetriever (BM25 + vector) → Reranker
  2c. HYBRID      → both 2a and 2b, chunks merged before generation
  3. AnswerGenerator → GeneratedAnswer with citations

All components are instantiated once at init and reused across queries.
This class is imported by both cli.py and the FastAPI server (Phase 4).
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from embeddings.embedder import Embedder
from fetchers.ergast_fetcher import JolpicaFetcher
from fetchers.openf1_fetcher import OpenF1Fetcher
from generation.generator import AnswerGenerator, GeneratedAnswer
from retrieval.bm25_retriever import BM25Retriever
from retrieval.classifier import ClassificationResult, QueryClassifier, QueryRoute
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from retrieval.structured_retriever import StructuredRetriever
from retrieval.vector_store import VectorStore

log = structlog.get_logger()


class PipelineResult(BaseModel):
    """Full output of one pipeline query — answer plus routing metadata.

    Wraps GeneratedAnswer with classification context so the CLI and API
    can show which retrieval path ran and how confident the classifier was.
    """

    answer: GeneratedAnswer
    route: QueryRoute
    reasoning: str       # classifier's one-sentence explanation
    confidence: float    # classifier's self-reported confidence (0–1)


class RAGPipeline:
    """End-to-end F1 RAG pipeline.

    Instantiate once at startup. All heavy components (embedding model,
    cross-encoder, ChromaDB) are loaded during __init__ and reused.
    query() is the only public method — call it for every user question.
    """

    def __init__(self) -> None:
        log.info("pipeline.initialising")

        # --- Shared infrastructure ---
        self._embedder = Embedder()
        self._vector_store = VectorStore()

        # --- Classification ---
        self._classifier = QueryClassifier()

        # --- Unstructured path ---
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
        )
        self._bm25 = BM25Retriever(vector_store=self._vector_store)
        self._hybrid = HybridRetriever(
            retriever=self._retriever,
            bm25_retriever=self._bm25,
        )
        self._reranker = Reranker()

        # --- Structured path ---
        self._openf1 = OpenF1Fetcher()
        self._jolpica = JolpicaFetcher()
        self._structured = StructuredRetriever(
            openf1=self._openf1,
            jolpica=self._jolpica,
        )

        # --- Generation ---
        self._generator = AnswerGenerator()

        log.info("pipeline.ready")

    def query(self, question: str) -> PipelineResult:
        """Run a question through the full pipeline and return a cited answer.

        Args:
            question: raw user question

        Returns:
            PipelineResult with the answer, citations, route, and classifier metadata.
        """
        log.info("pipeline.query", question=question)

        # Step 1 — Classify
        classification: ClassificationResult = self._classifier.classify(question)
        route = classification.route
        log.info("pipeline.route", route=route.value, confidence=classification.confidence)

        # Step 2 — Retrieve
        chunks: list[dict] = []

        if route in (QueryRoute.STRUCTURED, QueryRoute.HYBRID):
            api_text = self._structured.retrieve(question)
            # Wrap API result as a synthetic chunk so the generator handles it uniformly
            chunks.append({
                "chunk_id": "structured_data",
                "text": api_text,
                "metadata": {"source": "live_api", "type": "structured"},
            })

        if route in (QueryRoute.UNSTRUCTURED, QueryRoute.HYBRID):
            hybrid_results = self._hybrid.retrieve(question, top_k=5)
            reranked = self._reranker.rerank(question, hybrid_results, top_k=3)
            chunks.extend(reranked)

        # Step 3 — Generate
        answer: GeneratedAnswer = self._generator.generate(
            query=question,
            chunks=chunks,
        )

        return PipelineResult(
            answer=answer,
            route=route,
            reasoning=classification.reasoning,
            confidence=classification.confidence,
        )
