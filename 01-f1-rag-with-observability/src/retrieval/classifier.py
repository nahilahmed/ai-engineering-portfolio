"""
Query Classifier for Phase 2 — F1 Strategy RAG.

Responsibility: classify an incoming user query into one of three routes:
  - STRUCTURED   → answer comes from OpenF1 / Jolpica API data
  - UNSTRUCTURED → answer comes from vector store (press conferences, reports)
  - HYBRID       → needs both API data and vector store context

One Groq call with temperature=0 and JSON mode.
Result drives which retrieval path(s) run downstream.
"""

from __future__ import annotations

import json
import os
from enum import Enum

import structlog
from groq import Groq
from pydantic import BaseModel

log = structlog.get_logger()

GROQ_MODEL = "llama-3.3-70b-versatile"

CLASSIFIER_PROMPT = """\
You are a routing classifier for an F1 strategy question-answering system.

The system has two data sources:
1. STRUCTURED — live API data: lap times, pit stop durations, tyre stints, \
race results, driver standings, fastest laps, sector times. Use this when the \
question asks for a specific number, result, or recorded fact.

2. UNSTRUCTURED — text documents: FIA press conference transcripts, post-race \
interviews, team radio summaries, strategy analysis articles. Use this when the \
question asks for explanations, quotes, opinions, or reasoning.

Classify the user's question into exactly one of three routes:
- "structured"   → the answer is a specific fact or number from race telemetry or results APIs
- "unstructured" → the answer requires reading press conference transcripts or written analysis
- "hybrid"       → the answer needs both (e.g. a lap time delta AND what the team said about it)

Respond with valid JSON only, no other text:
{
  "route": "structured" | "unstructured" | "hybrid",
  "reasoning": "one sentence explaining your decision",
  "confidence": 0.0 to 1.0
}
"""


class QueryRoute(str, Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    HYBRID = "hybrid"


class ClassificationResult(BaseModel):
    route: QueryRoute
    reasoning: str
    confidence: float


class QueryClassifier:
    """Classifies a user query into a retrieval route via a single Groq call.

    Uses temperature=0 for determinism and JSON mode to avoid regex parsing.
    Instantiate once and reuse — the Groq client holds a persistent connection.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
        self._client = Groq(api_key=api_key)
        log.info("query_classifier.ready", model=GROQ_MODEL)

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query into STRUCTURED, UNSTRUCTURED, or HYBRID.

        Args:
            query: the raw user question

        Returns:
            ClassificationResult with route, reasoning, and confidence.
        """
        log.info("query_classifier.classifying", query=query)

        response = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        result = ClassificationResult(**data)

        log.info(
            "query_classifier.done",
            route=result.route,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )

        return result
