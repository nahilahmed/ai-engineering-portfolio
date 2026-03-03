# Project 01 — F1 Strategy RAG + Observability

## What this is
A hybrid RAG system that answers F1 strategy questions by combining:
- **Structured data** from the OpenF1 and Ergast APIs (lap times, tyre compounds, pit stops, telemetry)
- **Unstructured text** from race reports, press conference transcripts, and circuit guides

Example questions this system should answer well:
- *"Was Leclerc's lap 28 undercut on Norris the right call at Bahrain 2024?"*
- *"How does tyre deg at Monza compare to Silverstone historically?"*
- *"What did the team radio say when Verstappen pitted under the safety car in Brazil?"*

These questions require reasoning across structured telemetry AND unstructured race analysis —
that's what makes this technically interesting and hard to fake in an interview.

## Why this architecture is non-trivial
Most RAG demos use a single retrieval path: embed documents → query vector store → generate.
This system has two retrieval paths that need to be intelligently routed and merged:

```
User query
    ↓
Query classifier
    ↓                        ↓
Structured path          Unstructured path
OpenF1/Ergast API call   Vector search (ChromaDB)
    ↓                        ↓
         Merge results
              ↓
         Rerank chunks
              ↓
    Generate cited answer
```

The query classifier is a deliberate design decision worth defending in interviews.
Understand why it exists before you build it.

## Data Sources
- **OpenF1 API** — `https://api.openf1.org/v1/` — free, no auth, real telemetry data
- **Ergast API** — `https://ergast.com/api/f1/` — free, historical results back to 1950
- **Unstructured**: F1.com race reports, Wikipedia circuit/driver articles, press conference PDFs

---

## Phase 1 — Fundamentals
**Goal**: Get a working RAG pipeline end-to-end with F1 data and citations.

### Before you write any code, understand:
- What is chunking and why does overlap matter?
- What does an embedding actually represent geometrically?
- Why does top-k retrieval sometimes return irrelevant chunks?

### What to build
- [ ] OpenF1 data fetcher: lap times, pit stops, tyre stints for a given race
- [ ] Ergast fetcher: historical race results, constructor standings
- [ ] Unstructured ingestion: parse race report PDFs/HTML into clean text
- [ ] Chunking: 500–800 tokens, ~100 token overlap — document your reasoning for these numbers
- [ ] Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, M1-friendly)
- [ ] ChromaDB: store chunks with metadata (race, year, source type)
- [ ] Basic retrieval: top-k chunks for a query
- [ ] Answer generation with source citations via Groq
- [ ] CLI interface to test queries

### Deliverable
Ask "What happened with tyre strategy in the 2024 Monaco GP?" →
Get an answer that cites specific retrieved chunks by source.

### Files
```
phase-1-fundamentals/
├── fetchers/
│   ├── openf1_fetcher.py      # structured race data
│   └── ergast_fetcher.py      # historical results
├── ingestion/
│   ├── document_loader.py     # PDF/HTML → clean text
│   └── chunker.py             # chunking with overlap
├── embeddings/
│   └── embedder.py            # sentence-transformers wrapper
├── retrieval/
│   └── retriever.py           # ChromaDB query → top-k chunks
├── generation/
│   └── generator.py           # chunks + query → cited answer
├── app.py                     # CLI entrypoint
├── requirements.txt
└── .env.example
```

### Claude Code guidance for this phase
Use it to scaffold `openf1_fetcher.py` and `chunker.py`.
But before accepting any generated code, answer these yourself:
- Why 500–800 tokens and not 200 or 2000?
- What metadata should you store alongside each chunk in ChromaDB?
- How will you handle OpenF1 returning structured JSON vs text needing embedding?

---

## Phase 2 — Production Quality
**Goal**: Upgrade from demo to production-grade retrieval. This is where differentiation happens.

### Before you write any code, understand:
- What is BM25 and how does it differ from vector search mathematically?
- What does a cross-encoder do that a bi-encoder (embeddings) can't?
- Why would a system hallucinate even with retrieved context?

### What to build
- [ ] **Query classifier**: route query to structured path (API) vs unstructured path (vector) vs both
- [ ] **Hybrid retrieval**: BM25 keyword search + ChromaDB semantic search, results merged
- [ ] **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free)
- [ ] **Citation enforcement**: system declines to answer if retrieved chunks don't support a response
- [ ] **Prompt versioning**: all prompts in `configs/prompts.yaml`, version controlled
- [ ] **Pydantic output model**: structured response with answer + citations + confidence score

### The query classifier decision
For F1 queries, you need to decide: does this question need live API data, vector search, or both?
- "What was Verstappen's fastest lap in Bahrain 2024?" → structured API call
- "Why did Red Bull dominate 2023?" → unstructured vector search
- "Was the lap 34 pit stop the right call given tyre data?" → both

Design this classifier yourself before asking Claude Code to implement it.

### Files
```
phase-2-production/
├── routing/
│   └── query_classifier.py    # LLM-based query routing
├── retrieval/
│   ├── hybrid_retriever.py    # BM25 + vector fusion
│   └── reranker.py            # cross-encoder reranking
├── generation/
│   ├── citation_enforcer.py   # grounding validation
│   └── structured_output.py  # Pydantic response models
├── configs/
│   └── prompts.yaml           # versioned prompt templates
└── requirements.txt
```

---

## Phase 3 — Evaluation & CI Gating
**Goal**: Automated quality checks. Every code change triggers an eval run.

### Before you write any code, understand:
- What does RAGAS `faithfulness` actually measure?
- What's the difference between faithfulness and answer relevancy?
- Why does a CI quality gate matter for an AI system specifically?

### What to build
- [ ] Golden dataset: 50–100 F1 Q&A pairs manually verified by you
  - Mix of strategy questions, historical facts, tyre/telemetry questions
  - These must be questions YOU can verify the answer to — don't outsource this
- [ ] RAGAS evaluation script measuring:
  - `faithfulness` — is the answer grounded in retrieved chunks?
  - `answer_relevancy` — does the answer address the question?
  - `context_precision` — are the retrieved chunks actually useful?
- [ ] GitHub Actions workflow: eval runs on every PR
- [ ] Build fails if faithfulness < 0.75
- [ ] Eval report saved as artifact per run

### Your golden dataset should include
- Questions where the answer requires structured data (lap times, positions)
- Questions where the answer requires unstructured text (strategy reasoning)
- Questions where the answer requires BOTH
- A few trick questions where the corpus genuinely doesn't have the answer
  (the system should decline, not hallucinate)

### Files
```
phase-3-evaluation/
├── golden_dataset.json
├── evaluate.py
├── .github/
│   └── workflows/
│       └── eval.yml
└── reports/
```

---

## Phase 4 — Observability & Tracing
**Goal**: Full visibility into every request. Be able to explain any failure after the fact.

### Before you write any code, understand:
- What's the difference between logging and tracing?
- What does a trace span represent in a distributed system?
- Why do averages hide problems that percentiles reveal?

### Setup Langfuse locally (free, self-hosted)
```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up -d
# UI available at http://localhost:3000
```

### Instrument every step of the pipeline
For every single F1 query, trace:
- [ ] Query received + classified (which path: structured/unstructured/both)
- [ ] API calls made to OpenF1/Ergast (what was fetched)
- [ ] Chunks retrieved from ChromaDB (which ones, similarity scores)
- [ ] Reranker output (how ordering changed, by how much)
- [ ] Exact prompt sent to Groq
- [ ] LLM response + token counts
- [ ] Latency per step

### The value of this
You can answer: "Show me what happened on request #847 where the system hallucinated
Verstappen's pit stop lap." Pull up the trace → see which chunks were retrieved →
identify the root cause in 2 minutes.

### Files
```
phase-4-observability-tracing/
├── tracer.py                  # Langfuse integration wrapper
├── instrumented_pipeline.py   # full pipeline with tracing added
└── docker-compose.yml         # Langfuse local setup
```

---

## Phase 5 — Metrics Dashboard
**Goal**: Track system health over time. Think like an SRE for AI.

### Metrics to track

| Metric | Why |
|--------|-----|
| P50 / P95 latency | Averages hide worst-case experience |
| Cost per query | Quantify what each Groq call costs |
| Citation coverage | % of answers grounded in evidence |
| Classifier accuracy | Is routing working correctly? |
| Faithfulness over time | Is quality drifting? |

### Build two dashboards
1. **Langfuse built-in** — per-request traces and basic aggregations
2. **Grafana** (self-hosted Docker) — time-series metrics, percentile charts

### Files
```
phase-5-metrics-dashboard/
├── metrics_collector.py       # pulls from Langfuse API
├── cost_calculator.py         # per-query cost tracking
└── dashboards/
    ├── grafana_dashboard.json # importable Grafana config
    └── docker-compose.yml     # Grafana + Prometheus stack
```

---

## Phase 6 — Regression Gating
**Goal**: Connect observability back to CI. Quality regression = build blocked.

### What to build
- [ ] Golden dataset eval runs automatically in CI (extends Phase 3)
- [ ] Metrics compared to defined thresholds
- [ ] Prompt changes versioned in git with before/after metrics in changelog
- [ ] Alert if faithfulness drops more than 5% between runs

### Prompt versioning pattern
```yaml
# configs/prompts.yaml
version: "1.4.0"
changelog:
  - version: "1.4.0"
    date: "2025-03-15"
    change: "Tightened citation enforcement for structured data answers"
    faithfulness_before: 0.74
    faithfulness_after: 0.81
```

### Files
```
phase-6-regression-gating/
├── regression_check.py
├── .github/
│   └── workflows/
│       └── quality_gate.yml
└── PROMPT_CHANGELOG.md
```

---

## Interview talking points (earn these by building, not memorising)
- Why hybrid retrieval over pure vector search for F1 queries
- How the query classifier routes between structured and unstructured paths
- What RAGAS faithfulness measures and what threshold you chose and why
- How you identified a retrieval failure using Langfuse traces
- The trade-off between reranker quality and latency on M1

---

## LEARNINGS.md — fill this in as you go
After each phase, answer:
- What did you decide and why?
- What broke and how did you debug it?
- What would you change at 10x query volume?