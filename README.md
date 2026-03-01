# AI Engineering Portfolio

4 production-grade AI projects built to demonstrate real engineering depth —
not tutorial-following, but deliberate system design with measurable outcomes.

## How this was built
Each project follows a 3–6 phase progression. For every phase:
- Architecture and trade-off decisions were made independently
- Claude Code was used to accelerate scaffolding and boilerplate
- Every component was run, broken, debugged, and understood before moving on
- LEARNINGS.md documents what actually happened — not what was supposed to happen

## Projects

| # | Project | Core Skills | Target Signal |
|---|---------|-------------|---------------|
| 01 | F1 Strategy RAG + Observability | Hybrid retrieval, eval, CI, monitoring | All AI teams |
| 02 | Local Model Benchmarking | Infra, perf engineering, model selection | Fivetran, ClickHouse |
| 03 | Fine-Tuning (LoRA + DPO) | ML training, alignment, data quality | Model-centric teams |
| 04 | Streaming Log Analyzer | Real-time pipelines, latency, resilience | ClickHouse, Fivetran ⭐ |

## Budget
- **Monthly target**: $5–10 max
- **Primary LLM**: Groq free tier (`llama-3.3-70b-versatile`)
- **Embeddings**: `sentence-transformers` locally — M1 optimized, free
- **Vector store**: ChromaDB local — free
- **Monitoring**: Langfuse self-hosted via Docker — free
- **OpenAI**: Only for final demos where Groq falls short
- **Fine-tuning GPU**: Google Colab T4 free tier first, Runpod (~$0.30/hr) only if needed

## Stack
- **Orchestration**: LangChain / LangGraph
- **APIs**: OpenF1 (free, no auth), Ergast (free)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — local, free
- **Evaluation**: RAGAS
- **Monitoring**: Langfuse + Grafana (both self-hosted)
- **Streaming**: FastAPI + WebSockets
- **Local models**: Ollama (M1 native)