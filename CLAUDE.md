# CLAUDE.md — AI Engineering Portfolio

This file is read by Claude Code at the start of every session.
Do not delete or move it.

---

## What this repo is
A portfolio of 4 production-grade AI engineering projects built for two purposes:
1. **Learning** — understanding how real AI systems are designed and operated
2. **Showcasing** — demonstrating engineering depth to hiring managers at leading data companies.

The projects are intentionally sequenced — each one builds on the previous.

---

## How to work with me (Claude Code)
This portfolio is built for learning, not just output. Follow these rules in every session:

**Before generating any code:**
- Ask me what I already understand about the component we're building
- Explain the "why" behind architectural decisions, not just the "how"
- If I'm about to make a trade-off, surface it explicitly so I can decide

**When generating code:**
- Scaffold structure and boilerplate — don't write the entire logic unprompted
- Add comments explaining non-obvious decisions
- Keep functions small and single-purpose
- If something has multiple valid approaches, show me 2 options with trade-offs

**Never:**
- Generate an entire phase in one shot without explanation
- Use a paid API where a free alternative exists (see Budget section)
- Add dependencies without explaining what they do and why they're needed
- Skip error handling or logging — these are portfolio projects, not scripts

**Always:**
- Default to Groq for LLM calls (free tier)
- Use sentence-transformers locally for embeddings (never OpenAI embeddings)
- Remind me to fill in LEARNINGS.md after each phase completes
- Check PROGRESS.md to understand what's already been built

---

## Stack & Tool Decisions

### LLMs
- **Primary**: Groq API — model `llama-3.3-70b-versatile` (free tier)
- **Fallback**: OpenAI `gpt-4o-mini` — only when Groq is insufficient, use sparingly
- **Local**: Ollama — for Project 02 benchmarking only
- **Never suggest**: OpenAI embeddings, GPT-4o (too expensive), Claude API (unnecessary cost)

### Embeddings
- `sentence-transformers/all-MiniLM-L6-v2` — runs locally on M1, completely free
- Load once and reuse — don't reload the model per request

### Vector Store
- ChromaDB — local, no server needed for dev
- Persist to `/data/chroma_db/` — never commit this directory to git

### Monitoring
- Langfuse — self-hosted via Docker Compose, UI at `http://localhost:3000`
- Grafana — self-hosted via Docker Compose, UI at `http://localhost:3001`
- OrbStack preferred over Docker Desktop on M1 (lighter, faster)

### Reranker
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — local, free, M1-compatible
- Load once at startup, not per request

### Evaluation
- RAGAS — for RAG quality metrics
- Custom eval scripts for fine-tuning metrics

### API
- FastAPI for all HTTP/WebSocket endpoints
- Pydantic for all input/output validation — no raw dicts in function signatures

---

## Hardware Context
- **Machine**: M1 Mac
- **Storage**: Projects live on external SSD
- **RAM**: Limited — be mindful of loading multiple large models simultaneously
- **Docker**: OrbStack (lighter than Docker Desktop on M1)

### M1-specific notes
- Ollama runs natively on M1 — use Metal acceleration, it's enabled by default
- sentence-transformers runs well on M1 CPU — no GPU needed for MiniLM
- Cross-encoder reranker is CPU-bound but fast enough for dev workloads
- For fine-tuning (Project 03): use Google Colab T4, not local M1

---

## Budget Rules
- **Monthly target**: $5–10 maximum across all projects
- **Groq free tier**: use for all development and testing
- **OpenAI**: only for final demo recordings — track usage carefully
- **No paid vector DBs, no paid monitoring tools, no paid embedding APIs**
- If you suggest a tool that costs money, flag it explicitly with estimated cost

---

## Project Overview

### 01 — F1 Strategy RAG + Observability
**Data sources**: OpenF1 API + Ergast API (structured) + race reports/transcripts (unstructured)
**Key complexity**: Dual retrieval paths — query classifier routes between API calls and vector search
**Phases**: 6 (fundamentals → production → evaluation → tracing → metrics → regression gating)
**Status**: See PROGRESS.md

### 02 — Local Model Benchmarking
**Models**: llama3.2:1b, llama3.2:3b, mistral:7b + quantized variants
**Key deliverable**: Technical comparison report with actual benchmark numbers
**Phases**: 3 (setup → structured output → comparison report)
**Status**: See PROGRESS.md

### 03 — Fine-Tuning (LoRA + DPO)
**Task**: F1 incident/strategy event extraction from unstructured text → structured JSON
**Base model**: Qwen2.5-3B-Instruct
**Training**: Google Colab T4 (free tier) — not local M1
**Phases**: 2 (SFT → DPO)
**Status**: See PROGRESS.md

### 04 — Streaming Log Analyzer
**Architecture**: Log generator → FastAPI WebSocket → anomaly detector → Groq LLM → dashboard
**Key complexity**: Latency decomposition per pipeline stage + graceful degradation
**Optional**: ClickHouse Cloud integration (free tier)
**Phases**: 3 (pipeline → latency tracking → resilience)
**Status**: See PROGRESS.md

---

## Code Conventions

### Python
- Python 3.11+
- Type hints on all function signatures
- Pydantic models for all data structures that cross module boundaries
- `python-dotenv` for env vars — never hardcode keys
- Logging via `structlog` — structured JSON logs, not print statements
- One class or logical group per file — keep files focused

### Project structure per phase
```
phase-X-name/
├── README.md         # what this phase builds and why
├── requirements.txt  # phase-specific dependencies
├── .env.example      # env vars needed for this phase
└── src/              # all source code
```

### Git
- Branch per phase: `project-01/phase-1-fundamentals`
- Commit at meaningful checkpoints — not just "done"
- Commit message format: `feat:`, `fix:`, `docs:`, `chore:`
- Never commit: `.env`, `chroma_db/`, `*.gguf`, `models/`, `data/raw/`

### Environment
- Every phase has its own `requirements.txt`
- Use `venv` per project: `python -m venv .venv && source .venv/bin/activate`
- Shared utilities live in `shared/utils/` and are imported directly

---

## Current Session
When starting a session, tell me:
1. Which project and phase you're working on
2. What you completed last session
3. What specific component you want to build today

Then check PROGRESS.md together before writing any code.