# LEARNINGS — Project 01: F1 Strategy RAG + Observability

---

## Session 1 — 2026-03-01 | Phase 1: Fetchers

### What was built
- `phase-1-fundamentals/fetchers/openf1_fetcher.py` — OpenF1 API wrapper with Pydantic models
- `phase-1-fundamentals/fetchers/ergast_fetcher.py` — Jolpica/Ergast API wrapper
- `phase-1-fundamentals/api_test.ipynb` — interactive notebook for exploring both APIs

### Decisions made and why

**Jolpica over Ergast**
The original Ergast API (ergast.com) was deprecated after the 2024 season. Jolpica is a community-maintained fork with identical URL structure and JSON schema — drop-in replacement with no code changes needed beyond the base URL.

**Pydantic models with `extra="ignore"`**
Raw API dicts have no type guarantees and can change silently. Pydantic validates at the boundary — if a field is missing or the wrong type, it fails immediately with a clear error rather than silently downstream. `extra="ignore"` means new API fields added in future won't break the fetcher.

**`F1BaseModel` base class**
`model_config = ConfigDict(extra="ignore")` was being repeated on every model. Extracted into a shared base class so it's declared once and inherited — less repetition, easier to change globally if needed.

**`httpx.Client` inside a class (not standalone functions)**
`httpx.Client` holds a persistent TCP connection — reusing it across requests is faster than opening a new connection per call. Wrapping it in a class with `__enter__`/`__exit__` ensures the connection is always closed cleanly, even if something crashes mid-session.

**`_get()` as a private helper**
All four public methods (sessions, laps, pit stops, stints) need identical HTTP boilerplate: log the request, make the call, handle errors, return JSON. Extracting that into `_get()` means the logic lives in one place. Underscore prefix = internal only, not part of the public API.

**Dates kept as `str`, not `datetime`**
OpenF1 returns timestamps with timezone offsets (e.g. `2024-05-26T13:00:00+00:00`). Parsing to `datetime` introduces timezone handling decisions that aren't needed in Phase 1. Kept as strings — any component that needs to reason about time can parse it then, with full context.

**`float | None` on lap timing fields**
Safety car laps, in-laps, and DNF laps genuinely have no recorded time in the API — OpenF1 returns `null`. Declaring these as `float` (no default) would cause Pydantic to reject those rows entirely. `None` is a valid value here, not a data quality problem.

### What broke and how it was fixed

**OpenF1 returns HTTP 404 for empty results (not empty array)**
When a filter matches zero records, OpenF1 returns a 404 status rather than `[]`. `raise_for_status()` was treating that as a real error. Fixed by explicitly checking for 404 in `_get()` and returning `[]` instead of raising.

**Monaco `circuit_short_name` is `"Monte Carlo"` not `"Monaco"`**
OpenF1's naming isn't always intuitive. Discovered this when the session lookup returned 404. Fix: always browse `https://api.openf1.org/v1/sessions?year=2024` to find the exact `circuit_short_name` before filtering.

### Questions worth thinking about before next session
- How do we turn a driver's race worth of structured JSON into embeddable text? What does one chunk look like?
- What metadata should live alongside each chunk in ChromaDB?

---

## Session 2 — 2026-03-02 | Phase 1: Document Loader + Chunker

### What was built
- `phase-1-fundamentals/ingestion/document_loader.py` — `Document` model + `DocumentLoader` class
- `phase-1-fundamentals/ingestion/chunker.py` — `Chunk` model + `Chunker` class
- `beautifulsoup4` and `python-dotenv` added to `requirements.txt` and installed

### Decisions made and why

**`Document` and `Chunk` as separate Pydantic models**
Loader and chunker have distinct responsibilities. `Document` is raw clean text from one source. `Chunk` is an embeddable slice with a deterministic ID. Keeping them separate means each stage has a typed contract — the chunker can only receive valid Documents, the embedder can only receive valid Chunks.

**BeautifulSoup fallback container chain**
FIA pages don't use semantic HTML (`<article>`, `<main>`). Discovered this during smoke testing — the page returned nav/footer noise in the first and last 500 chars. Inspected the actual div classes and found `div.content-body` contains just the transcript. Built a fallback chain: `div.content-body` → `<article>` → `<main>` → `<body>` — most specific first, so FIA pages hit immediately and other sites still work.

**`decompose()` before text extraction**
Noise tags (`<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`) are removed from the tree permanently before `get_text()` is called. This is cleaner than post-processing the text with regex.

**180 words per chunk, 20 words overlap**
`all-MiniLM-L6-v2` has a hard 256-token limit. Tokens ≠ words — English prose averages ~1.3 tokens/word. 180 words × 1.3 ≈ 234 tokens, leaving safe headroom.

CRITICAL: chunk size is a correctness concern, not just a performance one. If you pass a chunk larger than 256 tokens to the embedding model, it does not error — it silently truncates. Tokens beyond position 256 are discarded before the model processes the input. The resulting vector represents only the first 256 tokens, not the full chunk. The embedding looks valid but is misleading. Originally proposed 600 words (≈780 tokens — 3x over the limit), which would have produced silently broken embeddings that only surface as poor retrieval quality much later.

Overlap confirmed working: last 20 words of chunk N == first 20 words of chunk N+1.

**Word-window chunker over LangChain**
Custom implementation chosen over LangChain's `RecursiveCharacterTextSplitter` — full transparency, zero extra dependencies, every line explainable in an interview. Trade-off: no sentence-boundary awareness, but acceptable for short conversational Q&A transcripts.

**Deterministic `chunk_id` via MD5**
`hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:12]` — same document always produces same IDs. Enables ChromaDB upsert (no duplicates on re-ingestion) without needing a separate ID store.

**structlog over print statements**
Key-value structured logs (`log.info("chunker.done", num_chunks=29)`) are machine-readable. Currently outputs to stdout only — no persistence. In Phase 4 (Observability) the same calls will be rewired to emit JSON into Langfuse with zero changes to the logging call sites.

### What broke and how it was fixed

**FIA page returned nav/footer noise in extracted text**
Initial `_extract_text` fell back to `<body>` for all pages. FIA doesn't use semantic HTML so nav/footer weren't being stripped by tag decomposition. Fixed by inspecting actual div class names with a quick script and adding `div.content-body` as the primary container target.

### Questions worth thinking about before next session
- The embedder needs to load `sentence-transformers` — first heavyweight model load. Install now or wait?
- ChromaDB setup: what metadata fields matter most for filtering at query time?

---

## Session 3 — 2026-03-02 | Phase 1: Embedder + VectorStore + Retriever

### What was built
- `phase-1-fundamentals/embeddings/embedder.py` — `EmbeddedChunk` model + `Embedder` class
- `phase-1-fundamentals/retrieval/vector_store.py` — `VectorStore` class (ChromaDB wrapper)
- `phase-1-fundamentals/retrieval/retriever.py` — `Retriever` class (text query → top-k chunks)
- `phase-1-fundamentals/rag_test.ipynb` — interactive notebook for exploring the full pipeline

### Decisions made and why

**`EmbeddedChunk` extends `Chunk` (not a flat duplicate)**
Pydantic inheritance means `EmbeddedChunk` gets `chunk_id`, `text`, and `metadata` from `Chunk` for free. Only the new field (`embedding: list[float]`) is declared. Avoids duplication and keeps the type contract clear — an `EmbeddedChunk` is always a valid `Chunk` plus a vector.

**`chunk.model_dump()` for Pydantic inheritance construction**
When building `EmbeddedChunk` from a `Chunk`, `**chunk.model_dump()` spreads all parent fields cleanly. Avoids manually threading `chunk_id=chunk.chunk_id, text=chunk.text, ...`. Safe because `model_dump()` returns only declared fields.

**`convert_to_numpy=True` + `.tolist()` on vectors**
`sentence-transformers` returns numpy arrays by default. Pydantic's `list[float]` validator rejects numpy arrays. `.tolist()` converts each row to a plain Python list — zero cost, required for Pydantic compatibility.

**Batch encoding — one `model.encode()` call on all texts**
Passing all chunk texts as a list lets `sentence-transformers` group them into batches of 32 internally and run matrix operations across the batch. Faster than calling `encode()` once per chunk. The model handles batching — our code just passes the full list.

**Model loaded once at `__init__`, never per request**
Loading `all-MiniLM-L6-v2` takes ~4s and ~300MB RAM. Loading it per call would be unusable in any pipeline. Instantiating `Embedder` once and reusing it is a hard requirement.

**Single ChromaDB collection with metadata filtering**
One collection (`f1_chunks`) for all unstructured F1 text. Source type, race, and year live in metadata. This allows filtering at query time (`where={"year": "2024"}`) without the operational overhead of managing multiple collections. Separate collections only justified if using different embedding models per source — not our case.

**`hnsw:space: cosine`**
ChromaDB defaults to L2 (Euclidean) distance. Cosine similarity is the correct metric for sentence-transformer embeddings — it measures angle between vectors (semantic direction) not absolute magnitude. Set explicitly so behaviour doesn't depend on ChromaDB's default changing.

**`upsert` not `add` in VectorStore**
`add` throws on duplicate `chunk_id`. `upsert` updates in place. Ingestion pipelines are often re-run (new data, bug fixes) — idempotency is a correctness requirement, not an optimisation.

**Dependencies injected into `Retriever`**
`Retriever.__init__` accepts `Embedder` and `VectorStore` rather than creating them internally. This keeps both as singletons in the pipeline — the same loaded model and open DB connection are shared across all retrieval calls. Creating them inside `Retriever` would cause re-loading on every instantiation.

**Query wrapped in a minimal `Chunk` for embedding**
`Embedder.embed()` expects `list[Chunk]`. For a query string, a placeholder `Chunk` is created with `chunk_id="query"` and empty metadata — only `.text` is used during embedding. The alternative (a separate `embed_text()` method) would duplicate the batch encoding logic.

### What broke and how it was fixed

**ChromaDB telemetry errors on startup**
`Failed to send telemetry event: capture() takes 1 positional argument but 3 were given` — version mismatch between ChromaDB and its internal telemetry library. Harmless, doesn't affect reads/writes. Can be silenced with `ANONYMIZED_TELEMETRY=False` env var if needed.

**`n_results` warning when corpus smaller than `top_k`**
`Number of requested results 3 is greater than number of elements in index 2` — ChromaDB auto-adjusts silently. Not an issue in production with hundreds of chunks.

### Observations from rag_test.ipynb
- Cosine distances on a 3-chunk corpus are close together (0.46–0.59) because all chunks are F1 content — semantically similar to any F1 query. In a real corpus of hundreds of chunks across many topics, irrelevant chunks score 0.7–0.9+ and the relevant ones stand out clearly.
- Ranking is correct — that's what matters, not absolute distance values.
- The reranker (Phase 2) will sharpen separation further using cross-encoder comparison.

### Questions worth thinking about before next session
- Answer generator: how should citations be structured? chunk_id only, or source + chunk_index?
- How many retrieved chunks should we pass to the LLM context? (top_k for generation vs retrieval may differ)

---

## Session 4 — 2026-03-03 | Phase 1: Answer Generator + CLI

### What was built
- `phase-1-fundamentals/generation/generator.py` — `Citation` + `GeneratedAnswer` Pydantic models + `AnswerGenerator` class
- `phase-1-fundamentals/cli.py` — interactive CLI wiring the full pipeline end to end

### Decisions made and why

**`AnswerGenerator` accepts pre-retrieved chunks, not a `Retriever` instance**
Phase 2 inserts a reranker between retrieval and generation. If the generator owned retrieval internally, adding the reranker would require modifying the generator. Keeping them separate means Phase 2 just slots in: `retrieve → rerank → generate`. Each component does one thing.

**Prompt stored as a module-level constant, not inside a method**
Burying the prompt inside `_build_user_message` makes it invisible — you have to read the method to find it. A module-level constant named `SYSTEM_PROMPT` is immediately findable, diffable in git, and clearly separated from logic. Phase 2 will migrate this to `configs/prompts.yaml` for full versioning with before/after metrics.

**Chunks labelled with their actual `chunk_id` in the prompt**
Each chunk in the prompt is prefixed `[{chunk_id}] source=...`. The LLM cites using those exact labels, and the regex parser looks up the same IDs in `chunk_by_id`. This tight coupling between prompt format and parser is intentional — the two must stay in sync.

**`temperature=0.1`**
Factual RAG needs the model to stay close to provided context, not be creative. Low temperature reduces the chance the model paraphrases away from the source material or invents plausible-sounding but unjustified claims.

**Token counts logged at `answer_generator.done`**
`prompt_tokens` and `completion_tokens` are already being captured from the Groq response. Phase 5 will read these structured logs to compute cost-per-query — the logging call sites won't need to change.

**`load_dotenv` before all imports in `cli.py`**
`GROQ_API_KEY` must be in the environment before `AnswerGenerator.__init__` runs (it checks the key at instantiation). If `load_dotenv` is called after the import, the class is already instantiated with a missing key. Order: load env → import components → instantiate.

**`sys.path.insert` in `cli.py`**
The CLI lives at the `phase-1-fundamentals/` root. Sibling packages (`embeddings/`, `retrieval/`, `generation/`) are not installed — they're just directories. `sys.path.insert(0, str(Path(__file__).parent))` adds the phase root so Python can find them when the script is run directly.

### What broke and how it was fixed

**Citations all returned empty on first smoke test**
`answer_generator.unknown_citation` warnings fired for every cited chunk. Root cause: the prompt labelled chunks as `[CHUNK 1]`, `[CHUNK 2]` etc., so the LLM cited those labels. But `_parse_citations` looked up IDs in `chunk_by_id` which uses actual chunk_ids (`c1`, `870e3a25577f`). The labels and the lookup keys were different types — one was a human-readable number, the other the raw DB ID.

Fix: label chunks with their actual `chunk_id` in `_build_user_message`. The prompt now shows `[c1] source=test` and the LLM cites `[c1]`, which the parser resolves correctly. Label format and parser must always be in sync.

### Observation: repeating text in test chunks

Chunks `870e3a25577f` and `f59ac3031799` showed the same sentence repeated multiple times in their text. This is a test data artifact — the source document was a single sentence (much shorter than the 180-word chunk target). When the chunker runs on text shorter than one chunk, the overlap window has nothing new to add, so the same content appears across adjacent chunk boundaries.

This does not happen with real documents. Race reports and press conference transcripts are hundreds to thousands of words — the chunker produces clean, non-overlapping 180-word windows. The overlap (20 words) only exists to preserve sentence context across chunk edges, not to duplicate content.

### Known limitations — to fix in Phase 2

**LLM hallucinated citation IDs**
When no chunk is relevant, the LLM invents a citation label (e.g. `[no relevant chunk]`, `[No relevant chunk ID available]`) instead of citing nothing. The `answer_generator.unknown_citation` warning catches and discards these, but the answer still goes through. Phase 2 citation enforcement should detect zero valid citations and either reject the answer or flag it with low confidence.

**Citation parser: comma-separated IDs in one bracket**
The LLM sometimes writes `[id1, id2, id3]` instead of `[id1] [id2] [id3]`. Fixed mid-session by splitting on commas inside each `[...]` match. Fragile — depends on the LLM consistently using commas as separators. Phase 2's structured output (Pydantic JSON schema for LLM response) would eliminate this regex parsing entirely.

**Citation excerpts show chunk headers, not the cited sentence**
The excerpt in each `Citation` is the first 30 words of the chunk, which is often the race result header (e.g. `"DRIVERS 1 – George Russell..."`) rather than the specific sentence the LLM cited. The excerpt is there to help verify the citation, but it's not pointing at the right part of the chunk. Phase 2 should either increase excerpt length or extract the sentence immediately surrounding the citation in the answer text.

**Corpus coverage gaps**
166 chunks from 4 press conferences. [Phase 2 note below]

---

## Session 5 — 2026-03-03 | Phase 2: Production Quality

### What was built
- Codebase restructured: all source packages moved to a single `src/` at the project root via `git mv` (preserves history). One project-level `requirements.txt` and `.env.example`.
- `src/retrieval/classifier.py` — `QueryRoute` enum, `ClassificationResult` Pydantic model, `QueryClassifier` (Groq, `temperature=0`, JSON mode)
- `src/fetchers/ergast_fetcher.py` — added `ScheduleRace` model + `get_schedule()` + `QualifyingResult` + `QualifyingResponse` models + `get_qualifying_results()`
- `src/retrieval/structured_retriever.py` — `StructuredRetriever` with 8-tool Groq function-calling loop (get_session, get_lap_times, get_pit_stops, get_stints, get_qualifying_results, get_race_results, get_driver_standings, get_constructor_standings)
- `src/retrieval/vector_store.py` — added `get_all()` for BM25 index construction
- `src/retrieval/bm25_retriever.py` — `BM25Okapi` index built once at startup; `retrieve()` returns chunks ordered by keyword relevance
- `src/retrieval/hybrid_retriever.py` — `_reciprocal_rank_fusion()` merges semantic + BM25 ranked lists; `HybridRetriever` fetches `2*top_k` candidates from each before fusion
- `src/retrieval/reranker.py` — `cross-encoder/ms-marco-MiniLM-L-6-v2` loaded once at init; `rerank()` adds `rerank_score` field to each chunk
- `src/pipeline.py` — `RAGPipeline` singleton orchestrator; `PipelineResult` Pydantic model wrapping `GeneratedAnswer` + routing metadata
- `cli.py` updated to use `RAGPipeline`, displaying route/confidence/reasoning

### Decisions made and why

**No LangChain / LangGraph for Phase 2**
LangChain abstracts away the exact API calls happening under the hood — you can't see what goes into `create()` without reading source. For a portfolio where every decision must be explainable, plain Python makes the architecture transparent. LangGraph deferred to future phases where its graph abstraction adds genuine value (complex conditional flows).

**Groq function calling (tool use), not a parameter extractor**
An alternative was a simple LLM call: "Extract year and race name from this question". That would only work for the simplest queries. Proper tool calling lets the LLM chain multiple API calls: `get_session → get_stints`, decide whether to use OpenF1 vs Jolpica, and stop when it has enough data. More powerful and a more interesting portfolio piece.

**Two data source separation (OpenF1 vs Jolpica)**
OpenF1: telemetry (lap times, sector times, pit durations, tyre compounds) — 2023 onwards.
Jolpica: results and standings (race results, qualifying, championship standings) — all seasons back to 1950.
The system prompt and tool descriptions are the contract: the LLM reads them at inference time to decide which tools to call. Tool descriptions embed lookup tables (circuit short names, driver numbers) because the LLM's training data may have stale values.

**`_resolve_round()` internal to StructuredRetriever**
The `get_race_results` and `get_qualifying_results` tools accept a human-readable `race_name` string. Internally, `_resolve_round()` calls `get_schedule()` to resolve it to a round number before the Jolpica fetch. This hides the Ergast round-number convention from the LLM — the tool's interface stays natural.

**Synthetic chunk for structured path**
`StructuredRetriever.retrieve()` returns a formatted text string. Rather than creating a special path in the generator, the text is wrapped as `{"chunk_id": "structured_data", "text": api_text, "metadata": {"source": "live_api"}}` — the same dict shape the vector store returns. `AnswerGenerator` handles all routes uniformly.

**`2*top_k` candidates for RRF**
If each retriever only fetches `top_k`, a chunk ranked 6th by BM25 and 7th by vector would be discarded before fusion — even though its combined signal might be the best result. Fetching `2*top_k` gives RRF enough material to find these cross-list winners.

**RRF over score normalisation**
An alternative is to normalise BM25 scores (0–1) and vector distances (0–1) and add them. Problem: different retrievers produce scores on incompatible scales — a BM25 score of 3.2 vs a cosine distance of 0.4 have no inherent relationship. RRF uses only rank position, which is comparable across any two retrievers regardless of how they score.

**Cross-encoder loaded once at startup**
Loading `cross-encoder/ms-marco-MiniLM-L-6-v2` takes ~2s and ~90MB RAM. Loading it per request would be unusable. Reranking 5–10 pairs on an already-loaded model takes ~50ms — acceptable for dev.

**`get_qualifying_results` added after discovering the gap**
"Who won pole?" routes to the structured path, but we had no qualifying tool. The LLM tried to infer pole from `get_lap_times` for individual drivers — inefficient and it guessed wrong driver numbers. Adding `get_qualifying_results` (Jolpica's `/qualifying` endpoint) gives the LLM a direct, single-call path.

### What broke and how it was fixed

**`result.race_name` AttributeError**
`RaceResultsResponse` field is `raceName` (camelCase, matching Ergast schema). Used `result.race_name` (snake_case). Fixed at call site.

**LLM called `get_session` before Jolpica queries**
System prompt said "ALWAYS call get_session first". LLM over-applied this to all tools. Fixed: "Call get_session ONLY if you need get_lap_times, get_pit_stops, or get_stints. Do NOT call get_session before Jolpica tools."

**Singapore 404 — wrong circuit_short_name**
Tool description had `"Marina Bay"`. Actual OpenF1 value is `"Singapore"`. Also: Abu Dhabi = `"Yas Marina Circuit"`, Spain = `"Catalunya"`, Austria = `"Spielberg"`. Fix: query the sessions API to get exact names, embed correct values in tool description.

**Groq `tool_use_failed` (400 error)**
`llama-3.3-70b-versatile` occasionally generates tool calls in Hermes-style XML format instead of OpenAI-compatible JSON. Groq returns 400. Fixed by wrapping `create()` in `try/except BadRequestError` — retries once. If retry also fails, returns a graceful error string rather than crashing.

**"Who won pole" returned empty**
LLM tried to infer pole from `get_lap_times` for a single driver (guessed #33, which doesn't exist). Root cause: no qualifying tool existed. Fixed by adding `get_qualifying_results` tool backed by Jolpica's `/qualifying` endpoint.

### Remaining Phase 2 items
- Prompt versioning (`configs/prompts.yaml`)
- Structured output + citation enforcement on the generator (Pydantic JSON schema response, decline if zero valid citations)

---

**Original corpus coverage note:**
166 chunks from 4 press conferences. Specific phrasings and topics not discussed in those transcripts return empty or hallucinated answers. More documents = better coverage. Before Phase 2 testing, consider ingesting 8–10 press conferences across more of the 2024 season.
