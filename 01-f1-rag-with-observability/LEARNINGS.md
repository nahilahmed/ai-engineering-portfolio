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
