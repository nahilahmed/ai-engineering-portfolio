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
