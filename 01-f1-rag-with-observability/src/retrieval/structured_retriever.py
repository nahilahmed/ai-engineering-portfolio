"""
Structured Retriever for Phase 2 — F1 Strategy RAG.

Responsibility: answer structured F1 queries by calling live APIs via an
LLM tool-calling loop. The LLM decides which tools to call and in what order;
this class executes those calls and returns a formatted text summary for the
AnswerGenerator.

Two data sources exposed as tools:
  OpenF1  — telemetry (lap times, pit stops, tyre stints) — 2023 onwards
  Jolpica — results and standings (all seasons back to 1950)

The LLM must call get_session before any other OpenF1 tool to obtain a
session_key. This chain is enforced via tool descriptions and the system prompt.
"""

from __future__ import annotations

import json
import os

import structlog
from groq import BadRequestError, Groq

from fetchers.ergast_fetcher import JolpicaFetcher
from fetchers.openf1_fetcher import OpenF1Fetcher

log = structlog.get_logger()

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """\
You are an F1 data analyst with access to two live APIs.

OpenF1 (telemetry — 2023 onwards only):
- get_session: returns a session_key needed by the other OpenF1 tools.
- get_lap_times: lap-by-lap timing, fastest lap, sector times.
- get_pit_stops: pit stop durations and which lap each stop occurred.
- get_stints: tyre compounds and stint lengths.

Jolpica (results and standings — all seasons back to 1950):
- get_qualifying_results: qualifying classification, pole position, Q1/Q2/Q3 times.
- get_race_results: final race classification, finishing positions, points, retirements.
- get_driver_standings: driver championship standings.
- get_constructor_standings: team championship standings.

Rules:
1. Call get_session ONLY if you need to use get_lap_times, get_pit_stops, or get_stints.
   Do NOT call get_session before Jolpica tools — they work without it.
2. For data before 2023, use only Jolpica tools (OpenF1 has no pre-2023 data).
3. Call only the tools you need — do not over-fetch.
4. After retrieving data, return a clear factual summary with specific numbers.\
"""

# ---------------------------------------------------------------------------
# Tool schemas — descriptions are the contract between us and the LLM.
# The LLM reads these at inference time to decide which tool to call.
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_session",
            "description": (
                "Look up an F1 session and return its session_key. "
                "ALWAYS call this before get_lap_times, get_pit_stops, or get_stints — "
                "those tools require a session_key. OpenF1 data: 2023 onwards only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Season year, e.g. 2024"},
                    "circuit_short_name": {
                        "type": "string",
                        "description": (
                            "OpenF1 circuit short name. Use exactly: "
                            "'Silverstone' (Britain), 'Monza' (Italy), "
                            "'Singapore' (Singapore — NOT 'Marina Bay'), "
                            "'Monte Carlo' (Monaco — NOT 'Monaco'), "
                            "'Spa-Francorchamps' (Belgium), 'Austin' (USA/COTA), "
                            "'Interlagos' (Brazil), 'Yas Marina Circuit' (Abu Dhabi), "
                            "'Sakhir' (Bahrain), 'Jeddah' (Saudi Arabia), "
                            "'Melbourne' (Australia), 'Suzuka' (Japan), "
                            "'Shanghai' (China), 'Imola', 'Catalunya' (Spain), "
                            "'Hungaroring' (Hungary), 'Zandvoort' (Netherlands), "
                            "'Baku' (Azerbaijan), 'Lusail' (Qatar), "
                            "'Spielberg' (Austria), 'Montreal' (Canada), "
                            "'Miami', 'Las Vegas', 'Mexico City'"
                        ),
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Session type. Defaults to 'Race' if not specified.",
                        "enum": ["Race", "Qualifying", "Sprint", "Practice 1", "Practice 2", "Practice 3"],
                    },
                },
                "required": ["year", "circuit_short_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lap_times",
            "description": (
                "Get lap-by-lap timing data for a driver. "
                "Use for: fastest lap time, sector times, lap count. "
                "Requires session_key from get_session. OpenF1: 2023 onwards only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer", "description": "From get_session"},
                    "driver_number": {
                        "type": "integer",
                        "description": (
                            "Driver's permanent race number. "
                            "VER=1, SAR=2, RIC=3, NOR=4, HAM=44, RUS=63, "
                            "LEC=16, SAI=55, ALO=14, STR=18, OCO=31, GAS=10, "
                            "TSU=22, HUL=27, MAG=20, BOT=77, ZHO=24, ALB=23, "
                            "PIA=81, COL=43"
                        ),
                    },
                },
                "required": ["session_key", "driver_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pit_stops",
            "description": (
                "Get pit stop data — stationary time in seconds and lap number. "
                "Use for: pit stop duration, number of stops, which lap each stop occurred. "
                "Requires session_key from get_session. OpenF1: 2023 onwards only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer"},
                },
                "required": ["session_key", "driver_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stints",
            "description": (
                "Get tyre stint data — compound (SOFT/MEDIUM/HARD/INTERMEDIATE/WET), "
                "stint length in laps, tyre age at stint start. "
                "Use for: tyre strategy, compound choice, undercut/overcut analysis. "
                "Requires session_key from get_session. OpenF1: 2023 onwards only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer"},
                },
                "required": ["session_key", "driver_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_qualifying_results",
            "description": (
                "Get qualifying classification — pole position, grid order, Q1/Q2/Q3 times. "
                "Use for: who won pole, qualifying order, fastest qualifying lap. "
                "P1 in results = pole sitter. "
                "Jolpica data: all seasons back to 1950."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "race_name": {
                        "type": "string",
                        "description": "Race name or partial name, e.g. 'Monaco', 'Abu Dhabi', 'British Grand Prix'",
                    },
                },
                "required": ["year", "race_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_race_results",
            "description": (
                "Get the final race classification — finishing position, grid, "
                "points scored, status (Finished / Retired / +N Laps). "
                "Use for: who won, where a driver finished, points scored, retirements. "
                "For the fastest lap award, look for fastest_lap_rank=1. "
                "Jolpica data: historical back to 1950, all seasons."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "race_name": {
                        "type": "string",
                        "description": "Race name or partial name, e.g. 'Monaco', 'Singapore', 'British Grand Prix'",
                    },
                },
                "required": ["year", "race_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_driver_standings",
            "description": (
                "Get driver championship standings — position, points, wins. "
                "Use for: championship order, points gap between drivers. "
                "Jolpica data: all seasons back to 1950."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "round_number": {
                        "type": "integer",
                        "description": "Standings after this round. Omit for end-of-season standings.",
                    },
                },
                "required": ["year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_constructor_standings",
            "description": (
                "Get constructor/team championship standings — position, points, wins. "
                "Use for: team standings, constructors' championship order. "
                "Jolpica data: all seasons back to 1958."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "round_number": {
                        "type": "integer",
                        "description": "Standings after this round. Omit for end-of-season standings.",
                    },
                },
                "required": ["year"],
            },
        },
    },
]


class StructuredRetriever:
    """Answers structured F1 queries by running a live tool-calling loop.

    The LLM reads tool descriptions, decides which APIs to call and in what
    order, and produces a factual summary. This class executes the tool calls
    and feeds results back into the conversation until the LLM is done.

    Dependencies are injected — OpenF1Fetcher and JolpicaFetcher are shared
    singletons from the pipeline startup, not re-instantiated per request.
    """

    def __init__(self, openf1: OpenF1Fetcher, jolpica: JolpicaFetcher) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
        self._client = Groq(api_key=api_key)
        self._openf1 = openf1
        self._jolpica = jolpica
        log.info("structured_retriever.ready")

    def retrieve(self, query: str) -> str:
        """Run the tool-calling loop and return formatted API results as text.

        The returned string is passed directly to the AnswerGenerator as context,
        the same way vector store chunks are used in the unstructured path.

        Args:
            query: the raw user question (already classified as STRUCTURED)

        Returns:
            Formatted text summary of API results, or an error message.
        """
        messages: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        log.info("structured_retriever.start", query=query)

        for iteration in range(MAX_ITERATIONS):
            try:
                response = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except BadRequestError as exc:
                # Groq occasionally returns 400 tool_use_failed when the model
                # generates a malformed tool call (e.g. Hermes-style XML instead
                # of JSON). Retry the same turn once — it resolves most of the time.
                if "tool_use_failed" not in str(exc):
                    raise
                log.warning(
                    "structured_retriever.tool_use_failed_retry",
                    iteration=iteration + 1,
                )
                try:
                    response = self._client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                except BadRequestError:
                    log.error(
                        "structured_retriever.tool_use_failed_unrecoverable",
                        iteration=iteration + 1,
                    )
                    return "Could not retrieve structured data — tool generation failed after retry."
            msg = response.choices[0].message

            # Append the assistant turn — may contain tool_calls or a final answer
            messages.append(msg)

            if not msg.tool_calls:
                # No more tool calls — LLM has enough data to answer
                log.info("structured_retriever.done", iterations=iteration + 1)
                return msg.content or "No structured data found for this query."

            log.info(
                "structured_retriever.tool_calls",
                count=len(msg.tool_calls),
                iteration=iteration + 1,
                tools=[tc.function.name for tc in msg.tool_calls],
            )

            # Execute each tool call and feed results back as tool messages
            for tc in msg.tool_calls:
                result = self._execute_tool(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        log.warning("structured_retriever.max_iterations_reached", query=query)
        return "Could not retrieve structured data — tool call limit reached."

    # -----------------------------------------------------------------------
    # Tool dispatch
    # -----------------------------------------------------------------------

    def _execute_tool(self, name: str, arguments_json: str) -> str:
        """Parse arguments and dispatch to the correct tool implementation."""
        args = json.loads(arguments_json)
        try:
            match name:
                case "get_session":
                    return self._tool_get_session(**args)
                case "get_lap_times":
                    return self._tool_get_lap_times(**args)
                case "get_pit_stops":
                    return self._tool_get_pit_stops(**args)
                case "get_stints":
                    return self._tool_get_stints(**args)
                case "get_qualifying_results":
                    return self._tool_get_qualifying_results(**args)
                case "get_race_results":
                    return self._tool_get_race_results(**args)
                case "get_driver_standings":
                    return self._tool_get_driver_standings(**args)
                case "get_constructor_standings":
                    return self._tool_get_constructor_standings(**args)
                case _:
                    return f"Unknown tool: {name}"
        except Exception as e:
            log.error("structured_retriever.tool_error", tool=name, error=str(e))
            return f"Tool '{name}' failed: {e}"

    # -----------------------------------------------------------------------
    # OpenF1 tool implementations
    # -----------------------------------------------------------------------

    def _tool_get_session(
        self, year: int, circuit_short_name: str, session_name: str = "Race"
    ) -> str:
        session = self._openf1.get_session(year, circuit_short_name, session_name)
        if session is None:
            return (
                f"No session found for '{circuit_short_name}' {year} ({session_name}). "
                f"Note: OpenF1 only has data from 2023 onwards."
            )
        return (
            f"Session found: {session.session_name} at {session.circuit_short_name}, "
            f"{session.country_name} {session.year}. "
            f"session_key={session.session_key}. "
            f"Date: {session.date_start}."
        )

    def _tool_get_lap_times(self, session_key: int, driver_number: int) -> str:
        laps = self._openf1.get_laps(session_key, driver_number)
        if not laps:
            return f"No lap data for driver #{driver_number} in session {session_key}."

        valid = [lap for lap in laps if lap.lap_duration is not None]
        if not valid:
            return f"Driver #{driver_number} completed {len(laps)} laps but none have recorded times."

        fastest = min(valid, key=lambda lap: lap.lap_duration)  # type: ignore[arg-type]
        lines = [f"Driver #{driver_number} — {len(laps)} laps, {len(valid)} timed."]
        lines.append(f"Fastest: Lap {fastest.lap_number} — {fastest.lap_duration:.3f}s")
        if fastest.duration_sector_1:
            lines.append(
                f"  S1={fastest.duration_sector_1:.3f}s  "
                f"S2={fastest.duration_sector_2:.3f}s  "  # type: ignore[arg-type]
                f"S3={fastest.duration_sector_3:.3f}s"  # type: ignore[arg-type]
            )
        lines.append("All laps:")
        for lap in laps:
            t = f"{lap.lap_duration:.3f}s" if lap.lap_duration else "no time"
            pit_flag = " [pit out]" if lap.is_pit_out_lap else ""
            lines.append(f"  Lap {lap.lap_number}: {t}{pit_flag}")
        return "\n".join(lines)

    def _tool_get_pit_stops(self, session_key: int, driver_number: int) -> str:
        stops = self._openf1.get_pit_stops(session_key, driver_number)
        if not stops:
            return f"No pit stop data for driver #{driver_number} in session {session_key}."
        lines = [f"Driver #{driver_number} — {len(stops)} pit stop(s):"]
        for stop in stops:
            dur = f"{stop.pit_duration:.1f}s" if stop.pit_duration else "duration not recorded"
            lines.append(f"  Lap {stop.lap_number}: {dur}")
        return "\n".join(lines)

    def _tool_get_stints(self, session_key: int, driver_number: int) -> str:
        stints = self._openf1.get_tyre_stints(session_key, driver_number)
        if not stints:
            return f"No stint data for driver #{driver_number} in session {session_key}."
        lines = [f"Driver #{driver_number} — {len(stints)} stint(s):"]
        for s in stints:
            end = str(s.lap_end) if s.lap_end else "ongoing"
            length = (s.lap_end - s.lap_start + 1) if s.lap_end else "?"
            age = f", {s.tyre_age_at_start} laps old at start" if s.tyre_age_at_start else ""
            lines.append(
                f"  Stint {s.stint_number}: {s.compound}, "
                f"laps {s.lap_start}–{end} ({length} laps{age})"
            )
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Jolpica tool implementations
    # -----------------------------------------------------------------------

    def _resolve_round(self, year: int, race_name: str) -> int | None:
        """Resolve a race name to a round number via the season schedule."""
        schedule = self._jolpica.get_schedule(year)
        name_lower = race_name.lower()
        for race in schedule:
            if name_lower in race.raceName.lower() or race.raceName.lower() in name_lower:
                return int(race.round)
        return None

    def _tool_get_qualifying_results(self, year: int, race_name: str) -> str:
        round_number = self._resolve_round(year, race_name)
        if round_number is None:
            return f"Could not find '{race_name}' in the {year} season calendar."

        result = self._jolpica.get_qualifying_results(year, round_number)
        if result is None:
            return f"No qualifying results found for {race_name} {year} (round {round_number})."

        lines = [f"{result.raceName} {result.season} Qualifying (Round {result.round}):"]
        for r in result.results:
            q1 = r.Q1 or "-"
            q2 = r.Q2 or "-"
            q3 = r.Q3 or "-"
            lines.append(
                f"  P{r.position}: {r.driver.givenName} {r.driver.familyName} "
                f"({r.constructor.name}) — Q1: {q1}  Q2: {q2}  Q3: {q3}"
            )
        return "\n".join(lines)

    def _tool_get_race_results(self, year: int, race_name: str) -> str:
        round_number = self._resolve_round(year, race_name)
        if round_number is None:
            return f"Could not find '{race_name}' in the {year} season calendar."

        result = self._jolpica.get_race_results(year, round_number)
        if result is None:
            return f"No results found for {race_name} {year} (round {round_number})."

        lines = [f"{result.raceName} {result.season} (Round {result.round}):"]
        for r in result.results:
            fl = " [FL]" if r.fastest_lap_rank == "1" else ""
            lines.append(
                f"  P{r.position}: {r.driver.givenName} {r.driver.familyName} "
                f"({r.constructor.name}) — {r.points}pts — "
                f"Grid: P{r.grid} — {r.status}{fl}"
            )
        return "\n".join(lines)

    def _tool_get_driver_standings(
        self, year: int, round_number: int | None = None
    ) -> str:
        standings = self._jolpica.get_driver_standings(year, round_number)
        if not standings:
            label = f"after round {round_number}" if round_number else "end of season"
            return f"No driver standings found for {year} {label}."
        label = f"after round {round_number}" if round_number else "(end of season)"
        lines = [f"{year} Driver Championship {label}:"]
        for s in standings:
            lines.append(
                f"  P{s.position}: {s.driver.givenName} {s.driver.familyName} "
                f"({s.constructor.name}) — {s.points}pts — {s.wins} wins"
            )
        return "\n".join(lines)

    def _tool_get_constructor_standings(
        self, year: int, round_number: int | None = None
    ) -> str:
        standings = self._jolpica.get_constructor_standings(year, round_number)
        if not standings:
            label = f"after round {round_number}" if round_number else "end of season"
            return f"No constructor standings found for {year} {label}."
        label = f"after round {round_number}" if round_number else "(end of season)"
        lines = [f"{year} Constructors' Championship {label}:"]
        for s in standings:
            lines.append(
                f"  P{s.position}: {s.constructor.name} — {s.points}pts — {s.wins} wins"
            )
        return "\n".join(lines)
