"""
Jolpica/Ergast API fetcher — historical F1 results and standings.

API base: https://api.jolpi.ca/ergast/f1/
Docs: https://github.com/jolpica/jolpica-f1
Auth: none required — fully public

Why Jolpica and not ergast.com?
    The original Ergast API (ergast.com) was deprecated after the 2024 season.
    Jolpica is the community-maintained fork with identical URL structure and
    JSON schema — drop-in replacement.

Data flow:
    All endpoints return a nested MRData envelope:
        { "MRData": { "RaceTable": { "Races": [...] } } }

    This fetcher unwraps that envelope and returns flat lists of typed models.
    The nesting is an Ergast legacy design — Jolpica preserves it for compatibility.
"""

from __future__ import annotations

import httpx
import structlog
from pydantic import BaseModel, ConfigDict

log = structlog.get_logger()

_BASE_URL = "https://api.jolpi.ca/ergast/f1"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Driver(BaseModel):
    """Minimal driver identity — enough for attribution in RAG chunks."""

    model_config = ConfigDict(extra="ignore")

    driverId: str           # e.g. "leclerc", "verstappen"
    code: str | None = None # 3-letter code, e.g. "LEC" — absent for older drivers
    givenName: str
    familyName: str
    permanentNumber: str | None = None


class Constructor(BaseModel):
    model_config = ConfigDict(extra="ignore")

    constructorId: str      # e.g. "ferrari", "red_bull"
    name: str               # e.g. "Ferrari", "Red Bull"


class RaceResult(BaseModel):
    """
    One driver's result in a race.

    Note on typing: Ergast returns numbers as strings (position, points, grid).
    They're kept as str here to avoid coercion surprises — convert at call site
    when arithmetic is needed.
    """

    model_config = ConfigDict(extra="ignore")

    number: str             # car number
    position: str           # finishing position (string per Ergast schema)
    positionText: str       # "1", "2", "R" (retired), "D" (disqualified)
    points: str             # championship points awarded
    grid: str               # starting grid position
    laps: str               # laps completed
    status: str             # "Finished", "+1 Lap", "Engine", etc.
    driver: Driver
    constructor: Constructor
    fastest_lap_rank: str | None = None   # "1" if this driver set fastest lap


class RaceResultsResponse(BaseModel):
    """Wraps a full race with its results list."""

    model_config = ConfigDict(extra="ignore")

    season: str
    round: str
    raceName: str           # e.g. "Monaco Grand Prix"
    date: str               # race date, ISO format
    results: list[RaceResult]


class ConstructorStanding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position: str
    points: str
    wins: str
    constructor: Constructor


class DriverStanding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position: str
    points: str
    wins: str
    driver: Driver
    constructor: Constructor   # the constructor at time of standing


class QualifyingResult(BaseModel):
    """One driver's qualifying result — position and Q1/Q2/Q3 times."""

    model_config = ConfigDict(extra="ignore")

    number: str
    position: str
    driver: Driver
    constructor: Constructor
    Q1: str | None = None   # not all drivers reach Q2/Q3
    Q2: str | None = None
    Q3: str | None = None


class QualifyingResponse(BaseModel):
    """Wraps a qualifying session with its full results list."""

    model_config = ConfigDict(extra="ignore")

    season: str
    round: str
    raceName: str
    results: list[QualifyingResult]


class ScheduleRace(BaseModel):
    """One race entry from the season calendar — used for race name → round lookup."""

    model_config = ConfigDict(extra="ignore")

    round: str       # "1", "2", ... as string per Ergast schema
    raceName: str    # "Bahrain Grand Prix", "Monaco Grand Prix"
    date: str        # race date, ISO format


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class JolpicaFetcher:
    """
    Thin wrapper around the Jolpica/Ergast REST API.

    Usage:
        with JolpicaFetcher() as f:
            results   = f.get_race_results(year=2024, round_number=8)
            standings = f.get_constructor_standings(year=2024)
    """

    def __init__(self) -> None:
        self._client = httpx.Client(base_url=_BASE_URL, timeout=30.0)
        self._log = log.bind(fetcher="jolpica")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        """
        GET {path}, return the full parsed JSON dict.
        Path should not include the base URL, e.g. "/2024/8/results.json"
        """
        self._log.info("api_request", path=path)
        response = self._client.get(path)
        response.raise_for_status()
        data = response.json()
        self._log.info("api_response", path=path)
        return data

    def _unwrap_races(self, data: dict) -> list[dict]:
        """
        Navigate the MRData envelope and return the Races list.
        Returns [] if the path doesn't exist (e.g. no results for that round).
        """
        try:
            return data["MRData"]["RaceTable"]["Races"]
        except KeyError:
            return []

    def _unwrap_standings(self, data: dict) -> list[dict]:
        """Navigate the MRData envelope and return the StandingsLists list."""
        try:
            return data["MRData"]["StandingsTable"]["StandingsLists"]
        except KeyError:
            return []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_race_results(self, year: int, round_number: int) -> RaceResultsResponse | None:
        """
        Fetch the full results for one race.

        Args:
            year: e.g. 2024
            round_number: 1-indexed round in the season calendar

        Returns:
            RaceResultsResponse if the race exists, None otherwise.

        Example:
            get_race_results(2024, 8)  # Monaco 2024
        """
        data = self._get(f"/{year}/{round_number}/results.json")
        races = self._unwrap_races(data)

        if not races:
            self._log.warning("race_not_found", year=year, round=round_number)
            return None

        race = races[0]

        # Jolpica nests results inside each Race object — flatten into our model
        return RaceResultsResponse(
            season=race["season"],
            round=race["round"],
            raceName=race["raceName"],
            date=race["date"],
            results=[
                RaceResult(
                    number=r["number"],
                    position=r["position"],
                    positionText=r["positionText"],
                    points=r["points"],
                    grid=r["grid"],
                    laps=r["laps"],
                    status=r["status"],
                    driver=Driver.model_validate(r["Driver"]),
                    constructor=Constructor.model_validate(r["Constructor"]),
                    fastest_lap_rank=r.get("FastestLap", {}).get("rank"),
                )
                for r in race.get("Results", [])
            ],
        )

    def get_constructor_standings(
        self, year: int, round_number: int | None = None
    ) -> list[ConstructorStanding]:
        """
        Fetch constructor championship standings.

        Args:
            year: season year
            round_number: if provided, standings after that round;
                          if omitted, returns end-of-season standings

        Returns:
            List of ConstructorStanding ordered by position (1st = index 0).
        """
        path = f"/{year}/constructorStandings.json"
        if round_number is not None:
            path = f"/{year}/{round_number}/constructorStandings.json"

        data = self._get(path)
        lists = self._unwrap_standings(data)

        if not lists:
            self._log.warning("standings_not_found", year=year, round=round_number)
            return []

        return [
            ConstructorStanding(
                position=s["position"],
                points=s["points"],
                wins=s["wins"],
                constructor=Constructor.model_validate(s["Constructor"]),
            )
            for s in lists[0].get("ConstructorStandings", [])
        ]

    def get_driver_standings(
        self, year: int, round_number: int | None = None
    ) -> list[DriverStanding]:
        """
        Fetch driver championship standings.

        Args:
            year: season year
            round_number: if provided, standings after that round

        Returns:
            List of DriverStanding ordered by position (1st = index 0).
        """
        path = f"/{year}/driverStandings.json"
        if round_number is not None:
            path = f"/{year}/{round_number}/driverStandings.json"

        data = self._get(path)
        lists = self._unwrap_standings(data)

        if not lists:
            self._log.warning("standings_not_found", year=year, round=round_number)
            return []

        return [
            DriverStanding(
                position=s["position"],
                points=s["points"],
                wins=s["wins"],
                driver=Driver.model_validate(s["Driver"]),
                # Ergast puts the constructor inside each driver standing entry
                constructor=Constructor.model_validate(s["Constructors"][0]),
            )
            for s in lists[0].get("DriverStandings", [])
        ]

    def get_qualifying_results(self, year: int, round_number: int) -> QualifyingResponse | None:
        """
        Fetch qualifying results for one race weekend.

        Args:
            year: season year
            round_number: 1-indexed round in the season calendar

        Returns:
            QualifyingResponse if data exists, None otherwise.
            P1 in results = pole position.
        """
        data = self._get(f"/{year}/{round_number}/qualifying.json")
        races = self._unwrap_races(data)

        if not races:
            self._log.warning("qualifying_not_found", year=year, round=round_number)
            return None

        race = races[0]
        return QualifyingResponse(
            season=race["season"],
            round=race["round"],
            raceName=race["raceName"],
            results=[
                QualifyingResult(
                    number=r["number"],
                    position=r["position"],
                    driver=Driver.model_validate(r["Driver"]),
                    constructor=Constructor.model_validate(r["Constructor"]),
                    Q1=r.get("Q1"),
                    Q2=r.get("Q2"),
                    Q3=r.get("Q3"),
                )
                for r in race.get("QualifyingResults", [])
            ],
        )

    def get_schedule(self, year: int) -> list[ScheduleRace]:
        """
        Fetch the full race calendar for a season.

        Used internally to resolve a race name (e.g. "Singapore") to a round
        number before calling get_race_results. Not exposed as a standalone tool.

        Args:
            year: season year

        Returns:
            List of ScheduleRace ordered by round number.
        """
        data = self._get(f"/{year}.json")
        races = self._unwrap_races(data)
        return [ScheduleRace.model_validate(r) for r in races]

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JolpicaFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
