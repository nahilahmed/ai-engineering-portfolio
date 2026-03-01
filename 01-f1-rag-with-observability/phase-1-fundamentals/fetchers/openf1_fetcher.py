"""
OpenF1 API fetcher — structured race telemetry data.

API base: https://api.openf1.org/v1/
Docs:     https://openf1.org/
Auth:     none required
"""

from __future__ import annotations

import httpx
import structlog
from pydantic import BaseModel, ConfigDict

log = structlog.get_logger()

_BASE_URL = "https://api.openf1.org/v1"


class F1BaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Session(F1BaseModel):
    session_key: int          # the ID everything else depends on
    meeting_key: int          # the race weekend this session belongs to
    session_name: str         # "Race", "Qualifying", "Sprint", "Practice 1"
    session_type: str         # "Race", "Qualifying", "Practice"
    date_start: str           # kept as str — timezone parsing is a problem for later
    year: int
    country_name: str         # "Monaco", "Bahrain"
    circuit_short_name: str   # "Monte Carlo", "Sakhir" — used for filtering


class Lap(F1BaseModel):
    session_key: int
    driver_number: int
    lap_number: int
    lap_duration: float | None = None         # None on SC laps, in-laps, DNF laps
    duration_sector_1: float | None = None
    duration_sector_2: float | None = None
    duration_sector_3: float | None = None
    is_pit_out_lap: bool | None = None        # True on the lap a driver exits the pits
    date_start: str | None = None


class PitStop(F1BaseModel):
    session_key: int
    driver_number: int
    lap_number: int
    pit_duration: float | None = None   # stationary time in seconds; timing system occasionally misses this
    date: str | None = None


class Stint(F1BaseModel):
    session_key: int
    driver_number: int
    stint_number: int
    lap_start: int
    lap_end: int | None = None            # None if stint was ongoing at session end
    compound: str                         # "SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"
    tyre_age_at_start: int | None = None  # laps of wear on the tyre when this stint began


class OpenF1Fetcher:

    def __init__(self) -> None:
        self._client = httpx.Client(base_url=_BASE_URL, timeout=30.0)
        self._log = log.bind(fetcher="openf1")

    def _get(self, endpoint: str, params: dict) -> list[dict]:
        self._log.info("api_request", endpoint=endpoint, params=params)
        response = self._client.get(endpoint, params=params)

        if response.status_code == 404:
            self._log.info("api_empty_result", endpoint=endpoint, params=params)
            return []

        response.raise_for_status()
        data = response.json()
        self._log.info("api_response", endpoint=endpoint, record_count=len(data))
        return data

    def get_session(
        self,
        year: int,
        circuit_short_name: str,
        session_name: str = "Race",
    ) -> Session | None:
        results = self._get(
            "/sessions",
            params={"year": year, "circuit_short_name": circuit_short_name, "session_name": session_name},
        )
        if not results:
            self._log.warning("session_not_found", year=year, circuit=circuit_short_name)
            return None
        return Session.model_validate(results[0])

    def get_laps(self, session_key: int, driver_number: int | None = None) -> list[Lap]:
        params: dict = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return [Lap.model_validate(row) for row in self._get("/laps", params=params)]

    def get_pit_stops(self, session_key: int, driver_number: int | None = None) -> list[PitStop]:
        params: dict = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return [PitStop.model_validate(row) for row in self._get("/pit", params=params)]

    def get_tyre_stints(self, session_key: int, driver_number: int | None = None) -> list[Stint]:
        params: dict = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return [Stint.model_validate(row) for row in self._get("/stints", params=params)]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenF1Fetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
