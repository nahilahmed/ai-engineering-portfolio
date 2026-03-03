"""
Document loader for Phase 1 — F1 Strategy RAG.

Responsibility: load raw text from a URL or local file and return a
normalised Document (clean text + metadata). No chunking, no embedding.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import structlog
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict

log = structlog.get_logger()

# HTML tags that add no informational value — strip before extracting text
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside"]


class Document(BaseModel):
    """Normalised output of the document loader.

    text     — clean, stripped content ready for chunking
    metadata — provenance: where it came from, what race/year it covers
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    metadata: dict[str, str]


class DocumentLoader:
    """Loads documents from URLs or local files into normalised Documents.

    Uses a persistent httpx.Client so TCP connections are reused across
    multiple load_from_url calls in the same session.
    """

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=30,
            headers={"User-Agent": "F1-RAG-Portfolio/1.0"},
            follow_redirects=True,
        )

    def load_from_url(self, url: str, metadata: dict[str, str]) -> Document:
        """Fetch a URL and return clean text as a Document."""
        log.info("document_loader.fetch", url=url)
        response = self._client.get(url)
        response.raise_for_status()
        text = self._extract_text(response.text)
        log.info("document_loader.fetched", url=url, chars=len(text))
        return Document(text=text, metadata={"source": url, **metadata})

    def load_from_file(self, path: str | Path, metadata: dict[str, str]) -> Document:
        """Read a local .html or .txt file and return clean text as a Document."""
        path = Path(path)
        log.info("document_loader.read_file", path=str(path))
        raw = path.read_text(encoding="utf-8")
        text = self._extract_text(raw) if path.suffix == ".html" else raw.strip()
        log.info("document_loader.read", path=str(path), chars=len(text))
        return Document(text=text, metadata={"source": str(path), **metadata})

    def _extract_text(self, html: str) -> str:
        """Strip HTML tags and return clean prose text.

        Tries progressively broader content containers:
        1. div.content-body  — FIA website article content
        2. <article>/<main>  — semantic HTML (most modern sites)
        3. <body>            — full page fallback
        """
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        container = (
            soup.find("div", class_="content-body")
            or soup.find("article")
            or soup.find("main")
            or soup.find("body")
            or soup
        )
        return " ".join(container.get_text(separator=" ", strip=True).split())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DocumentLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
