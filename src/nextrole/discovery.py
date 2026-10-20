"""Offline-first job discovery through the Perplexity Search API."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field, HttpUrl

from .config import Settings
from .models import JobPosting


class SearchResult(BaseModel):
    """A raw, minimally interpreted result returned by Perplexity."""

    title: str
    url: HttpUrl
    snippet: str = ""
    published_at: datetime | date | None = None
    last_updated: datetime | date | None = None


class DiscoveryItem(BaseModel):
    """A result and its optional safe normalization to a job posting."""

    result: SearchResult
    job: JobPosting | None = None
    note: str | None = None


class DiscoveryReport(BaseModel):
    """Results plus request and deduplication visibility."""

    items: list[DiscoveryItem] = Field(default_factory=list)
    requests_attempted: int = 0
    results_returned: int = 0
    duplicates_removed: int = 0
    mode: str = "fixture"


class SearchClient(Protocol):
    def search(self, queries: Sequence[str], **parameters: Any) -> Any:
        """Search a batch of no more than five queries."""


class SearchError(RuntimeError):
    """A search could not be completed or validated."""


class PerplexitySearchClient:
    """Thin adapter around the official Perplexity Python SDK."""

    def __init__(self, api_key: str) -> None:
        try:
            from perplexity import Perplexity
        except ImportError as error:  # pragma: no cover - packaging failure
            raise SearchError(
                "The Perplexity SDK is not installed. Run: "
                "python3 -m pip install -e '.[dev]'"
            ) from error
        self._client = Perplexity(api_key=api_key, max_retries=0)

    def search(self, queries: Sequence[str], **parameters: Any) -> Any:
        return self._client.search.create(query=list(queries), **parameters)


@dataclass(frozen=True)
class DiscoveryRequest:
    queries: tuple[str, ...]
    max_results: int = 20
    max_age_hours: int = 24
    country: str | None = None

    def batches(self) -> Iterable[tuple[str, ...]]:
        for start in range(0, len(self.queries), 5):
            yield self.queries[start : start + 5]

    def api_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {"max_results": self.max_results}
        if self.country:
            parameters["country"] = self.country
        if self.max_age_hours <= 1:
            parameters["search_recency_filter"] = "hour"
        elif self.max_age_hours <= 24:
            parameters["search_recency_filter"] = "day"
        elif self.max_age_hours <= 168:
            parameters["search_recency_filter"] = "week"
        elif self.max_age_hours <= 744:
            parameters["search_recency_filter"] = "month"
        else:
            parameters["search_recency_filter"] = "year"
        return parameters


def planned_attempts(request: DiscoveryRequest, settings: Settings) -> int:
    return len(tuple(request.batches())) * (1 + settings.max_search_retries)


def discover(
    request: DiscoveryRequest,
    settings: Settings,
    *,
    client: SearchClient | None = None,
    mode: str = "live",
    sleep: Callable[[float], None] = time.sleep,
) -> DiscoveryReport:
    """Run a bounded search and normalize results without inventing fields."""
    batches = tuple(request.batches())
    attempts_needed = len(batches) * (1 + settings.max_search_retries)
    if attempts_needed > settings.max_search_requests_per_run:
        raise SearchError(
            f"Planned attempts ({attempts_needed}) exceed the request budget "
            f"({settings.max_search_requests_per_run}). Reduce queries/retries "
            "or raise MAX_SEARCH_REQUESTS_PER_RUN explicitly."
        )
    if mode == "live" and client is None and not settings.perplexity_api_key:
        raise SearchError("PERPLEXITY_API_KEY is required for --live discovery.")
    if mode != "live":
        return (
            fixture_report(request) if mode == "fixture" else DiscoveryReport(mode=mode)
        )
    if client is None:
        client = PerplexitySearchClient(settings.perplexity_api_key.get_secret_value())

    report = DiscoveryReport(mode="live")
    seen: set[str] = set()
    for batch in batches:
        for retry in range(settings.max_search_retries + 1):
            report.requests_attempted += 1
            try:
                response = client.search(batch, **request.api_parameters())
                raw_results = _response_results(response)
                report.results_returned += len(raw_results)
                for raw in raw_results:
                    result = _parse_result(raw)
                    if result is None:
                        continue
                    key = normalize_url(str(result.url))
                    if key in seen:
                        report.duplicates_removed += 1
                        continue
                    seen.add(key)
                    report.items.append(_item(result, datetime.now(UTC)))
                break
            except Exception as error:  # SDK has multiple documented error types.
                if retry >= settings.max_search_retries:
                    raise SearchError(f"Perplexity search failed: {error}") from error
                delay = _retry_after(error) or 0.0
                sleep(delay)
    return report


def normalize_url(url: str) -> str:
    """Remove fragments and tracking query parameters for deduplication."""
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query)
        if not key.lower().startswith("utm_")
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(query),
            "",
        )
    )


def fixture_report(request: DiscoveryRequest) -> DiscoveryReport:
    """Return deterministic demo results without importing or calling an SDK."""
    result = SearchResult(
        title="Backend Engineer — Example Labs",
        url="https://example.com/careers/backend-engineer",
        snippet="Example Labs is hiring a backend engineer for its platform team.",
    )
    return DiscoveryReport(
        items=[_item(result, discovered_at=None)],
        results_returned=1,
        mode="fixture",
    )


def _response_results(response: Any) -> list[Any]:
    values = (
        response.get("results", [])
        if isinstance(response, dict)
        else getattr(response, "results", [])
    )
    if not isinstance(values, list):
        raise SearchError("Perplexity returned a malformed results field.")
    return values


def _parse_result(raw: Any) -> SearchResult | None:
    if not isinstance(raw, dict) and not hasattr(raw, "title"):
        return None
    get = (
        raw.get
        if isinstance(raw, dict)
        else lambda key, default=None: getattr(raw, key, default)
    )
    title, url = get("title"), get("url")
    if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
        return None
    try:
        return SearchResult(
            title=title.strip(),
            url=url,
            snippet=get("snippet") or "",
            published_at=get("date"),
            last_updated=get("last_updated"),
        )
    except Exception:
        return None


def _item(result: SearchResult, discovered_at: datetime | None = None) -> DiscoveryItem:
    if _is_obvious_search_page(str(result.url)):
        return DiscoveryItem(
            result=result, note="Skipped: likely a job-board search page."
        )
    text = f"{result.title} {result.snippet}".lower()
    job_words = (
        "engineer",
        "developer",
        "designer",
        "manager",
        "analyst",
        "scientist",
        "job",
        "hiring",
    )
    if not any(word in text for word in job_words):
        return DiscoveryItem(
            result=result,
            note="Uncertain: result was not identified as an individual job posting.",
        )
    job = JobPosting(
        title=result.title,
        url=result.url,
        description=result.snippet or None,
        source=urlsplit(str(result.url)).netloc,
        published_at=(
            result.published_at if isinstance(result.published_at, datetime) else None
        ),
        discovered_at=discovered_at,
    )
    return DiscoveryItem(result=result, job=job)


def _is_obvious_search_page(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(part in path.split("/") for part in ("search", "results")) or (
        "?q=" in url.lower()
    )


def _retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after") if headers else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None
