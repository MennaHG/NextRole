"""Fetch and locally extract job-page content without AI providers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import AnyUrl, TypeAdapter, ValidationError

from .models import JobPosting


class FetchError(RuntimeError):
    """A page could not be fetched or was not an acceptable HTML page."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    resolved_url: str
    html: str
    status_code: int
    content_type: str


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage:
        """Fetch one URL or raise a bounded, user-visible error."""


class HttpxPageFetcher:
    """Small, robots-aware HTTP adapter used by the enrichment service."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_bytes: int = 2_000_000,
        user_agent: str = "NextRole/0.1 (+job enrichment; respectful crawler)",
    ) -> None:
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_lock = Lock()

    def __enter__(self) -> HttpxPageFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def fetch(self, url: str) -> FetchedPage:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise FetchError("URL is not an HTTP(S) address")
        if not self._allowed_by_robots(url):
            raise FetchError("blocked by robots.txt")
        try:
            with self._client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise FetchError(f"HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").lower()
                if content_type and "html" not in content_type:
                    raise FetchError("response is not HTML")
                declared = response.headers.get("content-length")
                if declared and int(declared) > self.max_bytes:
                    raise FetchError("response exceeds the configured page limit")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise FetchError("response exceeds the configured page limit")
                    chunks.append(chunk)
                encoding = response.encoding or "utf-8"
                html = b"".join(chunks).decode(encoding, errors="replace")
                return FetchedPage(
                    url=url,
                    resolved_url=str(response.url),
                    html=html,
                    status_code=response.status_code,
                    content_type=content_type,
                )
        except httpx.HTTPError as error:
            raise FetchError(f"request failed: {error}") from error

    def _allowed_by_robots(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._robots_lock:
            if origin in self._robots:
                parser = self._robots[origin]
            else:
                parser = self._read_robots(origin)
                self._robots[origin] = parser
        return parser is None or parser.can_fetch(self.user_agent, url)

    def _read_robots(self, origin: str) -> RobotFileParser | None:
        try:
            response = self._client.get(urljoin(origin, "/robots.txt"))
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        parser = RobotFileParser()
        parser.set_url(urljoin(origin, "/robots.txt"))
        parser.parse(response.text.splitlines())
        return parser


@dataclass(frozen=True)
class ExtractionResult:
    description: str | None
    requirements: list[str]
    company: str | None
    location: str | None
    work_arrangement: str | None
    employment_type: str | None
    date_posted: datetime | None
    expiration_date: datetime | None
    title: str | None
    source: str
    validity_status: str
    error: str | None = None


def enrich_jobs(
    jobs: Sequence[JobPosting],
    *,
    fetcher: PageFetcher | None = None,
    max_pages: int = 50,
    concurrency: int = 4,
    dry_run: bool = False,
) -> list[JobPosting]:
    """Enrich jobs in input order, fetching each normalized URL at most once."""
    results = list(jobs)
    candidates: dict[str, list[int]] = {}
    for index, job in enumerate(results):
        if job.url:
            candidates.setdefault(_canonical_url(str(job.url)), []).append(index)
    selected = list(candidates.items())[:max_pages]
    if dry_run:
        for indices in candidates.values():
            for index in indices:
                results[index] = _with_status(
                    results[index], "skipped", "unavailable", "dry-run"
                )
        return results
    if fetcher is None:
        raise ValueError("A fetcher is required unless dry_run is enabled")
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_enrich_one, results[indices[0]], fetcher): key
            for key, indices in selected
        }
        completed: dict[str, JobPosting] = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                completed[key] = future.result()
            except Exception as error:  # keep one bad site from losing the run
                completed[key] = _with_status(
                    results[candidates[key][0]], "failed", "unavailable", str(error)
                )
    for key, indices in candidates.items():
        enriched = completed.get(key)
        if enriched is None:
            enriched = _with_status(
                results[indices[0]], "skipped", "unavailable", "page limit reached"
            )
        for index in indices:
            results[index] = enriched.model_copy(deep=True)
    return results


def load_job_input(path: Path) -> list[JobPosting]:
    """Load a list, discovery report, or ``{"jobs": [...]}`` payload."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read jobs JSON: {path}") from error
    if isinstance(payload, dict) and "jobs" in payload:
        payload = payload["jobs"]
    if isinstance(payload, dict) and "items" in payload:
        payload = [
            item.get("job")
            for item in payload["items"]
            if isinstance(item, dict) and item.get("job")
        ]
    if not isinstance(payload, list):
        raise ValueError("Jobs input must be a JSON list or discovery report")
    try:
        return [JobPosting.model_validate(item) for item in payload]
    except (TypeError, ValidationError) as error:
        raise ValueError(f"Jobs input contains invalid job data: {error}") from error


def write_jobs_json(path: Path, jobs: Iterable[JobPosting]) -> None:
    """Write deterministic JSON after serialization succeeds."""
    payload = (
        json.dumps(
            [job.model_dump(mode="json", exclude_none=True) for job in jobs],
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _enrich_one(job: JobPosting, fetcher: PageFetcher) -> JobPosting:
    try:
        page = fetcher.fetch(str(job.url))
    except FetchError as error:
        message = str(error)
        status = "blocked" if "blocked" in message else "failed"
        return _with_status(job, status, "unavailable", message)
    extracted = extract_job(page.html)
    updates: dict[str, Any] = {
        "resolved_url": TypeAdapter(AnyUrl).validate_python(page.resolved_url),
        "http_status": page.status_code,
        "extraction_status": "success" if extracted.description else "failed",
        "extraction_source": extracted.source,
        "validity_status": extracted.validity_status,
        "extraction_error": extracted.error,
    }
    for key in (
        "description",
        "requirements",
        "company",
        "location",
        "work_arrangement",
        "employment_type",
        "date_posted",
        "expiration_date",
    ):
        value = getattr(extracted, key)
        if value is not None and value != []:
            updates[key] = value
    if extracted.title and job.title == "":
        updates["title"] = extracted.title
    if extracted.date_posted and job.published_at is None:
        updates["published_at"] = extracted.date_posted
    return job.model_copy(update=updates)


def extract_job(html: str) -> ExtractionResult:
    """Extract schema.org data first, then a bounded HTML job-content region."""
    soup = BeautifulSoup(html, "html.parser")
    structured = _json_ld_posting(soup)
    if structured:
        result = _from_json_ld(structured)
        if result.description:
            return result
    text, requirements = _html_content(soup)
    if not text:
        return ExtractionResult(
            None,
            [],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "unavailable",
            "uncertain",
            "no job description found",
        )
    title = _text(soup.title) if soup.title else None
    invalid = _looks_like_non_job_page(soup, text)
    status = (
        "invalid"
        if invalid
        else ("valid" if len(text) >= 100 and requirements else "uncertain")
    )
    return ExtractionResult(
        text, requirements, None, None, None, None, None, None, title, "html", status
    )


def _json_ld_posting(soup: BeautifulSoup) -> dict[str, Any] | None:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or script.get_text())
        except (TypeError, json.JSONDecodeError):
            continue
        for item in _walk_json(data):
            types = item.get("@type", [])
            types = types if isinstance(types, list) else [types]
            if any(str(value).casefold() == "jobposting" for value in types):
                return item
    return None


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _from_json_ld(data: dict[str, Any]) -> ExtractionResult:
    description = _clean_html(data.get("description"))
    requirements = _as_text_list(data.get("qualifications")) + _as_text_list(
        data.get("responsibilities")
    )
    requirements += _as_text_list(data.get("skills")) + _as_text_list(
        data.get("experienceRequirements")
    )
    company = data.get("hiringOrganization")
    company = company.get("name") if isinstance(company, dict) else company
    location = data.get("jobLocation")
    location = _location_text(location)
    remote = data.get("jobLocationType") or data.get("applicantLocationRequirements")
    work = str(remote) if remote else None
    return ExtractionResult(
        description,
        _unique(requirements),
        _as_string(company),
        location,
        work,
        _as_string(data.get("employmentType")),
        _parse_date(data.get("datePosted")),
        _parse_date(data.get("validThrough")),
        _as_string(data.get("title")),
        "json_ld",
        "valid" if description and _as_string(data.get("title")) else "uncertain",
    )


def _html_content(soup: BeautifulSoup) -> tuple[str | None, list[str]]:
    for node in soup.select(
        "script, style, nav, footer, header, aside, form, iframe, noscript, "
        ".advertisement, .ads"
    ):
        node.decompose()
    candidates = soup.select(
        "[itemprop='description'], [class*='job-description'], "
        "[id*='job-description'], main, article"
    )
    node = max(
        candidates,
        key=lambda candidate: len(candidate.get_text(" ", strip=True)),
        default=None,
    )
    if node is None:
        return None, []
    parts: list[str] = []
    requirements: list[str] = []
    in_requirements = False
    for child in node.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        value = child.get_text(" ", strip=True)
        if not value or value in parts:
            continue
        parts.append(value)
        if child.name.startswith("h"):
            in_requirements = bool(
                re.search(r"qualif|require|skill|experience|responsib", value, re.I)
            )
        elif in_requirements:
            requirements.append(value)
    text = "\n".join(parts).strip()
    return (text or None), _unique(requirements)


def _looks_like_non_job_page(soup: BeautifulSoup, text: str) -> bool:
    lower = f"{soup.get_text(' ', strip=True)} {text}".casefold()
    signals = (
        "candidate profile",
        "subscribe to our newsletter",
        "job search results",
        "jobs in egypt",
    )
    return sum(signal in lower for signal in signals) >= 1 and len(text) < 1200


def _with_status(job: JobPosting, status: str, source: str, error: str) -> JobPosting:
    return job.model_copy(
        update={
            "extraction_status": status,
            "extraction_source": source,
            "validity_status": "uncertain",
            "extraction_error": error,
        }
    )


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return f"{parts.scheme.casefold()}://{parts.netloc.casefold()}{parts.path.rstrip('/')}?{parts.query}".rstrip(
        "?"
    )


def _text(value: Tag | None) -> str:
    return value.get_text(" ", strip=True) if value else ""


def _clean_html(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return text or None


def _as_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique([item for item in (_as_string(item) for item in value) if item])
    text = _clean_html(value)
    return [text] if text else []


def _location_text(value: Any) -> str | None:
    if isinstance(value, list):
        values = [_location_text(item) for item in value]
        return "; ".join(item for item in values if item) or None
    if isinstance(value, dict):
        address = value.get("address", value)
        if isinstance(address, dict):
            return ", ".join(
                str(address[key])
                for key in (
                    "streetAddress",
                    "addressLocality",
                    "addressRegion",
                    "addressCountry",
                )
                if address.get(key)
            )
        return _as_string(address)
    return _as_string(value)


def _parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))
