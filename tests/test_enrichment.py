import json
from pathlib import Path

from nextrole.enrichment import (
    FetchedPage,
    enrich_jobs,
    extract_job,
    load_job_input,
    write_jobs_json,
)
from nextrole.models import JobPosting

JSON_LD = """
<html><head><script type="application/ld+json">
{"@context":"https://schema.org","@graph":[
 {"@type":"BreadcrumbList","itemListElement":[]},
 {"@type":"JobPosting","title":"Platform Engineer",
  "description":"<p>Build reliable APIs.</p>",
  "hiringOrganization":{"@type":"Organization","name":"Example Labs"},
  "jobLocation":{"address":{"addressLocality":"Cairo","addressCountry":"EG"}},
  "jobLocationType":"TELECOMMUTE","employmentType":"FULL_TIME",
  "datePosted":"2026-10-20","validThrough":"2026-11-20",
  "qualifications":"Python and five years of experience."}
]}</script></head><body><main>ignored fallback</main></body></html>
"""

HTML_ONLY = """
<html><body><nav>Menu</nav><main><h1>Data Engineer</h1>
<h2>About the role</h2><p>Build data pipelines for our analytics platform.</p>
<h2>Requirements</h2><ul><li>Python and SQL</li><li>Cloud experience</li></ul>
<footer>Related jobs</footer></main></body></html>
"""


class FakeFetcher:
    def __init__(self, pages: dict[str, FetchedPage]):
        self.pages = pages
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        return self.pages[url]


def page(url: str, html: str) -> FetchedPage:
    return FetchedPage(url, url + "?resolved=1", html, 200, "text/html")


def test_json_ld_graph_extracts_job_metadata() -> None:
    result = extract_job(JSON_LD)

    assert result.source == "json_ld"
    assert result.validity_status == "valid"
    assert result.company == "Example Labs"
    assert result.location == "Cairo, EG"
    assert "Python" in result.requirements[0]
    assert result.work_arrangement == "TELECOMMUTE"


def test_html_fallback_preserves_headings_and_bullets() -> None:
    result = extract_job(HTML_ONLY)

    assert result.source == "html"
    assert result.validity_status == "valid"
    assert "Requirements" in (result.description or "")
    assert result.requirements == ["Python and SQL", "Cloud experience"]
    assert "Menu" not in (result.description or "")


def test_malformed_json_ld_and_non_job_pages_are_uncertain_or_invalid() -> None:
    malformed = extract_job(
        "<html><script type='application/ld+json'>{bad</script></html>"
    )
    category = extract_job(
        "<html><body><main><h1>Software Engineering Jobs in Egypt</h1>"
        "<p>Search results and jobs in Egypt. Filter jobs by category.</p>"
        "</main></body></html>"
    )

    assert malformed.source == "unavailable"
    assert malformed.validity_status == "uncertain"
    assert category.validity_status == "invalid"


def test_enrichment_deduplicates_limits_preserves_failures_and_reloads(
    tmp_path: Path,
) -> None:
    first = JobPosting(title="First", url="https://jobs.test/a")
    duplicate = JobPosting(title="Duplicate", url="https://JOBS.test/a/")
    limited = JobPosting(title="Limited", url="https://jobs.test/b")
    fetcher = FakeFetcher({str(first.url): page(str(first.url), HTML_ONLY)})

    enriched = enrich_jobs(
        [first, duplicate, limited], fetcher=fetcher, max_pages=1, concurrency=2
    )
    output = tmp_path / "enriched_jobs.json"
    write_jobs_json(output, enriched)
    reloaded = load_job_input(output)

    assert fetcher.calls == [str(first.url)]
    assert len(reloaded) == 3
    assert reloaded[0].extraction_status == "success"
    assert reloaded[1].resolved_url == reloaded[0].resolved_url
    assert reloaded[2].extraction_error == "page limit reached"
    assert json.loads(output.read_text(encoding="utf-8"))[0]["description"]


def test_dry_run_makes_no_fetch_calls() -> None:
    fetcher = FakeFetcher({})
    jobs = enrich_jobs(
        [JobPosting(title="Engineer", url="https://jobs.test/a")],
        fetcher=fetcher,
        dry_run=True,
    )

    assert fetcher.calls == []
    assert jobs[0].extraction_status == "skipped"
