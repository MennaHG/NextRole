from types import SimpleNamespace

import pytest

from nextrole.config import Settings
from nextrole.discovery import DiscoveryRequest, SearchError, discover, normalize_url


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def search(self, queries, **parameters):
        self.calls.append((queries, parameters))
        return next(self.responses)


def settings(**values):
    return Settings(_env_file=None, **values)


def test_search_normalizes_and_deduplicates_results():
    client = FakeClient(
        [
            SimpleNamespace(
                results=[
                    {
                        "title": "Backend Engineer",
                        "url": "https://EXAMPLE.com/job/?utm_source=x",
                        "snippet": "Hiring engineer",
                    },
                    {
                        "title": "Same",
                        "url": "https://example.com/job/",
                        "snippet": "Hiring engineer",
                    },
                ]
            )
        ]
    )
    report = discover(DiscoveryRequest(queries=("backend",)), settings(), client=client)
    assert len(report.items) == 1
    assert report.duplicates_removed == 1
    assert report.items[0].job is not None


def test_queries_are_batched_in_groups_of_five():
    queries = tuple(f"query-{number}" for number in range(6))
    client = FakeClient([SimpleNamespace(results=[]), SimpleNamespace(results=[])])
    report = discover(
        DiscoveryRequest(queries=queries),
        settings(max_search_requests_per_run=2),
        client=client,
    )
    assert [call[0] for call in client.calls] == [queries[:5], queries[5:]]
    assert report.requests_attempted == 2


def test_budget_includes_retries():
    with pytest.raises(SearchError, match="exceed the request budget"):
        discover(
            DiscoveryRequest(queries=tuple(f"query-{number}" for number in range(6))),
            settings(max_search_requests_per_run=2, max_search_retries=1),
            client=FakeClient([]),
        )


def test_dry_run_and_fixture_do_not_call_client():
    client = FakeClient([])
    request = DiscoveryRequest(queries=("backend",))
    assert (
        discover(request, settings(), client=client, mode="dry-run").mode == "dry-run"
    )
    assert (
        discover(request, settings(), client=client, mode="fixture").mode == "fixture"
    )
    assert client.calls == []


def test_live_requires_a_key():
    with pytest.raises(SearchError, match="PERPLEXITY_API_KEY"):
        discover(DiscoveryRequest(queries=("backend",)), settings())


def test_ambiguous_and_search_page_results_are_not_promoted():
    client = FakeClient(
        [
            {
                "results": [
                    {
                        "title": "Company homepage",
                        "url": "https://example.com",
                        "snippet": "About us",
                    },
                    {
                        "title": "Jobs search",
                        "url": "https://example.com/search?q=jobs",
                        "snippet": "jobs",
                    },
                ]
            }
        ]
    )
    report = discover(DiscoveryRequest(queries=("jobs",)), settings(), client=client)
    assert all(item.job is None for item in report.items)


def test_retry_and_retry_after_are_bounded():
    class RateLimited:
        response = SimpleNamespace(headers={"retry-after": "0"})

    class RetryClient:
        def __init__(self):
            self.calls = 0

        def search(self, queries, **parameters):
            self.calls += 1
            if self.calls == 1:
                raise RateLimited()
            return {"results": []}

    client = RetryClient()
    report = discover(
        DiscoveryRequest(queries=("backend",)),
        settings(max_search_retries=1, max_search_requests_per_run=2),
        client=client,
        sleep=lambda _: None,
    )
    assert report.requests_attempted == 2


def test_normalize_url_removes_tracking_and_fragment():
    assert (
        normalize_url("HTTPS://Example.com/job/?utm_campaign=x#details")
        == "https://example.com/job"
    )
