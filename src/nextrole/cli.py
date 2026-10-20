"""Command-line interface for the NextRole."""

import argparse
from collections.abc import Sequence

from .config import get_settings
from .discovery import DiscoveryRequest, SearchError, discover, planned_attempts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nextrole",
        description="A configurable job-discovery application.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="Validate basic local configuration.")
    discover_parser = subparsers.add_parser(
        "discover", help="Discover recent job postings (offline by default)."
    )
    discover_parser.add_argument(
        "--query",
        action="append",
        help="Search query; repeated values are batched in groups of five.",
    )
    discover_parser.add_argument(
        "--hours", type=int, help="Maximum posting age in hours."
    )
    discover_parser.add_argument(
        "--limit", type=int, help="Maximum results per request (1-20)."
    )
    discover_parser.add_argument(
        "--country", help="ISO country code used by Search API."
    )
    mode = discover_parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live", action="store_true", help="Make the explicitly requested API call."
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="Print the plan without making a call."
    )
    return parser


def _run_check() -> int:
    settings = get_settings()
    configured = [
        label
        for label, value in (
            ("LLM provider", settings.llm_provider),
            ("embedding provider", settings.embedding_provider),
            ("profile path", settings.profile_path),
        )
        if value is not None
    ]
    if configured:
        print("Configuration is valid. Configured: " + ", ".join(configured) + ".")
    else:
        print(
            "Configuration is valid. No optional providers or profile are configured."
        )
    return 0


def _run_discover(args: argparse.Namespace) -> int:
    settings = get_settings()
    queries = tuple(args.query or settings.search_queries)
    if not queries:
        queries = ("recent job openings",)
    try:
        request = DiscoveryRequest(
            queries=queries,
            max_results=(
                args.limit if args.limit is not None else settings.search_max_results
            ),
            max_age_hours=(
                args.hours if args.hours is not None else settings.search_max_age_hours
            ),
            country=args.country or settings.search_country,
        )
        attempts = planned_attempts(request, settings)
        if args.dry_run:
            print("Mode: dry-run (no API call)")
            print(f"Queries: {list(request.queries)}")
            print(f"API parameters: {request.api_parameters()}")
            print(f"Planned request attempts: {attempts}")
            print(f"Request budget: {settings.max_search_requests_per_run}")
            discover(request, settings, mode="dry-run")
            return 0
        report = discover(
            request,
            settings,
            mode="live" if args.live else "fixture",
        )
    except (SearchError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Mode: {report.mode}")
    print(f"Requests attempted: {report.requests_attempted}")
    print(f"Results returned: {report.results_returned}")
    print(f"Duplicates removed: {report.duplicates_removed}")
    for item in report.items:
        print(f"- {item.result.title}")
        print(f"  URL: {item.result.url}")
        if item.job:
            print(f"  Company: {item.job.company or 'unknown'}")
            print(f"  Location: {item.job.location or 'unknown'}")
            print(f"  Published: {item.job.published_at or 'unknown'}")
        print(f"  Last updated: {item.result.last_updated or 'unknown'}")
        if item.note:
            print(f"  Note: {item.note}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _run_check()
    if args.command == "discover":
        return _run_discover(args)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
