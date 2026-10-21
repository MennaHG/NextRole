"""Command-line interface for the NextRole."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import get_settings
from .cv_generation import CVGenerationError, generate_cv_from_profile, save_cv
from .digest import (
    DigestError,
    build_digest,
    load_delivery_state,
    load_matching_report,
    mark_digest_sent,
    save_delivery_state,
    send_gmail,
)
from .discovery import DiscoveryRequest, SearchError, discover, planned_attempts
from .enrichment import HttpxPageFetcher, enrich_jobs, load_job_input, write_jobs_json
from .matching import (
    MatchingError,
    create_live_models,
    load_jobs_json,
    match_jobs,
)
from .profile import ProfileError, generate_queries, load_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nextrole",
        description="A configurable job-discovery application.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="Validate basic local configuration.")
    profile_parser = subparsers.add_parser(
        "profile", help="Manage a local candidate profile."
    )
    profile_commands = profile_parser.add_subparsers(dest="profile_command")
    validate_parser = profile_commands.add_parser(
        "validate", help="Validate a Markdown candidate profile offline."
    )
    validate_parser.add_argument(
        "--file", required=True, help="Path to profile Markdown."
    )
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
    discover_parser.add_argument(
        "--profile", help="Use this Markdown profile to generate search queries."
    )
    mode = discover_parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live", action="store_true", help="Make the explicitly requested API call."
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="Print the plan without making a call."
    )
    match_parser = subparsers.add_parser(
        "match", help="Match normalized jobs against a candidate profile."
    )
    match_parser.add_argument(
        "--profile", required=True, help="Path to profile Markdown."
    )
    match_parser.add_argument(
        "--input", required=True, help="Path to normalized jobs JSON."
    )
    match_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum jobs to evaluate (default: configured limit).",
    )
    match_parser.add_argument(
        "--json", action="store_true", help="Print the matching report as JSON."
    )
    match_parser.add_argument(
        "--output",
        help="Write the matching report as formatted JSON to this path.",
    )
    match_parser.add_argument(
        "--debug-evidence",
        action="store_true",
        help="Include retrieved chunks and requirement assessments in output.",
    )
    match_mode = match_parser.add_mutually_exclusive_group()
    match_mode.add_argument(
        "--live", action="store_true", help="Allow paid provider calls."
    )
    match_mode.add_argument(
        "--dry-run", action="store_true", help="Show the plan only."
    )
    enrich_parser = subparsers.add_parser(
        "enrich", help="Fetch and locally extract job pages (no AI calls)."
    )
    enrich_parser.add_argument("--input", required=True, help="Input jobs JSON.")
    enrich_parser.add_argument("--output", required=True, help="Output enriched JSON.")
    enrich_parser.add_argument(
        "--timeout", type=float, help="Per-page timeout in seconds."
    )
    enrich_parser.add_argument(
        "--max-pages", type=int, help="Maximum unique pages to fetch."
    )
    enrich_parser.add_argument(
        "--concurrency", type=int, help="Maximum concurrent fetches."
    )
    enrich_parser.add_argument(
        "--max-bytes", type=int, help="Maximum HTML response size."
    )
    enrich_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write statuses without network requests.",
    )
    digest_parser = subparsers.add_parser(
        "digest", help="Preview or deliver a digest from matching results."
    )
    digest_parser.add_argument(
        "--input", required=True, help="Matching report JSON from `nextrole match`."
    )
    digest_parser.add_argument(
        "--preview", action="store_true", help="Render without connecting to Gmail."
    )
    digest_parser.add_argument(
        "--send", action="store_true", help="Submit the digest through Gmail SMTP."
    )
    digest_parser.add_argument(
        "--output", help="Optional path for the rendered HTML preview."
    )
    digest_parser.add_argument(
        "--state", help="Path to the persistent sent-job state JSON."
    )
    digest_parser.add_argument("--min-score", type=float)
    digest_parser.add_argument("--max-jobs", type=int)
    cv_parser = subparsers.add_parser(
        "generate-cv", help="Generate an evidence-grounded tailored CV from a job URL."
    )
    cv_parser.add_argument("--url", required=True, help="Job posting URL.")
    cv_parser.add_argument("--profile", help="Candidate profile Markdown path.")
    cv_parser.add_argument(
        "--output-root", default="outputs/cvs", help="Directory for generated CV JSON."
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
    try:
        profile_path = Path(args.profile) if args.profile else settings.profile_path
        profile = load_profile(profile_path) if profile_path else None
        if args.query:
            queries = tuple(args.query)
            query_source = "manual"
        elif profile:
            queries = tuple(generate_queries(profile))
            query_source = "profile"
        else:
            queries = tuple(settings.search_queries) or ("recent job openings",)
            query_source = "configuration" if settings.search_queries else "default"
        if not queries:
            raise ProfileError("no search queries were generated")
        request = DiscoveryRequest(
            queries=queries,
            max_results=(
                args.limit if args.limit is not None else settings.search_max_results
            ),
            max_age_hours=(
                args.hours
                if args.hours is not None
                else (
                    profile.preferences.max_age_hours
                    if profile and profile.preferences.max_age_hours is not None
                    else settings.search_max_age_hours
                )
            ),
            country=args.country or settings.search_country,
        )
        attempts = planned_attempts(request, settings)
        if args.dry_run:
            print("Mode: dry-run (no API call)")
            print(f"Query source: {query_source}")
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
    except (ProfileError, SearchError, ValueError) as error:
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


def _run_profile_validate(args: argparse.Namespace) -> int:
    try:
        profile = load_profile(Path(args.file))
    except ProfileError as error:
        print(f"Error: {error}")
        return 2
    print("Profile is valid.")
    print(f"Roles: {len(profile.preferences.roles)}")
    print(f"Locations: {len(profile.preferences.locations)}")
    print(f"Generated queries: {len(generate_queries(profile))}")
    return 0


def _run_match(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        profile = load_profile(Path(args.profile))
        jobs = load_jobs_json(Path(args.input))
        limit = args.limit if args.limit is not None else settings.match_max_jobs
        if limit < 1:
            raise MatchingError("--limit must be at least 1.")
        planned = min(
            sum(
                job.validity_status != "invalid"
                and bool(
                    (job.description or "").strip()
                    or any(item.strip() for item in job.requirements)
                )
                and bool(profile.career_history.strip())
                for job in jobs
            ),
            limit,
        )
        if args.dry_run:
            print("Mode: dry-run (no provider calls)")
            print(f"Jobs received: {len(jobs)}")
            print(f"Jobs planned: {planned}")
            print(
                "LLM provider/model: "
                f"{settings.llm_provider or 'not configured'} / "
                f"{settings.llm_model or 'not configured'}"
            )
            print(
                "Embedding provider/model: "
                f"{settings.embedding_provider or 'not configured'} / "
                f"{settings.embedding_model or 'not configured'}"
            )
            print(f"Planned LLM calls: {planned}")
            print(f"Planned embedding index build: {1 if planned else 0}")
            return 0
        if args.limit is not None:
            settings = settings.model_copy(update={"match_max_jobs": limit})
        if args.live:
            chat_model, embeddings = create_live_models(settings)
            report = match_jobs(
                profile,
                jobs,
                settings,
                live=True,
                chat_model=chat_model,
                embeddings=embeddings,
                debug_evidence=args.debug_evidence,
            )
        else:
            report = match_jobs(
                profile,
                jobs,
                settings,
                live=False,
                debug_evidence=args.debug_evidence,
            )
    except (MatchingError, ProfileError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    if args.json or args.output:
        payload = report.model_dump_json(indent=2) + "\n"
        if args.output:
            output_path = Path(args.output)
            temporary_path = output_path.with_name(f".{output_path.name}.tmp")
            try:
                temporary_path.write_text(payload, encoding="utf-8")
                temporary_path.replace(output_path)
            except OSError as error:
                print(f"Error: could not write matching report: {error}")
                return 2
            print(f"Wrote matching report to {output_path}")
        else:
            print(payload, end="")
        return 0
    print(f"Mode: {report.mode}")
    print(f"Jobs received: {report.jobs_received}")
    print(f"Jobs evaluated: {report.jobs_evaluated}")
    print(f"Duplicates removed: {report.duplicates_removed}")
    print(f"LLM calls attempted: {report.llm_calls_attempted}")
    print(f"Embedding documents: {report.embedding_documents}")
    print(f"Jobs skipped: {len(report.skipped_jobs)}")
    for skipped in report.skipped_jobs:
        print(f"- Skipped {skipped['title']}: {skipped['reason']}")
    for evaluation in report.evaluations:
        print(
            f"- [{evaluation.eligibility}] {evaluation.job.title} "
            f"(demonstrated fit: {evaluation.demonstrated_fit}/100; "
            f"evidence coverage: {evaluation.evidence_coverage}/100)"
        )
        print(f"  URL: {evaluation.job.url or 'unknown'}")
        print(f"  Reason: {evaluation.reason}")
        if evaluation.matching_strengths:
            print(f"  Strengths: {', '.join(evaluation.matching_strengths)}")
        if evaluation.gaps:
            print(f"  Gaps: {', '.join(evaluation.gaps)}")
        if evaluation.uncertainties:
            print(f"  Uncertainties: {', '.join(evaluation.uncertainties)}")
    if args.debug_evidence and report.evidence_debug is not None:
        print("Evidence debug:")
        print(report.model_dump_json(include={"evidence_debug"}, indent=2))
    return 0


def _run_enrich(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        jobs = load_job_input(Path(args.input))
        max_pages = args.max_pages or settings.enrichment_max_pages
        concurrency = args.concurrency or settings.enrichment_concurrency
        if max_pages < 1 or concurrency < 1:
            raise ValueError("--max-pages and --concurrency must be at least 1")
        if args.dry_run:
            enriched = enrich_jobs(jobs, max_pages=max_pages, dry_run=True)
        else:
            with HttpxPageFetcher(
                timeout=args.timeout or settings.enrichment_timeout_seconds,
                max_bytes=args.max_bytes or settings.enrichment_max_bytes,
            ) as fetcher:
                enriched = enrich_jobs(
                    jobs,
                    fetcher=fetcher,
                    max_pages=max_pages,
                    concurrency=concurrency,
                )
        write_jobs_json(Path(args.output), enriched)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    counts = {
        "fetched": sum(
            job.extraction_status not in {None, "skipped"} for job in enriched
        ),
        "successfully enriched": sum(
            job.extraction_status == "success" for job in enriched
        ),
        "invalid": sum(job.validity_status == "invalid" for job in enriched),
        "uncertain": sum(job.validity_status == "uncertain" for job in enriched),
        "failed": sum(
            job.extraction_status in {"failed", "blocked"} for job in enriched
        ),
    }
    print(f"Wrote {len(enriched)} jobs to {args.output}")
    print("; ".join(f"{key}: {value}" for key, value in counts.items()))
    if args.dry_run:
        print("Mode: dry-run (no network requests)")
    return 0


def _run_digest(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        if args.preview and args.send:
            raise DigestError("Choose either --preview or --send, not both.")
        report = load_matching_report(Path(args.input))
        state_path = Path(args.state) if args.state else settings.digest_state_path
        state = load_delivery_state(state_path)
        digest = build_digest(
            report,
            min_score=(
                args.min_score
                if args.min_score is not None
                else settings.digest_min_match_score
            ),
            max_jobs=args.max_jobs or settings.digest_max_jobs,
            sent_ids=set(state.sent),
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(digest.html, encoding="utf-8")
            print(f"Wrote HTML preview to {output_path}")
        if not digest.evaluations:
            print("No new qualifying jobs; no email sent.")
            return 0
        if not args.send:
            print("Mode: preview (no Gmail connection)")
            print(f"Subject: {digest.subject}")
            print(digest.text, end="")
            return 0
        password = (
            settings.gmail_app_password.get_secret_value()
            if settings.gmail_app_password
            else None
        )
        send_gmail(digest, address=settings.gmail_address, app_password=password)
        save_delivery_state(state_path, mark_digest_sent(state, digest.identities))
        print(f"Submitted digest with {len(digest.evaluations)} job(s) to Gmail.")
        return 0
    except (DigestError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2


def _run_generate_cv(args: argparse.Namespace) -> int:
    settings = get_settings()
    profile_path = Path(args.profile) if args.profile else settings.profile_path
    if profile_path is None:
        print("Error: provide --profile or configure NEXTROLE_PROFILE_PATH")
        return 2
    try:
        cv, job = generate_cv_from_profile(args.url, profile_path, settings=settings)
        path = save_cv(cv, job, Path(args.output_root))
    except (CVGenerationError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Company and job title: {job.company or 'unknown'} — {job.title}")
    print(f"Output file: {path}")
    bullet_count = sum(len(item.bullets) for item in cv.experience)
    print(f"Selected experience bullets: {bullet_count}")
    if cv.validation_warnings:
        print("Validation warnings:")
        for warning in cv.validation_warnings:
            print(f"- {warning}")
    else:
        print("Validation warnings: none")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _run_check()
    if args.command == "discover":
        return _run_discover(args)
    if args.command == "profile" and args.profile_command == "validate":
        return _run_profile_validate(args)
    if args.command == "match":
        return _run_match(args)
    if args.command == "enrich":
        return _run_enrich(args)
    if args.command == "digest":
        return _run_digest(args)
    if args.command == "generate-cv":
        return _run_generate_cv(args)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
