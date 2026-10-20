"""Command-line interface for the NextRole."""

import argparse
from collections.abc import Sequence

from .config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nextrole",
        description="A configurable AI-powered job-search agent called NextRole.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="Validate basic local configuration.")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _run_check()
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
