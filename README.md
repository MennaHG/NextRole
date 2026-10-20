# NextRole

An open-source Python project for discovering and evaluating recent job
postings against a candidate's preferences and career history. It is also a
small learning project for LangChain, embeddings, vector databases, and RAG.

## Current status: Phase 1

Implemented:

- A `src/`-layout Python package with Pydantic domain models.
- Environment-based configuration for an optional profile and independently
  named LLM and embedding providers.
- An offline CLI with `--help` and `check`.
- Unit tests, Ruff configuration, and GitHub Actions CI.
- A fictional example profile and repository hygiene rules.

Planned for later phases: Perplexity job discovery, private Google Drive
profiles, preference filtering, ChromaDB indexing, LangChain-based retrieval
and scoring, email digests, and scheduled GitHub Actions runs. Those
integrations are intentionally not implemented yet.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` if you want to experiment with local settings.
Phase 1 does not require credentials or a network connection at runtime.

## CLI

```bash
nextrole --help
nextrole check
```

The same commands work without installation through the module entry point:

```bash
python -m nextrole --help
python -m nextrole check
```

Configuration variables use the `NEXTROLE_` prefix. Provider names and model
names are recorded for future integration, but no provider APIs are called in
Phase 1.

## Development

```bash
pytest
ruff check .
ruff format --check .
```

All tests are offline and do not require credentials, paid services, or API
calls. Start with [`examples/profile.example.md`](examples/profile.example.md)
when creating a private profile, and keep real profile files out of version
control.

See [`docs/architecture.md`](docs/architecture.md) for the planned pipeline
and module boundaries.
