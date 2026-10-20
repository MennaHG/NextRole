# NextRole

An open-source Python project for discovering and evaluating recent job
postings against a candidate's preferences and career history. It is also a
small learning project for LangChain, embeddings, vector databases, and RAG.

## Current status: Phase 3 — profile-based job discovery

Implemented:

- A `src/`-layout Python package with Pydantic domain models.
- Environment-based configuration for an optional profile and independently
  named LLM and embedding providers.
- An offline CLI with `--help`, `check`, and fixture-backed `discover`.
- An opt-in Perplexity Search API adapter with query batching, URL
  deduplication, conservative normalization, and request budgets.
- A local Markdown candidate-profile parser and deterministic profile-based
  query generator.
- Unit tests, Ruff configuration, and GitHub Actions CI.
- A fictional example profile and repository hygiene rules.

Planned for later phases: private Google Drive profiles, preference filtering,
ChromaDB indexing, LangChain-based retrieval and scoring, email digests, and
scheduled GitHub Actions runs. Those integrations are intentionally not
implemented yet.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` if you want to experiment with local settings.
Offline fixture and dry-run discovery do not require credentials or a network
connection. Obtain a key from [Perplexity API Console](https://console.perplexity.ai)
and set `PERPLEXITY_API_KEY` only when you intend to use live discovery.

## CLI

```bash
nextrole --help
nextrole check
nextrole profile validate --file ./private/profile.md
nextrole discover --query "backend engineer Egypt"
nextrole discover --profile ./private/profile.md --dry-run
nextrole discover --profile ./private/profile.md --live
nextrole discover --query "backend engineer Egypt" --dry-run
nextrole discover --query "backend engineer Egypt" --hours 24 --limit 20 --live
```

The same commands work without installation through the module entry point:

```bash
python3 -m nextrole --help
python3 -m nextrole check
```

`discover` uses deterministic fixture mode unless `--live` is supplied. Use
`--dry-run` to inspect queries, documented API parameters, planned attempts,
and the request budget without making a call. Multiple `--query` values are
batched in groups of five. Queries can also be configured as a JSON list in
`SEARCH_QUERIES`.

### Candidate profiles

Create a private Markdown file such as `private/profile.md` using the format
shown in [`examples/profile.example.md`](examples/profile.example.md). It must
contain these sections:

```markdown
## Preferences
- Roles: Backend Engineer, Full Stack Engineer
- Locations: Cairo, Egypt
- Seniority: Mid-level, Senior
- Remote: Yes
- Keywords: Python, React, Node.js
- Maximum job age: 24 hours

## Career history
Describe professional experience here.

## Technical skills
Python, SQL, Docker

## Education
BSc in Computer Science.
```

Validate it offline with `nextrole profile validate --file
./private/profile.md`. Supplying `--profile` to `discover` generates up to
three stable queries from roles, technologies, and the first non-remote
location. The first query is intentionally broader for coverage. Explicit
`--query` values override generated queries. Profile validation and query
generation never call Perplexity and do not print career history.

Keep private profiles under `private/` or `profiles/`; both are ignored by Git.
The public example contains fictional data only. Phase 3 preserves ambiguous
results for future semantic evaluation rather than applying rigid candidate
or technology filtering. Career history is retained as text for Phase 4
indexing, which is not implemented yet.

The default budget is one attempted request per run and retries are disabled.
Retries can be set to at most one, and every retry counts against the budget.
Search API billing is per successful `POST /search` request, not per query;
batching up to five queries therefore uses one billing unit when it is one
successful request, but batching does not bypass the configured safety budget.
Invalid, rate-limited, and upstream-failed requests are documented as not
billed; a successful empty response is billed.

Results are general web results, not verified job postings. NextRole keeps the
original URL, snippet, and date metadata, leaves unknown fields unknown, and
skips obvious job-board search pages. Publication dates and `last_updated` are
kept distinct; an update date is never treated as a publication date. Review
results before using them downstream.

The `PERPLEXITY_API_KEY` and discovery variables in `.env.example` are
accepted alongside the existing `NEXTROLE_`-prefixed settings. Credentials
are never printed, logged, or needed by tests.

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
