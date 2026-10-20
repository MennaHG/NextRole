# Architecture

## Scope

This document describes the intended architecture while clearly marking what
exists today. Phase 3 adds local profile parsing and deterministic query
generation on top of bounded, opt-in Perplexity discovery.

## Planned pipeline

```text
job discovery -> normalization -> preference filtering -> profile retrieval
    -> job-fit evaluation -> ranking -> daily digest delivery
```

The eventual workflow will discover postings from the last 24 hours, normalize
them into `JobPosting`, filter them using `JobPreferences`, retrieve relevant
career-history passages with embeddings and a vector store, evaluate matches,
and deliver up to 20 results.

## Current modules

- `src/nextrole/models.py`: Pydantic models for `JobPosting`,
  `JobPreferences`, `CandidateProfile`, and `JobMatch`. Job fields that a
  source does not provide remain optional, and unknown source fields are
  preserved.
- `src/nextrole/config.py`: environment-backed settings. LLM and
  embedding provider/model names are independent so they can be configured
  separately later.
- `src/nextrole/discovery.py`: the Perplexity adapter, request budgeting,
  result normalization, URL deduplication, and deterministic fixture mode.
- `src/nextrole/profile.py`: private Markdown parsing and deterministic query
  generation from `CandidateProfile` preferences.
- `src/nextrole/cli.py`: the offline `--help`, `check`, and `discover` commands.

LangChain Core remains a dependency for future model, embedding, and retrieval
abstractions. Discovery does not filter candidates, call an LLM, scrape pages,
or enrich results with another service.

## Future boundaries

- The discovery adapter isolates the Perplexity Search API and is bounded to a
  configured request budget; live calls require explicit CLI opt-in.
- A profile adapter will support local Markdown first and private Google Drive
  later; profile parsing should produce `CandidateProfile` without mixing
  preferences into career-history text.
- A retrieval layer will own document splitting, embeddings, and ChromaDB.
- An evaluation layer will accept configurable LangChain chat-model and
  embedding implementations rather than hard-coding a vendor.
- Delivery and scheduling will be separate from matching so they can be tested
  without network access.

Each integration should be introduced behind a small interface with offline
unit tests and credentials supplied only through environment configuration.
