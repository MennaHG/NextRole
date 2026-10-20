# Architecture

## Scope

This document describes the intended architecture while clearly marking what
exists today. Phase 1 is the repository foundation only; no external provider
or persistence integration is active.

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
- `src/nextrole/cli.py`: the offline `--help` and `check` commands.

LangChain Core is a dependency for future model, embedding, and retrieval
abstractions. Phase 1 deliberately does not add provider packages or call an
LLM.

## Future boundaries

- A discovery adapter will isolate the Perplexity Search API.
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
