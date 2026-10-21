# Architecture

## Scope

This document describes the intended architecture while clearly marking what
exists today. Phase 4.1 adds bounded, local job-page enrichment before the
offline-first RAG matching stage.

## Planned pipeline

```text
job discovery -> normalization -> page enrichment -> profile retrieval -> job-fit evaluation
    -> ranking -> structured report -> daily digest delivery
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
- `src/nextrole/enrichment.py`: robots-aware `httpx` fetching, response limits,
  JSON-LD and conservative BeautifulSoup extraction, validity classification,
  bounded concurrency, and atomic enriched JSON output.
- `src/nextrole/profile.py`: private Markdown parsing and deterministic query
  generation from `CandidateProfile` preferences.
- `src/nextrole/matching.py`: career-history chunking, ephemeral Chroma (or
  offline in-memory) retrieval, provider construction, structured evaluation,
  requirement-to-evidence assessment, opt-in evidence tracing, deduplication,
  and JSON job input.
- `src/nextrole/cli.py`: the offline `--help`, `check`, `discover`, `enrich`,
  `profile validate`, and `match` commands.

Discovery does not filter candidates or call an LLM. Enrichment fetches only
HTML pages, respects robots.txt, does not render JavaScript, and preserves
failed, blocked, invalid, and uncertain records. Matching keeps the LLM and
embedding providers independent; invalid pages are excluded while uncertain
records remain visible. The enriched JSON can be matched repeatedly without
fetching pages again.

Evaluation remains one structured LLM call per job. The response models each
important job requirement with `RequirementAssessment`, including its evidence
relationship, but does not generate a numeric score. Python normalizes
unambiguous relationships and calculates `demonstrated_fit` and
`evidence_coverage` deterministically: required requirements have weight 3;
preferred and unspecified requirements have weight 1; unknown assessments are
excluded from demonstrated fit but counted in the coverage denominator.
Eligibility remains independent from both metrics. Candidate
preferences, candidate career chunks, and job evidence are separate prompt
sections. The optional `--debug-evidence` trace is disabled by default because
it can expose private career-history chunks.

The email delivery boundary is `src/nextrole/digest.py`. It consumes saved
`MatchingReport` JSON, renders deterministic HTML and plain text, and submits
multipart mail through `smtplib.SMTP_SSL` on Gmail port 465. It has no provider
or matching dependency. A local atomic JSON state file records successfully
submitted job identities after SMTP returns; a future durable store can replace
that file for scheduled GitHub Actions runs. Preview mode never opens SMTP.

## Future boundaries

- The discovery adapter isolates the Perplexity Search API and is bounded to a
  configured request budget; live calls require explicit CLI opt-in.
- A profile adapter will support local Markdown first and private Google Drive
  later; profile parsing should produce `CandidateProfile` without mixing
  preferences into career-history text.
- Delivery and scheduling will be separate from matching so they can be tested
  without network access.

Each integration should be introduced behind a small interface with offline
unit tests and credentials supplied only through environment configuration.
