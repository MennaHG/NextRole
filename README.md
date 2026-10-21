# NextRole

An open-source Python project for discovering and evaluating recent job
postings against a candidate's preferences and career history. It is also a
small learning project for LangChain, embeddings, vector databases, and RAG.

## Current status: Phase 6 — job-description enrichment, RAG matching, email digest, and CV generation

Implemented:

- A `src/`-layout Python package with Pydantic domain models.
- Environment-based configuration for an optional profile and independently
  named LLM and embedding providers.
- An offline CLI with `--help`, `check`, and fixture-backed `discover`.
- An opt-in Perplexity Search API adapter with query batching, URL
  deduplication, conservative normalization, and request budgets.
- A local Markdown candidate-profile parser and deterministic profile-based
  query generator.
- A RAG matching pipeline that chunks career history, builds one ephemeral
  vector index per run, and evaluates each job with one structured LangChain
  chat-model call.
- A bounded, robots-aware job-page enrichment command using `httpx`,
  Schema.org `JobPosting` JSON-LD first, and BeautifulSoup HTML fallback.
- An on-demand CV generation command that extracts one job URL, makes one
  structured LangChain call, and writes evidence-grounded CV JSON.
- Deterministic profile metadata preservation for contact details, employment
  metadata, and education, with validation warnings for unsupported claims.
- Unit tests, Ruff configuration, and GitHub Actions CI.
- A fictional example profile and repository hygiene rules.

Planned for later phases: private Google Drive profiles and scheduled GitHub
Actions runs. Google Drive, scheduling, automatic applications, and the
Perplexity Agent API are intentionally out of scope.

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

## Getting started

The quickest way to try NextRole is to run the pipeline offline first. These
commands do not call an LLM, send email, or require paid credentials.

1. Create and activate the virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -e ".[dev]"
   ```

2. Create a private candidate profile:

   ```bash
   mkdir -p private
   cp examples/profile.example.md private/profile.md
   nextrole profile validate --file private/profile.md
   ```

   Replace the example career history and preferences with your own information.
   Keep the file under `private/`; that directory is ignored by Git.

3. Discover jobs. Start with a dry run to inspect the search plan:

   ```bash
   nextrole discover --profile private/profile.md --dry-run
   ```

   To use live discovery later, configure `PERPLEXITY_API_KEY` and explicitly
   add `--live`.

4. Enrich the discovered jobs. This fetches job pages and extracts descriptions
   locally; it does not use an LLM:

   ```bash
   nextrole enrich --input jobs.json --output enriched_jobs.json
   ```

5. Run an offline matching check:

   ```bash
   nextrole match \
     --profile private/profile.md \
     --input enriched_jobs.json \
     --dry-run
   ```

   For a saved, inspectable matching report without paid provider calls, omit
   `--dry-run` and add `--json --output outputs/matching-report.json`. Fixture
   mode is intentionally conservative and returns uncertain evaluations. A
   live semantic evaluation requires the explicit `--live` flag and configured
   providers.

6. Preview the email digest without connecting to Gmail:

   ```bash
   nextrole digest \
     --input outputs/matching-report.json \
     --preview \
     --output outputs/job-digest.html
   ```

   The preview includes only validated jobs with usable application links,
   excludes explicitly ineligible jobs, and distinguishes requirements that are
   not met from requirements that remain unverified.

7. Send the digest only after configuring a Gmail App Password. The message is
   sent to the same address used as the sender:

   ```ini
   GMAIL_ADDRESS=yourname@gmail.com
   GMAIL_APP_PASSWORD=your_google_app_password
   ```

   Then run:

   ```bash
   nextrole digest \
     --input outputs/matching-report.json \
     --send
   ```

   Use `--max-jobs` and `--min-score` to limit a test send. Delivery state is
   stored in `data/digest_state.json`, so successfully submitted jobs are not
   intentionally sent again on later runs. See [Email job digest](#email-job-digest)
   for Gmail setup details and delivery limitations.

8. Generate a tailored CV from any accessible job posting URL. This requires
   the configured LLM provider and model:

   ```bash
   nextrole generate-cv \
     --url "https://company.com/jobs/backend-engineer" \
     --profile ./private/profile.md
   ```

   The generated JSON is saved under `outputs/cvs/`, with a company-and-title
   directory name. Contact information, employment metadata, and education are
   loaded from the profile; the LLM supplies tailored wording and selection.

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
nextrole enrich --input ./jobs.json --output ./enriched_jobs.json
nextrole enrich --input ./jobs.json --output ./enriched_jobs.json --dry-run
nextrole match --profile ./private/profile.md --input ./jobs.json
nextrole match --profile ./private/profile.md --input ./jobs.json --dry-run
nextrole match --profile ./private/profile.md --input ./jobs.json --live
nextrole match --profile ./private/profile.md --input ./enriched_jobs.json \
  --json --debug-evidence
nextrole match --profile ./private/profile.md --input ./enriched_jobs.json \
  --json --debug-evidence --output ./outputs/debug-evidence.json
nextrole digest --input ./outputs/matching-report.json --preview
nextrole digest --input ./outputs/matching-report.json --send
nextrole generate-cv --url "https://company.com/jobs/backend-engineer" \
  --profile ./private/profile.md
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
or technology filtering. Career history is retained as text for RAG indexing.

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

### Phase 4.2 matching

The evaluator returns semantic requirement assessments, including an evidence
relationship (`direct`, `equivalent`, `transferable`, `absent`, or
`contradictory`). Python calculates the reported metrics after validating those
assessments; the LLM does not generate a score. `demonstrated_fit` weights
required requirements by 3 and preferred or unspecified requirements by 1,
using values of 1.0, 0.5, and 0 for `met`, `partially_met`, and `not_met`.
`unknown` requirements are excluded from demonstrated fit. `evidence_coverage`
is the weighted proportion of requirements that are assessable. The legacy
`match_score` field remains as an alias for `demonstrated_fit`.

Eligibility remains independent from both metrics: a high demonstrated fit
does not turn uncertain location, authorization, certification, or other
mandatory constraints into an eligible decision.

### Phase 4.1 enrichment

Enrichment is a local extraction step; it does not call Perplexity, an LLM,
or an embedding provider. It follows redirects, checks `robots.txt`, accepts
HTML only, enforces a response-size limit, and uses bounded concurrency. The
limits can be set with `ENRICHMENT_TIMEOUT_SECONDS`, `ENRICHMENT_MAX_PAGES`,
`ENRICHMENT_CONCURRENCY`, and `ENRICHMENT_MAX_BYTES`. `--dry-run` writes an
inspectable file without network access.

Each output job keeps discovery metadata and may add `resolved_url`,
`description`, `requirements`, `company`, `location`, `work_arrangement`,
`employment_type`, `date_posted`, `validity_status`, `extraction_status`,
`extraction_source`, and `extraction_error`. Invalid pages stay in the JSON
for debugging but are excluded from matching. Uncertain and failed records
remain visible and are not safe to score from their title alone.

The generated `enriched_jobs.json` is ignored by Git by default. Reuse it for
repeatable offline matching. Extraction cannot execute JavaScript, bypass
authentication, solve CAPTCHAs, or recover content blocked by a site, so some
pages remain uncertain or unavailable.

Matching applies three pre-LLM gates: confirmed invalid postings are skipped,
postings without both a description and requirements are skipped, and no
matching/index work runs when candidate career evidence is unavailable.
Skipped records and reasons are reported in the matching JSON output.

```bash
nextrole match --profile ./private/profile.md \
  --input ./enriched_jobs.json --dry-run
nextrole match --profile ./private/profile.md \
  --input ./enriched_jobs.json
```

Matching accepts a JSON list of normalized jobs, or the JSON output from a
discovery report. A public fictional input is available at
[`examples/jobs.example.json`](examples/jobs.example.json):

```bash
nextrole match --profile ./private/profile.md \
  --input ./examples/jobs.example.json
nextrole match --profile ./private/profile.md \
  --input ./examples/jobs.example.json --dry-run
```

Without `--live`, matching uses deterministic fixture embeddings and a fixture
model, so it is offline and makes no provider calls. `--dry-run` only displays
the planned job count, providers, and call count. `--live` is required before
the configured LLM and embedding providers are initialized. Configure them
independently with `NEXTROLE_LLM_PROVIDER`, `NEXTROLE_LLM_MODEL`,
`NEXTROLE_EMBEDDING_PROVIDER`, and `NEXTROLE_EMBEDDING_MODEL`, plus the
generic `LLM_API_KEY`. Embeddings reuse `LLM_API_KEY` by default; set the
optional `EMBEDDING_API_KEY` only when they need a separate credential.
For local development, use Ollama for both providers; Ollama requires no API
key. The example configuration uses a small local chat model and
`nomic-embed-text` for embeddings. Install Ollama separately, pull the models,
and keep the Ollama service running. OpenAI, Anthropic (chat), and Google/Gemini
(chat and embeddings) remain available for deployment, subject to the
corresponding LangChain package being installed.

Example local setup:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python3 -m pip install langchain-ollama
nextrole match --profile ./private/profile.md --input ./jobs.json --live
```

The career-history section is split into chunks and indexed once in an
ephemeral Chroma collection, with an in-memory fallback for dependency-limited
offline development. Preferences are supplied directly to the evaluator; they
are not embedded as career history. Each selected job gets at most one LLM
call. The default limit is 20 jobs, and `MATCH_MAX_JOBS`, `RAG_CHUNK_SIZE`,
`RAG_CHUNK_OVERLAP`, and `RAG_TOP_K` configure the run. Unknown job fields are
retained and reported as uncertainties; matching does not reject unfamiliar
technologies or on-site roles before semantic evaluation.

Fixture output is deliberately `uncertain`: it exercises the pipeline without
pretending to perform semantic matching. No LLM or embedding calls occur in
tests, profile validation, or dry-run mode. Matching produces structured JSON
reports for the separate digest delivery layer; NextRole does not apply to jobs
or contact recruiters.

### Phase 5: Email job digest

The digest command consumes a saved matching report and does not run matching or
make additional LLM calls. It includes only jobs with `valid` enrichment status,
an application URL, and a score at or above the configured minimum. Explicitly
`ineligible` jobs are excluded. Existing ranking is preserved, and the default
maximum digest size is 10 jobs.

Preview without connecting to Gmail:

```bash
nextrole digest --input ./outputs/matching-report.json --preview \
  --output ./outputs/job-digest.html
```

Send the same report after configuring Gmail:

```bash
nextrole digest --input ./outputs/matching-report.json --send
```

Configure `DIGEST_MIN_MATCH_SCORE`, `DIGEST_MAX_JOBS`, and
`DIGEST_STATE_PATH` to change filtering and delivery-state storage. The state
file records job identities only after Gmail accepts the SMTP submission, so
repeated runs do not intentionally resend the same jobs. SMTP submission is
not a guarantee of inbox delivery, and a process failure between submission
and state persistence can still result in a duplicate on a later run.

Gmail setup:

1. Enable Google 2-Step Verification for the Gmail account in Google Account
   security settings.
2. Open **App passwords**, create a password for NextRole, and copy the
   generated 16-character value. App passwords may be unavailable for some
   managed or protected accounts.
3. Set the address and app password in a local `.env` file, or in the process
   environment:

```ini
GMAIL_ADDRESS=yourname@gmail.com
GMAIL_APP_PASSWORD=your_google_app_password
```

The address is used for both sender and recipient. The app password is never
printed or included in the delivery state. For a future GitHub Actions
workflow, store these values as repository or environment secrets named
`GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD`; do not commit them to the repository.

### Evidence-grounded evaluation

Each evaluation now includes optional structured `requirement_assessments`.
Every assessment records the job requirement, whether it is required or
preferred, its evidence status (`met`, `partially_met`, `not_met`, or
`unknown`), candidate evidence, and an explanation. The evaluator prompt keeps
candidate preferences, retrieved career history, and enriched job data in
separate tagged sections and requires these assessments before the final score.
Job-page content, including third-party profiles, is never candidate evidence.

Use `--debug-evidence` only when inspecting a run. It adds retrieved career
chunks, extracted job requirements, assessments, eligibility, and score to the
output; it is intentionally opt-in because retrieved chunks may contain
private profile information. Add `--output path.json` to save the formatted
report directly to a file.

### Tailored CV generation

Generate one structured CV JSON from a job URL and the existing Markdown profile:

```bash
nextrole generate-cv \
  --url "https://company.com/jobs/backend-engineer" \
  --profile ./private/profile.md
```

The command makes one structured LangChain model call after extracting the job
page. Contact details and education are copied from the profile, while
employment metadata is preserved from recognizable profile headings and dates.
Each generated experience bullet carries its own evidence references. Skills,
projects, metrics, and claim terms are checked against profile evidence;
validation warnings are included in the JSON and printed in the CLI summary.
An empty generated skills section fails validation. Generated files are saved
under `outputs/cvs/`; the profile is never modified.

CV generation currently produces structured JSON only. Google Docs, PDF, and
LaTeX rendering are not yet implemented.

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
