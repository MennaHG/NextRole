import json
from pathlib import Path

import pytest

from nextrole.cli import main
from nextrole.config import Settings
from nextrole.matching import (
    CareerIndex,
    DeterministicEmbeddings,
    MatchingError,
    build_career_index,
    chunk_career_history,
    load_jobs_json,
    match_jobs,
)
from nextrole.models import EvaluationDecision, JobPosting
from nextrole.profile import parse_profile

PROFILE = parse_profile(
    """## Preferences
- Roles: Backend Engineer
- Locations: Cairo, Egypt
- Remote: No
- Keywords: Python, APIs

## Career history
Built fictional Python APIs and data pipelines. Led a small platform team.

## Technical skills
Python, SQL, Docker

## Education
BSc in Computer Science.
"""
)


def job(
    title: str = "Backend Engineer", url: str = "https://jobs.test/1"
) -> JobPosting:
    return JobPosting(
        title=title,
        url=url,
        location="Cairo, Egypt",
        description="Build services using an unfamiliar technology.",
    )


class FakeChat:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type[EvaluationDecision]) -> "FakeChat":
        assert schema is EvaluationDecision
        return self

    def invoke(self, prompt: str) -> object:
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


class CountingRetriever:
    document_count = 1

    def __init__(self) -> None:
        self.calls = 0

    def similarity_search(self, query: str, k: int):
        del query, k
        self.calls += 1
        return []


def test_career_chunks_are_indexed_with_metadata() -> None:
    documents = chunk_career_history(PROFILE, chunk_size=35, chunk_overlap=5)

    assert documents
    assert all(
        document.metadata["source"] == "career_history" for document in documents
    )
    assert "Python APIs" in " ".join(document.page_content for document in documents)


def test_invalid_chunk_overlap_is_rejected() -> None:
    with pytest.raises(MatchingError, match="smaller"):
        chunk_career_history(PROFILE, chunk_size=10, chunk_overlap=10)


def test_build_index_is_constructed_once_and_receives_documents() -> None:
    created: list[object] = []

    class Store:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.documents = []
            created.append(self)

        def add_documents(self, documents) -> None:
            self.documents.extend(documents)

        def similarity_search(self, query: str, k: int):
            return self.documents[:k]

    index = build_career_index(
        PROFILE,
        DeterministicEmbeddings(),
        chunk_size=40,
        chunk_overlap=5,
        vector_store_factory=Store,
    )

    assert index.document_count > 0
    assert len(created) == 1
    assert created[0].kwargs["persist_directory"] is None


def test_matching_reuses_injected_index_and_calls_model_once_per_job() -> None:
    retriever = CountingRetriever()
    index = CareerIndex(retriever, document_count=1)
    model = FakeChat(
        {
            "eligibility": "eligible",
            "match_score": 82,
            "reason": "Relevant fictional API experience.",
            "matching_strengths": ["API experience"],
            "gaps": [],
            "uncertainties": [],
        }
    )
    settings = Settings(match_max_jobs=2)

    report = match_jobs(
        PROFILE,
        [job(), job("Platform Engineer", "https://jobs.test/2")],
        settings,
        chat_model=model,
        embeddings=DeterministicEmbeddings(),
        index=index,
    )

    assert report.jobs_evaluated == 2
    assert report.llm_calls_attempted == 2
    assert model.calls == 2
    assert retriever.calls == 2
    assert report.embedding_operations == 1
    assert all(item.eligibility == "eligible" for item in report.evaluations)
    assert "<job_evidence>" in model.prompts[0]


def test_requirement_assessments_are_structured_and_evidence_grounded() -> None:
    model = FakeChat(
        {
            "eligibility": "eligible",
            "match_score": 78,
            "reason": "The required backend experience is supported.",
            "matching_strengths": ["Node.js backend delivery"],
            "gaps": ["Camunda is not documented"],
            "uncertainties": ["RabbitMQ-specific experience is not documented"],
            "requirement_assessments": [
                {
                    "requirement": "Node.js",
                    "importance": "required",
                    "status": "met",
                    "candidate_evidence": ["Built Node.js backend services."],
                    "explanation": "The candidate evidence explicitly names Node.js.",
                },
                {
                    "requirement": "RabbitMQ",
                    "importance": "preferred",
                    "status": "partially_met",
                    "candidate_evidence": ["Used Azure Service Bus for messaging."],
                    "explanation": (
                        "Messaging is transferable, but RabbitMQ is not explicit."
                    ),
                },
                {
                    "requirement": "Camunda",
                    "importance": "required",
                    "status": "unknown",
                    "candidate_evidence": [],
                    "explanation": (
                        "The retrieved evidence does not establish Camunda experience."
                    ),
                },
            ],
        }
    )
    report = match_jobs(
        PROFILE,
        [job()],
        Settings(),
        chat_model=model,
        embeddings=DeterministicEmbeddings(),
    )

    assessments = report.evaluations[0].requirement_assessments
    assert [assessment.status for assessment in assessments] == [
        "met",
        "partially_met",
        "unknown",
    ]
    assert assessments[0].importance == "required"
    assert assessments[1].importance == "preferred"
    assert "RabbitMQ" in report.evaluations[0].uncertainties[0]
    assert "Camunda" in report.evaluations[0].gaps[0]


def test_scores_are_calculated_from_assessments_and_unknown_is_excluded() -> None:
    model = FakeChat(
        {
            "eligibility": "uncertain",
            "reason": "One mandatory requirement is unverified.",
            "requirement_assessments": [
                {
                    "requirement": "Required backend delivery",
                    "importance": "required",
                    "status": "met",
                    "evidence_relationship": "direct",
                    "candidate_evidence": ["Built backend services."],
                    "explanation": "Direct evidence.",
                },
                {
                    "requirement": "Required workflow tool",
                    "importance": "required",
                    "status": "not_met",
                    "evidence_relationship": "absent",
                    "candidate_evidence": [],
                    "explanation": "The evidence does not establish this tool.",
                },
                {
                    "requirement": "Preferred messaging",
                    "importance": "preferred",
                    "status": "met",
                    "evidence_relationship": "transferable",
                    "candidate_evidence": ["Used Azure Service Bus."],
                    "explanation": "Transferable messaging experience.",
                },
                {
                    "requirement": "Optional certification",
                    "importance": "unspecified",
                    "status": "not_met",
                    "evidence_relationship": "direct",
                    "candidate_evidence": ["Career history lists no certification."],
                    "explanation": "Explicitly not present.",
                },
            ],
        }
    )

    evaluation = match_jobs(
        PROFILE,
        [job()],
        Settings(),
        chat_model=model,
        embeddings=DeterministicEmbeddings(),
    ).evaluations[0]

    # (3*1 + 1*.5 + 1*0) / (3+1+1) = 70%; unknown required is excluded.
    assert evaluation.demonstrated_fit == 70.0
    # Assessable weight is 5 out of the total 8, so coverage is 62.5%.
    assert evaluation.evidence_coverage == 62.5
    assert evaluation.match_score == evaluation.demonstrated_fit
    assert evaluation.requirement_assessments[1].status == "unknown"
    assert evaluation.requirement_assessments[2].status == "partially_met"


def test_job_content_is_not_candidate_evidence() -> None:
    contaminated_job = JobPosting(
        title="Data Scientist",
        url="https://jobs.test/data-scientist",
        description=(
            "A different person's profile describes extensive machine learning."
        ),
    )
    model = FakeChat(
        {
            "eligibility": "uncertain",
            "match_score": 20,
            "reason": "The job-page person's experience is not candidate evidence.",
            "matching_strengths": ["Backend experience is documented."],
            "gaps": [],
            "uncertainties": ["Data science experience is unknown."],
            "requirement_assessments": [
                {
                    "requirement": "Machine learning",
                    "importance": "unspecified",
                    "status": "unknown",
                    "candidate_evidence": [],
                    "explanation": "Only the job page mentions this experience.",
                }
            ],
        }
    )
    report = match_jobs(
        parse_profile(
            """## Preferences
- Roles: Backend Engineer
- Locations: Cairo

## Career history
Built Node.js and .NET backend services.

## Technical skills
Node.js, .NET

## Education
Computer Science degree.
"""
        ),
        [contaminated_job],
        Settings(),
        chat_model=model,
        embeddings=DeterministicEmbeddings(),
    )

    prompt = model.prompts[0]
    assert "Built Node.js and .NET backend services." in prompt
    assert "extensive machine learning" in prompt
    assert prompt.index("<candidate_career_evidence>") < prompt.index("<job_evidence>")
    assessment = report.evaluations[0].requirement_assessments[0]
    assert assessment.status == "unknown"
    assert assessment.candidate_evidence == []


def test_debug_evidence_is_opt_in() -> None:
    report = match_jobs(
        PROFILE,
        [job()],
        Settings(),
        chat_model=FakeChat(
            {"eligibility": "uncertain", "match_score": 0, "reason": "unknown"}
        ),
        embeddings=DeterministicEmbeddings(),
    )
    assert report.evidence_debug is None

    debug_report = match_jobs(
        PROFILE,
        [job()],
        Settings(),
        chat_model=FakeChat(
            {"eligibility": "uncertain", "match_score": 0, "reason": "unknown"}
        ),
        embeddings=DeterministicEmbeddings(),
        debug_evidence=True,
    )
    assert debug_report.evidence_debug is not None
    assert debug_report.evidence_debug[0].job_title == "Backend Engineer"


def test_hard_gates_skip_yassir_and_evaluate_blackstone_once() -> None:
    blackstone = JobPosting(
        title="Senior Backend Developer - Node.js | BlackStone eIT",
        url="https://jobs.test/blackstone",
        company="BlackStone eIT",
        description=(
            "Build Node.js microservices with Azure Service Bus, Camunda, and RabbitMQ."
        ),
    )
    yassir = JobPosting(
        title="Senior Back-End Engineer (Node j.s) Super App - Yassir",
        url="https://jobs.test/yassir",
    )
    model = FakeChat(
        {
            "eligibility": "uncertain",
            "match_score": 0,
            "reason": "Insufficient evidence for fixture evaluation.",
        }
    )

    report = match_jobs(
        PROFILE,
        [blackstone, yassir],
        Settings(),
        chat_model=model,
        embeddings=DeterministicEmbeddings(),
    )

    assert model.calls == 1
    assert report.jobs_evaluated == 1
    assert report.evaluations[0].job.title == blackstone.title
    assert report.skipped_jobs == [
        {
            "title": yassir.title,
            "reason": "description and requirements are missing",
        }
    ]


def test_missing_candidate_evidence_skips_without_llm_call() -> None:
    profile = PROFILE.model_copy(update={"career_history": "   "})
    model = FakeChat(
        {"eligibility": "uncertain", "match_score": 0, "reason": "unknown"}
    )

    report = match_jobs(
        profile,
        [job()],
        Settings(),
        chat_model=model,
        embeddings=DeterministicEmbeddings(),
    )

    assert model.calls == 0
    assert report.jobs_evaluated == 0
    assert report.embedding_documents == 0
    assert (
        report.skipped_jobs[0]["reason"] == "candidate career evidence is unavailable"
    )


def test_duplicate_jobs_and_limit_are_reported() -> None:
    model = FakeChat(
        {"eligibility": "uncertain", "match_score": 0, "reason": "unknown"}
    )
    report = match_jobs(
        PROFILE,
        [job(), job(), job("Other", "https://jobs.test/2")],
        Settings(match_max_jobs=1),
        chat_model=model,
        embeddings=DeterministicEmbeddings(),
    )

    assert report.duplicates_removed == 1
    assert report.jobs_received == 3
    assert report.jobs_evaluated == 1


def test_malformed_model_output_becomes_uncertain() -> None:
    report = match_jobs(
        PROFILE,
        [job()],
        Settings(),
        chat_model=FakeChat({"eligibility": "not-a-status"}),
        embeddings=DeterministicEmbeddings(),
    )

    assert report.evaluations[0].eligibility == "uncertain"
    assert report.evaluations[0].uncertainties


def test_jobs_json_and_match_dry_run_are_offline(tmp_path: Path, capsys) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text(
        """## Preferences
- Roles: Engineer
- Locations: Cairo

## Career history
Fictional engineering experience.

## Technical skills
Python

## Education
Fictional university.
""",
        encoding="utf-8",
    )
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps([job().model_dump(mode="json")]), encoding="utf-8")

    assert load_jobs_json(jobs_path)[0].title == "Backend Engineer"
    assert (
        main(
            [
                "match",
                "--profile",
                str(profile_path),
                "--input",
                str(jobs_path),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Mode: dry-run" in output
    assert "Planned LLM calls: 1" in output


def test_match_json_report_can_be_written_to_file(tmp_path: Path, capsys) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text(
        """## Preferences
- Roles: Engineer
- Locations: Cairo

## Career history
Fictional engineering experience.

## Technical skills
Python

## Education
Fictional university.
""",
        encoding="utf-8",
    )
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps([job().model_dump(mode="json")]), encoding="utf-8")
    output_path = tmp_path / "debug-evidence.json"

    assert (
        main(
            [
                "match",
                "--profile",
                str(profile_path),
                "--input",
                str(jobs_path),
                "--json",
                "--debug-evidence",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["evidence_debug"][0]["job_title"] == "Backend Engineer"
    assert "Wrote matching report" in capsys.readouterr().out


def test_invalid_jobs_json_is_clear(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(MatchingError, match="must be a JSON list"):
        load_jobs_json(path)
