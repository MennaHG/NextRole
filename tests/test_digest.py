from email.message import EmailMessage
from pathlib import Path

import pytest

from nextrole.digest import (
    DeliveryState,
    DigestError,
    build_digest,
    create_message,
    job_identity,
    load_delivery_state,
    mark_digest_sent,
    save_delivery_state,
    select_digest_jobs,
    send_gmail,
    validate_gmail_config,
)
from nextrole.models import (
    EvaluationDecision,
    JobEvaluation,
    JobPosting,
    MatchingReport,
    RequirementAssessment,
)


def evaluation(
    title: str,
    *,
    score: float = 80,
    eligibility: str = "eligible",
    validity: str = "valid",
    url: str | None = "https://jobs.example/apply",
) -> JobEvaluation:
    job = JobPosting(
        title=title,
        company="Example <Company>",
        url=url,
        resolved_url=url,
        location="Cairo",
        work_arrangement="Remote",
        validity_status=validity,
    )
    decision = EvaluationDecision(
        eligibility=eligibility,
        reason="Relevant evidence.",
        matching_strengths=["Built reliable APIs & services"],
        gaps=[],
        uncertainties=[],
        requirement_assessments=[
            RequirementAssessment(
                requirement="Camunda",
                importance="required",
                status="unknown",
                candidate_evidence=[],
                explanation="Not verified.",
            ),
            RequirementAssessment(
                requirement="Node.js",
                importance="required",
                status="met",
                candidate_evidence=["Built Node.js services."],
                explanation="Direct evidence.",
            ),
        ],
    )
    return JobEvaluation(
        job=job,
        **decision.model_dump(),
        demonstrated_fit=score,
        evidence_coverage=50,
        match_score=score,
    )


def test_digest_filters_sorts_and_deduplicates() -> None:
    first = evaluation("High", score=90)
    duplicate = evaluation("Duplicate", score=95)
    duplicate = duplicate.model_copy(update={"job": first.job})
    report = MatchingReport(
        evaluations=[
            evaluation("Ineligible", score=100, eligibility="ineligible"),
            evaluation("Unvalidated", score=100, validity="uncertain"),
            evaluation("Low", score=20),
            first,
            duplicate,
        ]
    )

    selected = select_digest_jobs(
        report, min_score=50, max_jobs=10, sent_ids={job_identity(first)}
    )

    assert selected == []
    selected = select_digest_jobs(report, min_score=50, max_jobs=10)
    assert [item.job.title for item in selected] == ["High"]


def test_digest_renders_html_and_plain_text_safely() -> None:
    digest = build_digest(MatchingReport(evaluations=[evaluation("<Senior> Dev")]))

    assert "&lt;Senior&gt; Dev" in digest.html
    assert "Example &lt;Company&gt;" in digest.html
    assert "Requirements still unverified" in digest.html
    assert "<Senior> Dev" in digest.text
    message = create_message(digest, "candidate@example.com")
    assert isinstance(message, EmailMessage)
    assert message["From"] == "candidate@example.com"
    assert message["To"] == "candidate@example.com"
    assert message.get_body("html") is not None
    assert message.get_body("plain") is not None


def test_empty_digest_is_explicit() -> None:
    digest = build_digest(MatchingReport())

    assert digest.evaluations == []
    assert "No new qualifying jobs" in digest.html
    assert "No new qualifying jobs" in digest.text


def test_delivery_state_is_persisted_atomically(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = mark_digest_sent(DeliveryState(), ["job-1"])
    save_delivery_state(path, state)

    assert load_delivery_state(path).sent == state.sent


def test_gmail_configuration_validation() -> None:
    assert validate_gmail_config("candidate@example.com", "app-password") == (
        "candidate@example.com",
        "app-password",
    )
    with pytest.raises(DigestError, match="GMAIL_ADDRESS"):
        validate_gmail_config("not-an-email", "app-password")
    with pytest.raises(DigestError, match="GMAIL_APP_PASSWORD"):
        validate_gmail_config("candidate@example.com", None)


def test_smtp_is_mocked_and_receives_credentials() -> None:
    calls: list[tuple[str, object]] = []

    class FakeSMTP:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            calls.append(("connect", (host, port, timeout)))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, address: str, password: str) -> None:
            calls.append(("login", (address, password)))

        def send_message(self, message: EmailMessage) -> None:
            calls.append(("send", message["Subject"]))

    digest = build_digest(MatchingReport(evaluations=[evaluation("Backend")]))
    send_gmail(
        digest,
        address="candidate@example.com",
        app_password="app-password",
        smtp_factory=FakeSMTP,
    )

    assert calls[0] == ("connect", ("smtp.gmail.com", 465, 30))
    assert calls[1] == ("login", ("candidate@example.com", "app-password"))
    assert calls[2][0] == "send"
