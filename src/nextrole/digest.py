"""Deterministic email digest generation and Gmail SMTP delivery."""

from __future__ import annotations

import hashlib
import html
import re
import smtplib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .models import JobEvaluation, MatchingReport

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


class DigestError(RuntimeError):
    """Raised when a digest cannot be generated or delivered."""


class DeliveryState(BaseModel):
    """Previously submitted job identities and their submission timestamps."""

    sent: dict[str, str] = {}


@dataclass(frozen=True)
class Digest:
    """A selected digest and its two email representations."""

    subject: str
    html: str
    text: str
    evaluations: list[JobEvaluation]
    identities: list[str]


def load_matching_report(path: Path) -> MatchingReport:
    """Load a JSON report produced by the matching command."""
    try:
        return MatchingReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as error:
        raise DigestError(f"Could not read matching report: {path}") from error


def load_delivery_state(path: Path) -> DeliveryState:
    """Read delivery state, treating a missing state file as empty."""
    if not path.exists():
        return DeliveryState()
    try:
        return DeliveryState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as error:
        raise DigestError(f"Could not read digest delivery state: {path}") from error


def save_delivery_state(path: Path, state: DeliveryState) -> None:
    """Persist delivery state atomically after successful SMTP submission."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(
            state.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(path)
    except OSError as error:
        raise DigestError(f"Could not write digest delivery state: {path}") from error


def job_identity(evaluation: JobEvaluation) -> str:
    """Return a stable identity based on the application URL and job metadata."""
    url = str(evaluation.job.resolved_url or evaluation.job.url or "").strip().lower()
    raw = url or "|".join(
        (
            evaluation.job.title.strip().lower(),
            (evaluation.job.company or "").strip().lower(),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def select_digest_jobs(
    report: MatchingReport,
    *,
    min_score: float,
    max_jobs: int,
    sent_ids: set[str] | None = None,
) -> list[JobEvaluation]:
    """Filter ranked evaluations into new, validated, linkable recommendations."""
    if min_score < 0 or min_score > 100:
        raise DigestError("Minimum digest score must be between 0 and 100.")
    if max_jobs < 1:
        raise DigestError("Digest maximum jobs must be at least 1.")
    sent_ids = sent_ids or set()
    selected: list[JobEvaluation] = []
    seen: set[str] = set()
    for evaluation in sorted(
        report.evaluations,
        key=lambda item: (
            {"eligible": 0, "uncertain": 1, "ineligible": 2}[item.eligibility],
            -item.match_score,
        ),
    ):
        job = evaluation.job
        identity = job_identity(evaluation)
        if (
            job.validity_status != "valid"
            or evaluation.eligibility == "ineligible"
            or evaluation.match_score < min_score
            or not (job.resolved_url or job.url)
            or identity in sent_ids
            or identity in seen
        ):
            continue
        seen.add(identity)
        selected.append(evaluation)
        if len(selected) >= max_jobs:
            break
    return selected


def build_digest(
    report: MatchingReport,
    *,
    min_score: float = 0,
    max_jobs: int = 10,
    sent_ids: set[str] | None = None,
) -> Digest:
    """Build escaped HTML and plain text without any provider calls."""
    evaluations = select_digest_jobs(
        report, min_score=min_score, max_jobs=max_jobs, sent_ids=sent_ids
    )
    identities = [job_identity(item) for item in evaluations]
    subject = f"NextRole job digest — {len(evaluations)} new match"
    if len(evaluations) != 1:
        subject += "es"
    return Digest(
        subject=subject,
        html=_render_html(evaluations),
        text=_render_text(evaluations),
        evaluations=evaluations,
        identities=identities,
    )


def _render_html(evaluations: list[JobEvaluation]) -> str:
    cards = "".join(_render_html_job(item) for item in evaluations)
    empty = "<p>No new qualifying jobs were found.</p>" if not cards else ""
    return f"""<!doctype html>
<html><body style="margin:0;background:#f4f6f8;color:#17202a;">
<main style="max-width:680px;margin:auto;padding:20px;">
<h1 style="font-size:24px;">NextRole job digest</h1>
{empty}{cards}
</main></body></html>"""


def _render_html_job(evaluation: JobEvaluation) -> str:
    job = evaluation.job
    title = html.escape(job.title)
    company = html.escape(job.company or "Company not specified")
    location = html.escape(job.location or "Location not specified")
    arrangement = html.escape(job.work_arrangement or "Arrangement not specified")
    url = html.escape(str(job.resolved_url or job.url), quote=True)
    strengths = _html_list(
        evaluation.matching_strengths, "No specific strengths recorded."
    )
    missing = _requirement_items(evaluation, "not_met")
    unknown = _requirement_items(evaluation, "unknown")
    gaps = _html_list(missing, "None explicitly identified.")
    uncertainties = _html_list(unknown, "None identified.")
    return f"""<article style="background:#fff;border-radius:10px;margin:16px 0;
padding:18px;">
<h2 style="margin-top:0;font-size:19px;">{title}</h2>
<p><strong>{company}</strong><br>{location} · {arrangement}</p>
<p><strong>Demonstrated fit:</strong> {evaluation.demonstrated_fit:.2f}/100
 · <strong>Evidence coverage:</strong> {evaluation.evidence_coverage:.2f}/100
 · <strong>Eligibility:</strong> {html.escape(evaluation.eligibility)}</p>
<h3>Why it matches</h3>{strengths}
<h3>Requirements explicitly not met</h3>{gaps}
<h3>Requirements still unverified</h3>{uncertainties}
<p><a href="{url}">Apply for this job</a></p>
</article>"""


def _html_list(items: list[str], empty: str) -> str:
    values = items or [empty]
    return (
        "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in values) + "</ul>"
    )


def _requirement_items(evaluation: JobEvaluation, status: str) -> list[str]:
    return [
        assessment.requirement
        for assessment in evaluation.requirement_assessments
        if assessment.status == status
    ]


def _render_text(evaluations: list[JobEvaluation]) -> str:
    if not evaluations:
        return "NextRole job digest\n\nNo new qualifying jobs were found.\n"
    blocks = ["NextRole job digest", ""]
    for index, evaluation in enumerate(evaluations, start=1):
        job = evaluation.job
        blocks.extend(
            [
                f"{index}. {job.title}",
                f"Company: {job.company or 'Company not specified'}",
                f"Location: {job.location or 'Location not specified'}",
                f"Arrangement: {job.work_arrangement or 'Arrangement not specified'}",
                f"Demonstrated fit: {evaluation.demonstrated_fit:.2f}/100",
                f"Evidence coverage: {evaluation.evidence_coverage:.2f}/100",
                f"Eligibility: {evaluation.eligibility}",
                "Strengths: " + _text_values(evaluation.matching_strengths),
                "Not met: " + _text_values(_requirement_items(evaluation, "not_met")),
                "Unverified: "
                + _text_values(_requirement_items(evaluation, "unknown")),
                f"Apply: {job.resolved_url or job.url}",
                "",
            ]
        )
    return "\n".join(blocks)


def _text_values(values: list[str]) -> str:
    return "; ".join(values) if values else "None identified"


def create_message(digest: Digest, address: str) -> EmailMessage:
    """Create a multipart email with text and HTML alternatives."""
    message = EmailMessage()
    message["From"] = address
    message["To"] = address
    message["Subject"] = digest.subject
    message.set_content(digest.text)
    message.add_alternative(digest.html, subtype="html")
    return message


def validate_gmail_config(
    address: str | None, app_password: str | None
) -> tuple[str, str]:
    """Validate delivery credentials before opening an SMTP connection."""
    if (
        not address
        or parseaddr(address)[1] != address
        or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address)
    ):
        raise DigestError("GMAIL_ADDRESS must be a valid email address.")
    if not app_password:
        raise DigestError("GMAIL_APP_PASSWORD is required for delivery.")
    return address, app_password


def send_gmail(
    digest: Digest,
    *,
    address: str | None,
    app_password: str | None,
    smtp_factory: Callable[..., Any] = smtplib.SMTP_SSL,
) -> None:
    """Submit one digest through Gmail SMTP over SSL."""
    sender, password = validate_gmail_config(address, app_password)
    message = create_message(digest, sender)
    try:
        with smtp_factory(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            smtp.login(sender, password)
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as error:
        raise DigestError(f"Gmail SMTP delivery failed: {error}") from error


def mark_digest_sent(state: DeliveryState, identities: list[str]) -> DeliveryState:
    """Return state updated after SMTP confirms message submission."""
    timestamp = datetime.now(UTC).isoformat()
    updated = state.model_copy(deep=True)
    for identity in identities:
        updated.sent[identity] = timestamp
    return updated
