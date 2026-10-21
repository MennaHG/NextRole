"""Domain models for jobs, candidates, and future matching results."""

from datetime import datetime
from typing import Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field


class JobPosting(BaseModel):
    """A normalized job posting with unknown fields left optional."""

    model_config = ConfigDict(extra="allow")

    title: str
    company: str | None = None
    url: AnyUrl | None = None
    location: str | None = None
    seniority: str | None = None
    description: str | None = None
    requirements: list[str] = Field(default_factory=list)
    work_arrangement: str | None = None
    employment_type: str | None = None
    date_posted: datetime | None = None
    expiration_date: datetime | None = None
    validity_status: Literal["valid", "invalid", "uncertain"] | None = None
    extraction_status: (
        Literal["success", "failed", "blocked", "skipped", "unavailable"] | None
    ) = None
    extraction_source: Literal["json_ld", "html", "unavailable"] | None = None
    resolved_url: AnyUrl | None = None
    extraction_error: str | None = None
    http_status: int | None = None
    source: str | None = None
    published_at: datetime | None = None
    discovered_at: datetime | None = None


class JobPreferences(BaseModel):
    """A candidate's search constraints, separate from career history."""

    roles: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    seniority: list[str] = Field(default_factory=list)
    remote: bool | None = None
    keywords: list[str] = Field(default_factory=list)
    max_age_hours: int | None = Field(default=24, ge=1)


class CandidateProfile(BaseModel):
    """Candidate preferences and career-history content."""

    name: str | None = None
    preferences: JobPreferences
    career_history: str
    technical_skills: list[str] = Field(default_factory=list)
    education: str | None = None


class JobMatch(BaseModel):
    """A future evaluation of how well a job matches a candidate."""

    job: JobPosting
    score: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None
    matched_experience: list[str] = Field(default_factory=list)


class RequirementAssessment(BaseModel):
    """Evidence-grounded assessment of one job requirement."""

    requirement: str
    importance: Literal["required", "preferred", "unspecified"]
    status: Literal["met", "partially_met", "not_met", "unknown"]
    evidence_relationship: Literal[
        "direct", "equivalent", "transferable", "absent", "contradictory"
    ] = "direct"
    candidate_evidence: list[str] = Field(default_factory=list)
    explanation: str


class EvaluationDecision(BaseModel):
    """Model-only decision fields; scoring is calculated after model output."""

    eligibility: Literal["eligible", "ineligible", "uncertain"]
    reason: str
    matching_strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    requirement_assessments: list[RequirementAssessment] = Field(default_factory=list)


class JobEvaluation(EvaluationDecision):
    """Structured semantic evaluation produced for one discovered job."""

    job: JobPosting
    demonstrated_fit: float = Field(ge=0, le=100)
    evidence_coverage: float = Field(ge=0, le=100)
    match_score: float = Field(ge=0, le=100)


class EvidenceDebugTrace(BaseModel):
    """Opt-in trace for inspecting one evaluation's evidence boundary."""

    job_title: str
    retrieved_chunks: list[str] = Field(default_factory=list)
    job_requirements: list[str] = Field(default_factory=list)
    requirement_assessments: list[RequirementAssessment] = Field(default_factory=list)
    eligibility: Literal["eligible", "ineligible", "uncertain"]
    demonstrated_fit: float = Field(ge=0, le=100)
    evidence_coverage: float = Field(ge=0, le=100)
    match_score: float = Field(ge=0, le=100)


class MatchingReport(BaseModel):
    """All evaluations and usage visibility for one matching run."""

    evaluations: list[JobEvaluation] = Field(default_factory=list)
    jobs_received: int = 0
    jobs_evaluated: int = 0
    duplicates_removed: int = 0
    llm_calls_attempted: int = 0
    embedding_documents: int = 0
    embedding_operations: int = 0
    mode: Literal["fixture", "live", "dry-run"] = "fixture"
    token_usage: dict[str, int] | None = None
    evidence_debug: list[EvidenceDebugTrace] | None = None
    skipped_jobs: list[dict[str, str]] = Field(default_factory=list)
