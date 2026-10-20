"""Domain models for jobs, candidates, and future matching results."""

from datetime import datetime

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
