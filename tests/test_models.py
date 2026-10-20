from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nextrole.models import (
    CandidateProfile,
    JobMatch,
    JobPosting,
    JobPreferences,
)


def test_job_posting_keeps_unknown_fields_and_optional_data() -> None:
    job = JobPosting(title="Backend Engineer", extra_field="from-source")

    assert job.company is None
    assert job.model_extra == {"extra_field": "from-source"}


def test_candidate_profile_separates_preferences_from_history() -> None:
    profile = CandidateProfile(
        name="Ada Example",
        preferences=JobPreferences(roles=["Backend Engineer"]),
        career_history="Built fictional Python services.",
    )

    assert profile.preferences.roles == ["Backend Engineer"]
    assert profile.career_history.startswith("Built")


def test_job_match_validates_score_range() -> None:
    job = JobPosting(
        title="Engineer",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    match = JobMatch(job=job, score=0.8)
    assert match.score == 0.8

    with pytest.raises(ValidationError):
        JobMatch(job=job, score=1.1)
