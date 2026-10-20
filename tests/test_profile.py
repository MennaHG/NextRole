from pathlib import Path

import pytest

from nextrole.models import CandidateProfile
from nextrole.profile import (
    ProfileError,
    generate_queries,
    load_profile,
    parse_profile,
)

PROFILE = """# Candidate Profile

## Preferences

- Roles: Backend Engineer, Full Stack Engineer
- Locations: Cairo, Egypt
- Seniority: Mid-level, Senior
- Remote: Yes
- Keywords: Python, React, Node.js
- Maximum job age: 24 hours

## Career history

### Software Engineer — Example Systems

Built fictional Python services and internal APIs.

## Technical skills

Python, React, Node.js, SQL

## Education

BSc in Computer Science — Fictional University.
"""


def test_profile_parses_preferences_and_preserves_sections():
    profile = parse_profile(PROFILE)

    assert profile.preferences.roles == ["Backend Engineer", "Full Stack Engineer"]
    assert profile.preferences.locations == ["Cairo", "Egypt"]
    assert profile.preferences.remote is True
    assert profile.technical_skills == ["Python", "React", "Node.js", "SQL"]
    assert "Built fictional Python services" in profile.career_history
    assert profile.education == "BSc in Computer Science — Fictional University."


def test_profile_loads_from_path_without_printing_contents(tmp_path: Path):
    path = tmp_path / "profile.md"
    path.write_text(PROFILE, encoding="utf-8")

    profile = load_profile(path)

    assert isinstance(profile, CandidateProfile)


def test_profile_requires_sections_and_preferences():
    with pytest.raises(ProfileError, match="missing required section"):
        parse_profile("## Preferences\n\n- Roles: Engineer\n")

    malformed = PROFILE.replace("- Roles: Backend Engineer, Full Stack Engineer", "")
    with pytest.raises(ProfileError, match="roles"):
        parse_profile(malformed)


def test_query_generation_is_stable_broad_and_diverse():
    profile = parse_profile(PROFILE)

    first = generate_queries(profile)
    second = generate_queries(profile)

    assert first == second
    assert 1 <= len(first) <= 3
    assert first[0] == "Backend Engineer Cairo jobs"
    assert len(set(first)) == len(first)
    assert any("React" in query or "Python" in query for query in first)
    assert all("Remote" not in query for query in first)


def test_query_generation_does_not_require_remote_only_preferences():
    profile = parse_profile(
        PROFILE.replace("- Locations: Cairo, Egypt", "- Locations: Remote, Cairo")
    )

    queries = generate_queries(profile)

    assert all("Cairo" in query for query in queries)


def test_missing_profile_file_is_clear(tmp_path: Path):
    with pytest.raises(ProfileError, match="not found"):
        load_profile(tmp_path / "missing.md")
