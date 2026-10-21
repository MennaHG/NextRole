"""Simple, private Markdown candidate-profile loading and query generation."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from .models import CandidateProfile, JobPreferences


class ProfileError(ValueError):
    """A profile could not be loaded or did not follow the supported format."""


_SECTION_NAMES = {
    "contact": "contact",
    "contact info": "contact",
    "contact info & links": "contact",
    "contact information": "contact",
    "preferences": "preferences",
    "career history": "career_history",
    "technical skills": "technical_skills",
    "education": "education",
}


def load_profile(path: Path) -> CandidateProfile:
    """Load and validate a local Markdown profile without logging its contents."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ProfileError(f"Profile file not found: {path}") from error
    except OSError as error:
        raise ProfileError(f"Could not read profile file: {path}") from error
    try:
        return parse_profile(text)
    except ProfileError as error:
        raise ProfileError(f"Invalid profile {path}: {error}") from error


def parse_profile(markdown: str) -> CandidateProfile:
    """Parse the documented four-section Markdown profile format."""
    sections = _sections(markdown)
    required_sections = (
        "preferences",
        "career_history",
        "technical_skills",
        "education",
    )
    missing = [name for name in required_sections if name not in sections]
    if missing:
        labels = ", ".join(name.replace("_", " ") for name in missing)
        raise ProfileError(f"missing required section(s): {labels}")

    values = _preference_values(sections["preferences"])
    required_preferences = [key for key in ("roles", "locations") if key not in values]
    if required_preferences:
        raise ProfileError(
            "preferences must define: " + ", ".join(required_preferences)
        )

    try:
        preferences = JobPreferences(
            roles=_list_value(values["roles"]),
            locations=_list_value(values["locations"]),
            seniority=_list_value(values.get("seniority", "")),
            remote=_bool_value(values.get("remote")),
            keywords=_list_value(values.get("keywords", "")),
            max_age_hours=_age_value(values.get("maximum job age", "24")),
        )
    except (KeyError, ValueError, ValidationError) as error:
        raise ProfileError(f"invalid preference value: {error}") from error

    career_history = sections["career_history"].strip()
    if not career_history:
        raise ProfileError("career history must not be empty")
    technical_skills = _content_list(sections["technical_skills"])
    if not technical_skills:
        raise ProfileError("technical skills must not be empty")
    education = sections["education"].strip()
    if not education:
        raise ProfileError("education must not be empty")

    return CandidateProfile(
        name=_preference_values(sections.get("contact", "")).get("name"),
        contact=_preference_values(sections.get("contact", "")),
        preferences=preferences,
        career_history=career_history,
        technical_skills=technical_skills,
        education=education,
    )


def generate_queries(profile: CandidateProfile, *, max_queries: int = 3) -> list[str]:
    """Generate stable, broad-to-specific search queries from profile signals."""
    if max_queries < 1:
        return []
    preferences = profile.preferences
    locations = [
        value for value in preferences.locations if value.casefold() != "remote"
    ] or preferences.locations
    location = locations[0] if locations else ""
    roles = preferences.roles
    technologies = preferences.keywords or profile.technical_skills
    queries: list[str] = []

    def add(role: str, technology: str | None = None) -> None:
        parts = [role]
        if technology:
            parts.append(technology)
        if location:
            parts.append(location)
        parts.append("jobs")
        query = " ".join(part.strip() for part in parts if part.strip())
        if query and query not in queries:
            queries.append(query)

    add(roles[0])
    if technologies:
        add(roles[0], technologies[0])
    second_role = roles[1] if len(roles) > 1 else roles[0]
    second_technology = technologies[1] if len(technologies) > 1 else None
    add(second_role, second_technology)
    return queries[:max_queries]


def _sections(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    content: list[str] = []
    for line in markdown.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            if current is not None:
                sections[current] = "\n".join(content).strip()
            current = _SECTION_NAMES.get(heading.group(1).casefold())
            content = []
            continue
        if current is not None:
            content.append(line)
    if current is not None:
        sections[current] = "\n".join(content).strip()
    return sections


def _preference_values(section: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in section.splitlines():
        match = re.match(r"^\s*[-*]\s*([^:]+):\s*(.*?)\s*$", line)
        if match:
            values[match.group(1).strip().casefold()] = match.group(2).strip()
    return values


def _list_value(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]


def _content_list(section: str) -> list[str]:
    values: list[str] = []
    for line in section.splitlines():
        value = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if value:
            values.extend(_list_value(value))
    return values


def _bool_value(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().casefold()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ValueError("remote must be yes or no")


def _age_value(value: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise ValueError("maximum job age must contain hours")
    return int(match.group())
