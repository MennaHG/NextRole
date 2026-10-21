"""On-demand, evidence-grounded CV generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import Settings
from .enrichment import HttpxPageFetcher, extract_job
from .matching import _create_chat
from .models import (
    CandidateProfile,
    CVContact,
    GeneratedCV,
    JobPosting,
)
from .profile import ProfileError, load_profile


class CVGenerationError(RuntimeError):
    """A job page, model response, or evidence validation failed."""


def generate_cv(
    url: str,
    profile: CandidateProfile,
    *,
    chat_model: Any | None = None,
    settings: Settings | None = None,
    fetcher: Any | None = None,
) -> tuple[GeneratedCV, JobPosting]:
    """Fetch one posting and generate one structured CV with one model call."""
    if not urlsplit(url).scheme or not urlsplit(url).netloc:
        raise CVGenerationError("URL is not a valid HTTP(S) address")
    owned_fetcher = fetcher is None
    if owned_fetcher:
        fetcher = HttpxPageFetcher(
            timeout=(settings.enrichment_timeout_seconds if settings else 15.0),
            max_bytes=(settings.enrichment_max_bytes if settings else 2_000_000),
        )
    try:
        try:
            page = fetcher.fetch(url)
        except Exception as error:
            raise CVGenerationError(f"could not retrieve job page: {error}") from error
        extracted = extract_job(page.html)
        if not extracted.description or extracted.validity_status == "invalid":
            raise CVGenerationError(
                "job posting extraction was insufficient: "
                f"{extracted.error or 'missing description'}"
            )
        job = JobPosting(
            title=extracted.title or "Unknown title",
            company=extracted.company,
            url=url,
            description=extracted.description,
            requirements=extracted.requirements,
            location=extracted.location,
            extraction_status="success",
            extraction_source=extracted.source,
            validity_status=extracted.validity_status,
        )
        if chat_model is None:
            if settings is None:
                raise CVGenerationError("LLM settings are required for live generation")
            if not settings.llm_provider or not settings.llm_model:
                raise CVGenerationError(
                    "Set NEXTROLE_LLM_PROVIDER and NEXTROLE_LLM_MODEL"
                )
            key = (
                settings.llm_api_key.get_secret_value()
                if settings.llm_api_key
                else None
            )
            if not key and settings.llm_provider.casefold() != "ollama":
                raise CVGenerationError("Set LLM_API_KEY for live CV generation")
            chat_model = _create_chat(settings.llm_provider, settings.llm_model, key)
        try:
            try:
                structured = chat_model.with_structured_output(
                    GeneratedCV, method="function_calling"
                )
            except TypeError:
                structured = chat_model.with_structured_output(GeneratedCV)
            response = structured.invoke(_prompt(profile, job))
            cv = GeneratedCV.model_validate(response)
        except Exception as error:
            raise CVGenerationError(f"LLM returned invalid CV data: {error}") from error
        # Contact details and education are immutable profile facts, never model prose.
        normalized_skills = {
            category: skills
            for category, skills in cv.technical_skills.items()
            if skills
        }
        if not normalized_skills:
            raise CVGenerationError(
                "Generated CV validation failed: technical skills are empty"
            )
        cv = cv.model_copy(
            update={
                "contact": _profile_contact(profile),
                "education": profile.education,
                "experience": _preserve_experience_metadata(cv.experience, profile),
                "summary": None,
                "technical_skills": normalized_skills,
            }
        )
        cv = cv.model_copy(update={"validation_warnings": validate_cv(cv, profile)})
        return cv, job
    finally:
        if owned_fetcher:
            fetcher.__exit__(None, None, None)


def generate_cv_from_profile(
    url: str, profile_path: Path, **kwargs: Any
) -> tuple[GeneratedCV, JobPosting]:
    try:
        profile = load_profile(profile_path)
    except ProfileError as error:
        raise CVGenerationError(str(error)) from error
    return generate_cv(url, profile, **kwargs)


def validate_cv(cv: GeneratedCV, profile: CandidateProfile) -> list[str]:
    """Return warnings for claims that cannot be proven by profile text."""
    source = "\n".join(
        (profile.career_history, *profile.technical_skills, profile.education or "")
    ).casefold()
    warnings: list[str] = []
    for field in ("email", "phone", "linkedin", "github"):
        if not profile.contact.get(field):
            warnings.append(f"Missing contact field: {field}")
    if not cv.technical_skills:
        warnings.append("Technical skills section is unexpectedly empty")
    for experience in cv.experience:
        metadata_supported = _same_words(experience.company, source) and _same_words(
            experience.job_title, source
        )
        if not metadata_supported:
            warnings.append(
                "Experience metadata is not traceable: "
                f"{experience.job_title} at {experience.company}"
            )
        if not experience.bullets:
            warnings.append(f"Experience has no bullets: {experience.job_title}")
        for bullet in experience.bullets:
            if not bullet.evidence_references:
                warnings.append(f"Missing evidence reference: {bullet.text}")
            for reference in bullet.evidence_references:
                if reference.casefold() not in source:
                    warnings.append(f"Unsupported evidence reference: {reference}")
            for number in _numbers(bullet.text):
                if number not in _numbers(source):
                    warnings.append(
                        f"Numerical claim is not found in profile: {number}"
                    )
            for term in _capitalized_terms(bullet.text):
                if term.casefold() not in source:
                    warnings.append(f"Unsupported claim term: {term}")
    supported = {item.casefold() for item in profile.technical_skills}
    for _category, skills in cv.technical_skills.items():
        for skill in skills:
            if skill.casefold() not in supported and skill.casefold() not in source:
                warnings.append(f"Unsupported skill: {skill}")
    for project in cv.projects:
        for reference in project.evidence_references:
            if reference.casefold() not in source:
                warnings.append(f"Unsupported evidence reference: {reference}")
    if not cv.education.strip():
        warnings.append("Education is empty")
    return list(dict.fromkeys(warnings))


def _profile_contact(profile: CandidateProfile) -> CVContact:
    values = {key.casefold(): value for key, value in profile.contact.items()}
    return CVContact(
        name=profile.name or values.get("name"),
        email=values.get("email"),
        phone=values.get("phone"),
        linkedin=values.get("linkedin") or values.get("linkedin url"),
        github=values.get("github") or values.get("github url"),
    )


def _preserve_experience_metadata(
    experiences: list[Any], profile: CandidateProfile
) -> list[Any]:
    """Replace model-written metadata when the profile has recognizable headings."""
    metadata: list[tuple[str, str, str | None, str | None]] = []
    lines = profile.career_history.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^\s*#{3,}\s+(.+?)\s+[—–-]\s+(.+?)\s*$", line)
        if match:
            heading_text = match.group(0).split(None, 1)[1]
            parts = [
                part.strip() for part in re.split(r"\s+[—–-]\s+", heading_text)
            ]
            title = parts[0]
            company = parts[1]
            location = parts[2] if len(parts) > 2 else None
            dates = next(
                (
                    value.strip()
                    for value in lines[index + 1 :]
                    if value.strip() and _looks_like_dates(value)
                ),
                None,
            )
            metadata.append((title, company, location, dates))
    if not metadata:
        return experiences
    preserved: list[Any] = []
    for index, experience in enumerate(experiences):
        if index >= len(metadata):
            preserved.append(experience)
            continue
        title, company, location, dates = metadata[index]
        preserved.append(
            experience.model_copy(
                update={
                    "job_title": title,
                    "company": company,
                    "location": location,
                    "employment_dates": dates,
                }
            )
        )
    return preserved


def save_cv(
    cv: GeneratedCV, job: JobPosting, output_root: Path = Path("outputs/cvs")
) -> Path:
    company = _slug(job.company or "unknown-company")
    title = _slug(job.title or "unknown-role")
    directory = output_root / f"{company}_{title}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "cv.json"
    if path.exists():
        path = directory / "cv-2.json"
        index = 2
        while path.exists():
            index += 1
            path = directory / f"cv-{index}.json"
    path.write_text(cv.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _prompt(profile: CandidateProfile, job: JobPosting) -> str:
    return f"""You are an expert technical resume writer specializing in ATS-friendly
resumes for software engineering positions.

Your task is to generate a tailored, professional CV using the provided job
description and candidate profile.

Your objective is to maximize relevance to the target role while accurately
representing the candidate's real experience.

## 1. Analyze the Target Job

Identify the primary responsibilities, engineering challenges, required and
preferred technologies, architectural concepts, domain experience, and relative
importance of each requirement. Prioritize required qualifications and core
responsibilities. Treat the job description as a relevance guide, NOT as
evidence of the candidate's capabilities.

## 2. CV Structure and Experience

Return sections in this order: Header (name, headline, and contact
information), Professional Experience, Technical Skills, optional Projects,
and Education. Omit the professional summary by default.

Professional experience must be reverse chronological. Aim for 3-5 relevant
bullets for the most recent role and 2-3 for earlier roles. Each bullet should
ideally follow Action + Technical Implementation + Impact when supported by
evidence.

## 3. Select and Rewrite Experience

Evaluate ALL achievements in the candidate profile. Prioritize technical depth,
ownership, measurable impact, direct experience, and transferable relevance.
Use 3-5 bullets for the most relevant position and 2-3 for other relevant
positions as guidelines, but prioritize strong evidence over filling quotas.

Rewrite selected achievements to explain what was built, the technical approach,
and the impact when supported. Use precise action verbs and natural job
terminology. Avoid generic descriptions, unsupported buzzwords, repetition,
exaggerated ownership, and long sentences. Prefer concise 1-2 line bullets.

## 4. Technical Skills

Use ONLY skills supported by the candidate profile. Prioritize job-required
skills present in the profile, then technologies demonstrated in experience and
relevant complementary skills. Organize them into clear categories such as
Languages, Backend & Frameworks, Databases, Cloud & DevOps, Architecture &
Messaging, and Frontend. Use only categories containing relevant skills. Do not
add technologies merely because they appear in the job description.

## 5. Projects, Headline, and Summary

Include projects only when they strengthen the application; an empty projects
section is acceptable. Generate a concise headline reflecting actual experience,
without automatically adopting the job's seniority or adding unsupported
specializations. Omit the professional summary by default.

## 6. ATS Optimization and Factual Accuracy

Use conventional ATS-readable section names and naturally incorporate supported
job keywords. The candidate profile is the ONLY source of truth.

NEVER invent technologies, achievements, responsibilities, projects, metrics,
employment dates, company names, job titles, education, contact information, or
seniority. Do not turn transferable experience into direct experience. Every
generated experience bullet must include its own exact supporting evidence
reference or references from the candidate profile. If evidence is
insufficient, omit the claim.

## 7. Output Requirements

Return only the requested structured output. Generate a complete CV suitable for
a concise, preferably one-page software engineering CV. Do not add explanations,
reasoning, or commentary outside the structured output.

<job>
{job.model_dump_json(exclude_none=True)}
</job>

<profile>
{profile.model_dump_json(exclude_none=True)}
</profile>
"""


def _same_words(value: str, source: str) -> bool:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    return bool(words) and all(word in source for word in words)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "unknown"


def _numbers(value: str) -> set[str]:
    return {
        item.replace(",", "")
        for item in re.findall(r"\b\d[\d,.]*%?", value.casefold())
    }


def _looks_like_dates(value: str) -> bool:
    return bool(
        re.search(
            r"\b(?:19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
            value,
            re.I,
        )
    )


def _capitalized_terms(value: str) -> set[str]:
    words = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)?\b", value)
    return {word for word in words[1:]}
