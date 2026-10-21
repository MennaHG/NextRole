from pathlib import Path

import pytest

from nextrole.cli import main
from nextrole.cv_generation import (
    CVGenerationError,
    generate_cv,
    save_cv,
    validate_cv,
)
from nextrole.enrichment import FetchedPage
from nextrole.models import (
    CandidateProfile,
    CVBullet,
    CVExperience,
    GeneratedCV,
    JobPreferences,
)

PROFILE = CandidateProfile(
    name="Ada Example",
    contact={"email": "ada@example.test", "linkedin": "https://linkedin.test/ada"},
    preferences=JobPreferences(roles=["Backend Engineer"], locations=["Remote"]),
    career_history=(
        "### Backend Engineer — Example Systems\n"
        "Jan 2024 – Present\n"
        "Built Python APIs that processed 10,000 events daily."
    ),
    technical_skills=["Python", "SQL", "Docker"],
    education="BSc in Computer Science.",
)

HTML = """
<html><head><script type="application/ld+json">
{"@type":"JobPosting","title":"Backend Engineer","hiringOrganization":{"name":"Acme"},
"description":"Build reliable Python APIs and data services for customers.",
"qualifications":"Python and API experience","responsibilities":"Own backend services"}
</script></head><body></body></html>
"""


class FakeFetcher:
    def fetch(self, url: str) -> FetchedPage:
        return FetchedPage(url, url, HTML, 200, "text/html")


class FakeChat:
    def with_structured_output(self, schema: type[GeneratedCV]) -> "FakeChat":
        assert schema is GeneratedCV
        return self

    def invoke(self, prompt: str) -> dict:
        assert "Backend Engineer" in prompt
        return {
            "headline": "Backend Engineer",
            "summary": "Backend engineer focused on Python APIs.",
            "contact": {"name": "Invented Name", "email": "bad@example.test"},
            "experience": [{
                "company": "Invented Company",
                "job_title": "Invented Title",
                "bullets": [{
                    "text": "Built Python APIs that processed 10,000 events daily.",
                    "evidence_references": [
                        "Built Python APIs that processed 10,000 events daily."
                    ],
                }],
            }],
            "technical_skills": {"Languages": ["Python", "SQL"]},
            "projects": [],
            "education": "Invented education",
        }


def test_generation_uses_one_mocked_model_call_and_profile_facts() -> None:
    cv, job = generate_cv(
        "https://jobs.test/backend",
        PROFILE,
        fetcher=FakeFetcher(),
        chat_model=FakeChat(),
    )

    assert job.company == "Acme"
    assert cv.contact.email == "ada@example.test"
    assert cv.education == PROFILE.education
    assert cv.experience[0].company == "Example Systems"
    assert cv.experience[0].job_title == "Backend Engineer"
    assert cv.experience[0].employment_dates == "Jan 2024 – Present"
    assert len(cv.experience[0].bullets) == 1
    assert cv.experience[0].bullets[0].evidence_references
    assert any(
        "Missing contact field: phone" in warning
        for warning in cv.validation_warnings
    )


def test_json_is_serialized(tmp_path: Path) -> None:
    cv, job = generate_cv(
        "https://jobs.test/backend",
        PROFILE,
        fetcher=FakeFetcher(),
        chat_model=FakeChat(),
    )
    output = save_cv(cv, job, tmp_path)

    assert output.name == "cv.json"
    assert '"headline": "Backend Engineer"' in output.read_text()


def test_validation_flags_missing_references_metrics_and_unsupported_claims() -> None:
    cv = GeneratedCV(
        headline="Backend Engineer",
        contact={},
        experience=[
            CVExperience(
                company="Example Systems",
                job_title="Backend Engineer",
                bullets=[
                    CVBullet(
                        text="Implemented Kubernetes and reduced costs by 99%.",
                        evidence_references=[
                            "Built Python APIs that processed 10,000 events daily."
                        ],
                    ),
                    CVBullet(text="Unreferenced claim"),
                ],
            )
        ],
        technical_skills={"Languages": ["Python"]},
        education=PROFILE.education or "",
    )

    warnings = validate_cv(cv, PROFILE)
    assert any("Unsupported claim term: Kubernetes" in item for item in warnings)
    assert any("99%" in item for item in warnings)
    assert any("Missing evidence reference" in item for item in warnings)


def test_empty_skills_are_rejected() -> None:
    class EmptySkillsChat(FakeChat):
        def invoke(self, prompt: str) -> dict:
            response = super().invoke(prompt)
            response["technical_skills"] = {}
            return response

    with pytest.raises(CVGenerationError, match="technical skills are empty"):
        generate_cv(
            "https://jobs.test/backend",
            PROFILE,
            fetcher=FakeFetcher(),
            chat_model=EmptySkillsChat(),
        )


def test_cli_generate_cv_execution_is_mockable(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "cv.json"

    def fake_generate(url: str, profile: Path, **_: object):
        assert url == "https://jobs.test/backend"
        assert profile == tmp_path / "profile.md"
        return GeneratedCV(
            headline="Backend Engineer",
            contact={},
            technical_skills={"Languages": ["Python"]},
            education="BSc",
        ), type("Job", (), {"company": "Acme", "title": "Backend Engineer"})()

    monkeypatch.setattr("nextrole.cli.generate_cv_from_profile", fake_generate)
    monkeypatch.setattr("nextrole.cli.save_cv", lambda cv, job, root: output)
    assert main(
        [
            "generate-cv",
            "--url",
            "https://jobs.test/backend",
            "--profile",
            str(tmp_path / "profile.md"),
        ]
    ) == 0
    assert "Output file:" in capsys.readouterr().out
