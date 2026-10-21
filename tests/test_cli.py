import pytest

from nextrole.cli import main


def test_help_command(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "check" in capsys.readouterr().out


def test_check_works_without_credentials(capsys) -> None:
    assert main(["check"]) == 0
    assert "Configuration is valid" in capsys.readouterr().out


def test_profile_validate_and_discover_dry_run(tmp_path, capsys) -> None:
    profile = tmp_path / "profile.md"
    profile.write_text(
        """## Preferences
- Roles: Backend Engineer
- Locations: Cairo
- Remote: Yes

## Career history
Built fictional services.

## Technical skills
Python, SQL

## Education
BSc in Computer Science.
""",
        encoding="utf-8",
    )

    assert main(["profile", "validate", "--file", str(profile)]) == 0
    assert "Profile is valid" in capsys.readouterr().out

    assert main(["discover", "--profile", str(profile), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Query source: profile" in output
    assert "Backend Engineer Cairo jobs" in output
    assert "no API call" in output


def test_manual_queries_override_profile_in_dry_run(tmp_path, capsys) -> None:
    profile = tmp_path / "profile.md"
    profile.write_text(
        """## Preferences
- Roles: Backend Engineer
- Locations: Cairo

## Career history
Built fictional services.

## Technical skills
Python

## Education
BSc in Computer Science.
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "discover",
                "--profile",
                str(profile),
                "--query",
                "custom query",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Query source: manual" in output
    assert "custom query" in output
    assert "Backend Engineer Cairo jobs" not in output


def test_configured_profile_path_is_used_for_discovery(tmp_path, monkeypatch, capsys):
    profile = tmp_path / "profile.md"
    profile.write_text(
        """## Preferences
- Roles: Data Engineer
- Locations: Alexandria

## Career history
Built fictional data services.

## Technical skills
Python, SQL

## Education
BSc in Computer Science.
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEXTROLE_PROFILE_PATH", str(profile))

    assert main(["discover", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Query source: profile" in output
    assert "Data Engineer Alexandria jobs" in output
