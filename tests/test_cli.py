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
