import os

import pytest

from humanoid_companion import cli, settings
from humanoid_companion.teammates import TEAMMATES, TeammateError, all_teammates, teammate_from_file, teammate_template


@pytest.fixture
def config(tmp_path, monkeypatch):
    """An empty settings folder, and no settings leaking in from the environment."""
    monkeypatch.setenv("HUMANOID_CONFIG_DIR", str(tmp_path))
    for variable in settings.ENVIRONMENT.values():
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def test_settings_fill_in_unset_variables_and_never_override_exported_ones(config, monkeypatch):
    (config / "settings.toml").write_text(
        '[user]\nname = "Alex"\n[llm]\nbase_url = "http://127.0.0.1:11500/v1"\nmodel = ""\n'
        '[voice]\nvoice = "af_sky"\n[songs]\ntempo = "~/my-songs"\n'
    )
    monkeypatch.setenv("HUMANOID_TTS_VOICE", "af_heart")
    applied = settings.apply()
    assert os.environ["HUMANOID_USER_NAME"] == "Alex" and os.environ["HUMANOID_LLM_BASE_URL"].endswith(":11500/v1")
    assert os.environ["HUMANOID_TTS_VOICE"] == "af_heart"  # exported: kept
    assert "HUMANOID_LLM_MODEL" not in os.environ  # empty values are not settings
    assert sorted(applied) == ["HUMANOID_LLM_BASE_URL", "HUMANOID_USER_NAME"]
    assert settings.song_library("tempo").name == "my-songs" and settings.song_library("byte") is None


def test_without_a_settings_file_nothing_changes_and_a_broken_one_says_where(config):
    assert settings.load() == {} and settings.apply() == []
    (config / "settings.toml").write_text("[user\nname = 1")
    with pytest.raises(settings.SettingsError, match="settings.toml"):
        settings.load()


def test_init_writes_a_template_once(config):
    path = settings.write_template()
    assert path.exists() and "[llm]" in path.read_text() and "API keys" in path.read_text()
    with pytest.raises(settings.SettingsError, match="already exists"):
        settings.write_template()


def test_a_teammate_file_starts_from_a_built_in_and_overrides_it(config):
    folder = config / "teammates"
    folder.mkdir()
    (folder / "nova.toml").write_text(
        'name = "Nova"\nbased_on = "tempo"\naccessory = "none"\ndances = false\n'
        'role = "Your role: you tell short, true stories about space."\n[colours]\nglow = "#9fd0ff"\n'
    )
    mates = all_teammates()
    assert set(mates) == {"byte", "tempo", "nova"}
    nova = mates["nova"]
    assert (nova.name, nova.accessory, nova.dances, nova.voice) == ("Nova", "none", False, TEAMMATES["tempo"].voice)
    assert nova.look.glow == (159, 208, 255) and nova.look.background == TEAMMATES["tempo"].look.background
    assert nova.source.endswith("nova.toml") and "stories about space" in nova.persona("")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('accessory = "hat"', "accessory"),
        ('resting_expression = "angry"', "resting_expression"),
        ('based_on = "nobody"', "based_on"),
        ('[colours]\nglow = "blue"', "colours.glow"),
        ('role = "  "', "role"),
    ],
)
def test_a_wrong_teammate_file_names_the_file_and_the_field(tmp_path, content, message):
    path = tmp_path / "oops.toml"
    path.write_text(content)
    with pytest.raises(TeammateError, match=message) as error:
        teammate_from_file(path)
    assert "oops.toml" in str(error.value)


def test_the_template_is_a_valid_teammate_file(tmp_path):
    path = tmp_path / "echo.toml"
    path.write_text(teammate_template("echo", "tempo"))
    echo = teammate_from_file(path)
    assert echo.name == "Echo" and echo.dances and echo.look.glow == TEAMMATES["tempo"].look.glow


def test_the_teammate_command_creates_and_lists_teammates(config, capsys):
    cli.teammate(["new", "nova", "--from", "byte"])
    cli.teammate(["list"])
    listing = capsys.readouterr().out
    assert "nova" in listing and "byte" in listing and "built in" in listing
    with pytest.raises(SystemExit, match="already exists"):
        cli.teammate(["new", "nova"])
    with pytest.raises(SystemExit, match="no teammate"):
        cli.teammate(["show", "ghost"])
