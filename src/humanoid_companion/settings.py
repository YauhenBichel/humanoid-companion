"""Your settings in one file, so the commands work from any folder without exporting variables.

    humanoid-teammate init        # writes ~/.config/humanoid-companion/settings.toml to edit

    [user]
    name = "Alex"                 # HUMANOID_USER_NAME
    name_be = ""                  # HUMANOID_USER_NAME_BE

    [llm]
    base_url = "http://127.0.0.1:11434/v1"   # HUMANOID_LLM_BASE_URL
    model = "llama3.1:8b"                    # HUMANOID_LLM_MODEL

    [voice]
    tts_base_url = "http://127.0.0.1:8880/v1"    # HUMANOID_TTS_BASE_URL
    stt_base_url = "http://127.0.0.1:8000/v1"    # HUMANOID_STT_BASE_URL
    be_tts_url = "http://127.0.0.1:11810/v1"     # HUMANOID_BE_TTS_URL
    voice = "af_heart"                           # HUMANOID_TTS_VOICE

    [songs]
    tempo = "~/songs"             # a teammate's song library, used when --songs is not given

Each value only fills in an environment variable that is not already set, so a variable you export
always wins. API keys do not belong in this file: keep HUMANOID_LLM_API_KEY in your environment or a
secret manager. The folder is $HUMANOID_CONFIG_DIR, else $XDG_CONFIG_HOME/humanoid-companion, else
~/.config/humanoid-companion; custom teammates live in its teammates/ folder.
"""

import os
import tomllib
from pathlib import Path

ENVIRONMENT = {
    ("user", "name"): "HUMANOID_USER_NAME",
    ("user", "name_be"): "HUMANOID_USER_NAME_BE",
    ("llm", "base_url"): "HUMANOID_LLM_BASE_URL",
    ("llm", "model"): "HUMANOID_LLM_MODEL",
    ("voice", "tts_base_url"): "HUMANOID_TTS_BASE_URL",
    ("voice", "stt_base_url"): "HUMANOID_STT_BASE_URL",
    ("voice", "be_tts_url"): "HUMANOID_BE_TTS_URL",
    ("voice", "voice"): "HUMANOID_TTS_VOICE",
}

TEMPLATE = """# humanoid-companion settings: every value fills in the environment variable named next to it,
# unless that variable is already set. Keep API keys out of this file (use HUMANOID_LLM_API_KEY).

[user]
name = ""                                      # HUMANOID_USER_NAME: the robot calls you by it
name_be = ""                                   # HUMANOID_USER_NAME_BE: your name in Belarusian, for the goodbye

[llm]
base_url = "http://127.0.0.1:11434/v1"         # HUMANOID_LLM_BASE_URL: any OpenAI-compatible chat server
model = ""                                     # HUMANOID_LLM_MODEL

[voice]
tts_base_url = "http://127.0.0.1:8880/v1"      # HUMANOID_TTS_BASE_URL: /audio/speech
stt_base_url = "http://127.0.0.1:8000/v1"      # HUMANOID_STT_BASE_URL: /audio/transcriptions
be_tts_url = "http://127.0.0.1:11810/v1"       # HUMANOID_BE_TTS_URL: the Belarusian goodbye
voice = "af_heart"                             # HUMANOID_TTS_VOICE: the plain robot's voice

[songs]
# tempo = "~/songs"                            # a teammate's song library, used when --songs is not given
"""


class SettingsError(ValueError):
    """The settings file cannot be used as it is."""


def config_dir() -> Path:
    if explicit := os.environ.get("HUMANOID_CONFIG_DIR"):
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base).expanduser() / "humanoid-companion"


def settings_file() -> Path:
    return config_dir() / "settings.toml"


def load() -> dict:
    """The settings file's contents, or {} when there is none."""
    path = settings_file()
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise SettingsError(f"{path}: {error}") from None


def apply(settings: dict | None = None) -> list[str]:
    """Fill in unset environment variables from the settings; returns the variables it set."""
    settings = load() if settings is None else settings
    applied = []
    for (section, key), variable in ENVIRONMENT.items():
        value = settings.get(section, {}).get(key)
        if isinstance(value, str) and value.strip() and variable not in os.environ:
            os.environ[variable] = value.strip()
            applied.append(variable)
    return applied


def song_library(teammate_key: str, settings: dict | None = None) -> Path | None:
    """The song library configured for a teammate under [songs], if any."""
    settings = load() if settings is None else settings
    folder = settings.get("songs", {}).get(teammate_key)
    return Path(folder).expanduser() if isinstance(folder, str) and folder.strip() else None


def write_template() -> Path:
    """Create the settings file to edit; an existing file is left alone."""
    path = settings_file()
    if path.exists():
        raise SettingsError(f"{path} already exists; edit it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE)
    return path
