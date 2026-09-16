"""Teammates: the companion as two distinct characters, for the laptop and for video clips.

    byte    explains computer science: algorithms, data structures, complexity, one idea at a time
    tempo   the singer: performs songs and dances to them

A teammate is a look (face colours, the head's trim and an accessory), a voice, and a role added to
the companion's persona (humanoid_companion.conversation.persona). The same teammate is used live
(`humanoid-talk --teammate byte`) and in videos (humanoid_companion.character, `humanoid-perform`).
Names are neutral on purpose; the persona tells the model not to guess anyone's pronouns.

Your own teammates are TOML files in the settings folder's teammates/ directory
(humanoid_companion.settings; `humanoid-teammate new nova --from byte` writes one to edit). The file
name is the key; a file named byte.toml changes the built-in Byte:

    name = "Nova"
    tagline = "tells stories about space"
    based_on = "byte"                  # start from a built-in teammate; the fields below override it
    accessory = "antenna"              # antenna, headphones or none
    voice = "af_sky"                   # a voice your speech server knows
    dances = false                     # sways and bounces on the beat in clips
    resting_expression = "happy"
    role = "Your role: you tell short, true stories about space."

    [colours]
    glow = "#9fd0ff"
    background = "#05070f"
    caption = "#dfeeff"
    trim = "#2a3552"
"""

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from humanoid_companion.conversation import NAME, persona
from humanoid_companion.face.look import Colour, Look
from humanoid_companion.face.server import EXPRESSIONS

ACCESSORIES = ("antenna", "headphones", "none")
KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class TeammateError(ValueError):
    """A teammate file that cannot be used as it is."""


@dataclass(frozen=True)
class Teammate:
    key: str
    name: str
    tagline: str
    look: Look
    trim: Colour  # the head's shell and shoulders
    accessory: str  # "antenna", "headphones" or "none", drawn by humanoid_companion.character
    voice: str  # speaking voice (a Kokoro voice name)
    role: str  # added to the persona
    resting_expression: str = "neutral"
    dances: bool = False  # sways and bounces on the beat in clips (humanoid_companion.perform)
    source: str = "built in"  # or the file it was loaded from

    def persona(self, name: str = NAME, more_role: str = "") -> str:
        """The companion's persona as this teammate; `more_role` adds to the role (a singer's repertoire)."""
        return persona(name, identity=f"{self.name}, a humanoid teammate", role=f"{self.role} {more_role}".strip())


BYTE = Teammate(
    key="byte",
    name="Byte",
    tagline="explains computer science",
    look=Look(glow=(120, 255, 190), background=(4, 12, 10), caption=(214, 255, 234), shadow=None),
    trim=(38, 70, 64),
    accessory="antenna",
    voice="af_heart",  # the calm, clear voice the graph-algorithm clips already use
    role=(
        "Your role: you explain computer science: algorithms, data structures, complexity, how computers and "
        "programs work. Explain one idea at a time with a small concrete example, in plain words; say what a "
        "term means the first time you use it; prefer 'for example' over jargon. When asked about big-O, say "
        "what grows and why. If you are not sure, say so instead of guessing. Look 'thinking' while you work "
        "something out and 'happy' when an idea clicks."
    ),
    resting_expression="neutral",
)

TEMPO = Teammate(
    key="tempo",
    name="Tempo",
    tagline="sings songs",
    look=Look(glow=(255, 150, 220), background=(14, 5, 16), caption=(255, 226, 244), shadow=None),
    trim=(78, 40, 84),
    accessory="headphones",
    voice="af_bella",
    role=(
        "Your role: you are a singer. You love music, rhythm and songs, and you like talking about melodies, "
        "styles and what a song is about. Only sing songs you have been given; never invent lyrics on the spot "
        "and never sing someone else's copyrighted song. Choose 'dance' when music comes up and 'celebrate' "
        "when someone likes a song."
    ),
    resting_expression="happy",
    dances=True,
)

TEAMMATES = {teammate.key: teammate for teammate in (BYTE, TEMPO)}  # the built-in teammates


def parse_colour(text: str, field: str) -> Colour:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(text).strip())
    if not match:
        raise TeammateError(f"{field}: expected a colour like #78ffbe, got {text!r}")
    value = match.group(1)
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def teammate_from_file(path: Path, built_in: dict[str, Teammate] = TEAMMATES) -> Teammate:
    """A teammate from its TOML file, checked field by field; errors name the file and the field."""
    key = path.stem
    if not KEY_PATTERN.match(key):
        raise TeammateError(f"{path}: the file name must be lowercase letters, digits and hyphens")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise TeammateError(f"{path}: {error}") from None
    base_key = data.get("based_on", key if key in built_in else "byte")
    if base_key not in built_in:
        raise TeammateError(f"{path}: based_on must be one of {', '.join(built_in)}")
    base = built_in[base_key]
    try:
        colours = data.get("colours", {})
        look = Look(
            glow=parse_colour(colours["glow"], "colours.glow") if "glow" in colours else base.look.glow,
            background=parse_colour(colours["background"], "colours.background")
            if "background" in colours
            else base.look.background,
            caption=parse_colour(colours["caption"], "colours.caption") if "caption" in colours else base.look.caption,
            shadow=None,
        )
        teammate = replace(
            base,
            key=key,
            name=str(data.get("name", base.name if key == base_key else key.capitalize())),
            tagline=str(data.get("tagline", base.tagline)),
            look=look,
            trim=parse_colour(colours["trim"], "colours.trim") if "trim" in colours else base.trim,
            accessory=str(data.get("accessory", base.accessory)),
            voice=str(data.get("voice", base.voice)),
            role=str(data.get("role", base.role)),
            resting_expression=str(data.get("resting_expression", base.resting_expression)),
            dances=bool(data.get("dances", base.dances)),
            source=str(path),
        )
    except TeammateError as error:
        raise TeammateError(f"{path}: {error}") from None
    if teammate.accessory not in ACCESSORIES:
        raise TeammateError(f"{path}: accessory must be one of {', '.join(ACCESSORIES)}")
    if teammate.resting_expression not in EXPRESSIONS:
        raise TeammateError(f"{path}: resting_expression must be one of {', '.join(EXPRESSIONS)}")
    if not teammate.role.strip():
        raise TeammateError(f"{path}: role must say what the teammate is for")
    return teammate


def all_teammates(directory: Path | None = None) -> dict[str, Teammate]:
    """The built-in teammates and your own from `directory` (default: the settings folder's teammates/)."""
    if directory is None:
        from humanoid_companion.settings import config_dir

        directory = config_dir() / "teammates"
    teammates = dict(TEAMMATES)
    if directory.is_dir():
        for path in sorted(directory.glob("*.toml")):
            teammate = teammate_from_file(path)
            teammates[teammate.key] = teammate
    return teammates


def teammate_template(key: str, based_on: str = "byte") -> str:
    """A teammate file to edit, starting from a built-in teammate's values."""
    if not KEY_PATTERN.match(key):
        raise TeammateError("a teammate's key must be lowercase letters, digits and hyphens")
    base = TEAMMATES[based_on]
    colours = {
        "glow": base.look.glow,
        "background": base.look.background,
        "caption": base.look.caption,
        "trim": base.trim,
    }
    hexes = {name: "#{:02x}{:02x}{:02x}".format(*colour) for name, colour in colours.items()}
    role = base.role.replace('"""', "'")
    return f'''# A humanoid-companion teammate. The file name ({key}.toml) is its key: humanoid-talk --teammate {key}
name = "{key.capitalize()}"
tagline = "{base.tagline}"
# The built-in teammate this one starts from; every field here overrides it.
based_on = "{based_on}"
# antenna, headphones or none
accessory = "{base.accessory}"
# A voice your speech server knows (Kokoro: af_heart, af_bella, af_sky, am_michael, bf_emma, ...)
voice = "{base.voice}"
# Sways and bounces on the beat in clips
dances = {str(base.dances).lower()}
# One of {", ".join(EXPRESSIONS)}
resting_expression = "{base.resting_expression}"
# What it is for and how it talks; added to the companion's persona
role = """{role}"""

[colours]
glow = "{hexes["glow"]}"
background = "{hexes["background"]}"
caption = "{hexes["caption"]}"
trim = "{hexes["trim"]}"
'''
