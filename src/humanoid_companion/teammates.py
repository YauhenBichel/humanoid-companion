"""Teammates: the companion as two distinct characters, for the laptop and for video clips.

    byte    explains computer science: algorithms, data structures, complexity, one idea at a time
    tempo   the singer: performs songs and dances to them

A teammate is a look (face colours, the head's trim and an accessory), a voice, and a role added to
the companion's persona (humanoid_companion.conversation.persona). The same teammate is used live
(`humanoid-talk --teammate byte`) and in videos (humanoid_companion.character, `humanoid-perform`).
Names are neutral on purpose; the persona tells the model not to guess anyone's pronouns.
"""

from dataclasses import dataclass

from humanoid_companion.conversation import NAME, persona
from humanoid_companion.face.look import Colour, Look


@dataclass(frozen=True)
class Teammate:
    key: str
    name: str
    tagline: str
    look: Look
    trim: Colour  # the head's shell and shoulders
    accessory: str  # "antenna" (byte) or "headphones" (tempo), drawn by humanoid_companion.character
    voice: str  # speaking voice (a Kokoro voice name)
    role: str  # added to the persona
    resting_expression: str = "neutral"

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
)

TEAMMATES = {teammate.key: teammate for teammate in (BYTE, TEMPO)}
