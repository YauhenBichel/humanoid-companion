"""A singer's song library: folders of finished songs the teammate may perform, never made up on the spot.

    songs/
      sun-shines/
        audio.wav      the song (any format ffmpeg reads, named audio.*)
        times.json     the sung lines: [{"text": "...", "start": s, "end": s, "words": [...]}, ...]
        song.json      optional: {"title": "..."}

The singer's persona lists the songs by id and first line; when asked to sing, the model answers
with action.kind "sing" and the song's id, and the runtime plays that song. An id that is not in the
library plays nothing.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Song:
    key: str
    title: str
    audio: Path
    lines: list[dict]

    @property
    def first_line(self) -> str:
        return caption_text(self.lines[0]["text"]) if self.lines else ""


def caption_text(text: str) -> str:
    """A line's sung text: bilingual lines ("sung | translation") keep their sung half."""
    return text.split("|")[0].strip()


def load_library(directory: Path) -> dict[str, Song]:
    songs = {}
    for folder in sorted(path for path in Path(directory).iterdir() if path.is_dir()):
        audio = next(iter(sorted(folder.glob("audio.*"))), None)
        times = folder / "times.json"
        if audio is None or not times.exists():
            continue
        metadata = folder / "song.json"
        title = json.loads(metadata.read_text()).get("title", "") if metadata.exists() else ""
        songs[folder.name] = Song(folder.name, title or folder.name, audio, json.loads(times.read_text()))
    return songs


def find_song(library: dict[str, Song], instruction: str) -> Song | None:
    """The song the model asked for: its id, or failing that, the id or title containing the instruction's words."""
    wanted = instruction.strip().lower()
    if wanted in library:
        return library[wanted]
    words = set(re.findall(r"\w+", wanted))
    if not words:
        return None
    matches = [
        song for song in library.values() if words <= set(re.findall(r"\w+", f"{song.key} {song.title}".lower()))
    ]
    return matches[0] if len(matches) == 1 else None


def caption_at(lines: list[dict], seconds: float) -> str:
    """The line being sung at `seconds`, or "" between lines."""
    for line in lines:
        if line["start"] <= seconds <= line["end"]:
            return caption_text(line["text"])
    return ""


def repertoire(library: dict[str, Song]) -> str:
    """The persona paragraph that tells the singer what it can sing."""
    if not library:
        return "You have no songs to sing right now; say so kindly if asked."
    listing = "; ".join(f"'{song.key}' ({song.first_line})" for song in library.values())
    return (
        f"Songs you can sing: {listing}. When the person asks you to sing, say a short line, set action.kind to "
        "'sing' and action.instruction to the song's id exactly as written; for any other request use 'none'."
    )
