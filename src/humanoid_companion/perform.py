"""A teammate performs an audio file: speech or a song in, a character video out.

    humanoid-perform --teammate byte --audio narration.wav --out byte.mov
    humanoid-perform --teammate tempo --audio song/audio.wav --words song/times.json --out tempo.webm
    humanoid-perform --teammate tempo --audio song/audio.wav --words song/times.json \
        --background "#101018" --out tempo.mp4
    humanoid-perform --teammate alesia --audio song/audio.wav --words song/times.json --size 360x480 --out alesia.webm

The output keeps the audio. Its format follows the file name:

    .mov   ProRes 4444 with alpha: for editors and for laying the character over another video
    .webm  VP9 with alpha: smaller, for the web and ffmpeg overlays
    .mp4   H.264 on a solid --background colour: to watch or post as it is

How the character moves:
- the mouth follows the voice's loudness. For a song the loudness is mostly the music, so pass the
  sung words (`--words`, a JSON list of lines with `start`, `end` and optional `words` with their own
  `start` and `end`): the mouth then opens only while a word is sung. The singers Alesia and Maks also
  shape the mouth by the vowel being sung (humanoid_companion.singers.vowel_track);
- a singer (Tempo, Alesia, Maks) dances on the beat: `--bpm` and `--beat-offset`, or estimated from the audio;
- `--cues` sets expressions over time: a JSON list of {"at": seconds, "expression": name}.
"""

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_companion.character import CharacterRenderer
from humanoid_companion.face.render import mouth_track
from humanoid_companion.face.server import EXPRESSIONS
from humanoid_companion.singers import SINGERS, SingerRenderer, vowel_track
from humanoid_companion.teammates import Teammate, all_teammates

RATE = 24000  # audio is analysed at this rate; the output keeps the original audio
WORD_PAD_S = 0.04  # the mouth starts opening just before a word and closes just after it

ENCODERS = {
    ".mov": ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-c:a", "pcm_s16le"],
    ".webm": ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30", "-c:a", "libopus"],
    ".mp4": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart"],
}


def decode_audio(path: Path, rate: int = RATE) -> np.ndarray:
    """Any audio file as mono float32 samples at `rate` (decoded by ffmpeg)."""
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, dtype=np.float32)


def frame_energy(samples: np.ndarray, rate: int, fps: float, frames: int) -> np.ndarray:
    """Loudness per video frame scaled to 0..1 by the loud end of this audio (95th percentile), smoothed."""
    hop = rate / fps
    rms = np.array([np.sqrt(np.mean(samples[int(i * hop) : int((i + 1) * hop)] ** 2) or 0.0) for i in range(frames)])
    reference = np.percentile(rms, 95) if frames else 0.0
    scaled = np.clip(rms / reference, 0.0, 1.0) if reference > 0 else np.zeros(frames)
    smoothed, level = np.zeros(frames), 0.0
    for i, value in enumerate(scaled):
        level += (value - level) * 0.35
        smoothed[i] = level
    return smoothed


def estimate_beat(
    samples: np.ndarray, rate: int, low_bpm: float = 70.0, high_bpm: float = 180.0
) -> tuple[float, float]:
    """(tempo in beats per minute, time of the first beat in seconds) from the rises in loudness.

    The onset strength (positive changes of loudness at 100 Hz) is autocorrelated over the lags of
    `low_bpm`..`high_bpm`; the beat's offset is where a comb at that period collects the most onsets."""
    hop = rate // 100
    envelope = np.array([np.sqrt(np.mean(samples[i : i + hop] ** 2)) for i in range(0, len(samples) - hop, hop)])
    onsets = np.maximum(np.diff(envelope, prepend=envelope[:1]), 0.0)
    onsets -= onsets.mean()
    lags = np.arange(int(100 * 60 / high_bpm), int(100 * 60 / low_bpm) + 1)
    scores = [float(np.dot(onsets[:-lag], onsets[lag:])) for lag in lags if lag < len(onsets)]
    if not scores:
        return 120.0, 0.0
    period = int(lags[int(np.argmax(scores))])
    offset = int(np.argmax([onsets[start::period].sum() for start in range(period)]))
    return 6000.0 / period, offset / 100.0


def sung_word_spans(lines: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of every sung word, or of whole lines when a line has no word timings."""
    spans = []
    for line in lines:
        words = line.get("words") or [line]
        spans += [(float(word["start"]), float(word["end"])) for word in words]
    return spans


def word_gate(spans: list[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    """1.0 at the times inside a sung word (padded by WORD_PAD_S), else 0.0."""
    gate = np.zeros(len(times))
    for start, end in spans:
        gate[(times >= start - WORD_PAD_S) & (times <= end + WORD_PAD_S)] = 1.0
    return gate


def expression_at(cues: list[dict], seconds: float, default: str) -> str:
    current = default
    for cue in sorted(cues, key=lambda cue: cue["at"]):
        if cue["at"] <= seconds and cue["expression"] in EXPRESSIONS:
            current = cue["expression"]
    return current


@dataclass(frozen=True)
class Performance:
    """Everything the character does, one value per video frame."""

    fps: float
    expressions: list[str]
    mouth: np.ndarray
    energy: np.ndarray
    beat_phase: np.ndarray | None  # None: the character does not dance
    vowels: list[str | None] | None = None  # the vowel sung per frame ("a", "e", "o"), from the words


def plan_performance(
    teammate: Teammate,
    samples: np.ndarray,
    rate: int,
    fps: float,
    words: list[dict] | None = None,
    cues: list[dict] | None = None,
    bpm: float | None = None,
    beat_offset: float = 0.0,
    dance: bool | None = None,
) -> Performance:
    frames = max(1, math.ceil(len(samples) / rate * fps))
    times = np.arange(frames) / fps
    energy = frame_energy(samples, rate, fps, frames)
    if words:
        # Music under the voice keeps the loudness up, so the words decide when the mouth moves.
        mouth = word_gate(sung_word_spans(words), times) * (0.35 + 0.65 * energy)
    else:
        mouth = mouth_track(samples, rate, fps, frames)
    dance = teammate.dances if dance is None else dance
    beat_phase = None
    if dance:
        if bpm is None:
            bpm, beat_offset = estimate_beat(samples, rate)
        beat_phase = ((times - beat_offset) * bpm / 60.0) % 1.0
    expressions = [expression_at(cues or [], t, teammate.resting_expression) for t in times]
    vowels = vowel_track(words, times) if words else None
    return Performance(fps, expressions, mouth, energy, beat_phase, vowels)


def output_arguments(out: Path, background: str | None) -> list[str]:
    suffix = out.suffix.lower()
    if suffix not in ENCODERS:
        raise SystemExit(f"--out must end in {', '.join(ENCODERS)}")
    if suffix == ".mp4" and not background:
        raise SystemExit("an .mp4 has no transparency: give --background, or write .mov or .webm")
    return ENCODERS[suffix]


def character_renderer(teammate: Teammate, width: int, height: int, fps: float):
    """The renderer that draws this teammate: the robot bust, or one of the singers."""
    if teammate.character in SINGERS:
        return SingerRenderer(SINGERS[teammate.character], width, height, fps=fps)
    return CharacterRenderer(teammate, width, height, fps=fps)


def performance_frames(performance: Performance, teammate: Teammate, size: tuple[int, int]):
    """Every frame of the performance as an RGBA array, in order."""
    renderer = character_renderer(teammate, *size, fps=performance.fps)
    singer = isinstance(renderer, SingerRenderer)
    for i, expression in enumerate(performance.expressions):
        phase = None if performance.beat_phase is None else float(performance.beat_phase[i])
        mouth, energy = float(performance.mouth[i]), float(performance.energy[i])
        if singer:
            vowel = performance.vowels[i] if performance.vowels else None
            yield renderer.frame(expression, mouth, energy, phase, vowel)
        else:
            yield renderer.frame(expression, mouth, energy, phase)


def render(
    performance: Performance,
    teammate: Teammate,
    audio: Path,
    out: Path,
    size: tuple[int, int] = (540, 720),
    background: str | None = None,
) -> Path:
    """Draw every frame and encode them with the audio into `out`."""
    width, height = size
    video_input = ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(performance.fps)]
    if background:
        colour = background.lstrip("#")
        backdrop = f"color=c=0x{colour}:s={width}x{height}:r={performance.fps}"
        video_graph = ["-filter_complex", f"{backdrop}[bg];[bg][0:v]overlay=shortest=1[v]", "-map", "[v]"]
    else:
        video_graph = ["-map", "0:v"]
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error", *video_input, "-i", "-", "-i", str(audio), *video_graph, "-map", "1:a",
        *output_arguments(out, background), "-shortest", str(out),
    ]  # fmt: skip
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in performance_frames(performance, teammate, size):
            encoder.stdin.write(frame.tobytes())
    finally:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {out}")
    return out


def perform_file(
    teammate: Teammate,
    audio: Path,
    out: Path,
    words: Path | list[dict] | None = None,
    cues: Path | list[dict] | None = None,
    bpm: float | None = None,
    beat_offset: float = 0.0,
    dance: bool | None = None,
    size: tuple[int, int] = (540, 720),
    fps: float = 30.0,
    background: str | None = None,
) -> Path:
    """The whole of humanoid-perform as one call, for other programs:

        from humanoid_companion.perform import perform_file
        from humanoid_companion.teammates import all_teammates
        tempo = all_teammates()["tempo"]
        perform_file(tempo, Path("song/audio.wav"), Path("tempo.webm"), words=Path("song/times.json"))

    `words` and `cues` are JSON files or the already-loaded lists."""
    output_arguments(out, background)  # refuse a wrong combination before any work

    def loaded(value):
        return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value

    performance = plan_performance(
        teammate, decode_audio(audio), RATE, fps, words=loaded(words), cues=loaded(cues),
        bpm=bpm, beat_offset=beat_offset, dance=dance,
    )  # fmt: skip
    return render(performance, teammate, audio, out, size, background)


def parse_size(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    return int(width), int(height)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    teammates = all_teammates()
    parser.add_argument(
        "--teammate", required=True, choices=sorted(teammates), help="built in: byte, tempo, alesia, maks; or your own"
    )
    parser.add_argument("--audio", required=True, type=Path, help="speech or a song, any format ffmpeg reads")
    parser.add_argument("--out", required=True, type=Path, help=".mov or .webm (transparent), .mp4 (with --background)")
    parser.add_argument("--words", type=Path, help="JSON lines with start/end and optional words, for songs")
    parser.add_argument("--cues", type=Path, help='JSON list of {"at": seconds, "expression": name}')
    parser.add_argument("--bpm", type=float, help="the song's tempo; estimated from the audio when missing")
    parser.add_argument("--beat-offset", type=float, default=0.0, help="seconds to the first beat, with --bpm")
    parser.add_argument("--dance", action=argparse.BooleanOptionalAction, help="default: the singers (Tempo, Alesia, Maks) dance, Byte does not")
    parser.add_argument("--size", type=parse_size, default=(540, 720), help="WIDTHxHEIGHT, default 540x720")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--background", help="a solid colour such as #101018 (required for .mp4)")
    args = parser.parse_args(argv)

    perform_file(
        teammates[args.teammate], args.audio, args.out, words=args.words, cues=args.cues, bpm=args.bpm,
        beat_offset=args.beat_offset, dance=args.dance, size=args.size, fps=args.fps, background=args.background,
    )  # fmt: skip
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
