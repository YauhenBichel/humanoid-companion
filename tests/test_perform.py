import json
import shutil
import subprocess

import numpy as np
import pytest

from humanoid_companion import voice
from humanoid_companion.perform import (
    RATE,
    estimate_beat,
    expression_at,
    main,
    output_arguments,
    plan_performance,
    sung_word_spans,
    word_gate,
)
from humanoid_companion.teammates import BYTE, TEMPO

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def click_track(bpm: float, offset_s: float, seconds: float = 8.0) -> np.ndarray:
    samples = np.zeros(int(seconds * RATE), np.float32)
    click = 0.8 * np.sin(np.linspace(0, 2 * np.pi * 60, int(0.03 * RATE))).astype(np.float32)
    for beat in np.arange(offset_s, seconds - 0.05, 60.0 / bpm):
        start = int(beat * RATE)
        samples[start : start + len(click)] += click
    return samples


def test_the_beat_is_found_in_a_click_track():
    bpm, offset = estimate_beat(click_track(120, 0.25), RATE)
    assert abs(bpm - 120) < 2 and abs(offset - 0.25) < 0.03


def test_the_mouth_opens_only_while_a_word_is_sung():
    lines = [{"text": "one line", "start": 1.0, "end": 2.0, "words": [{"word": "one", "start": 1.0, "end": 1.3}]}]
    times = np.array([0.5, 1.1, 1.6])
    assert sung_word_spans(lines) == [(1.0, 1.3)]
    assert word_gate(sung_word_spans(lines), times).tolist() == [0.0, 1.0, 0.0]
    assert sung_word_spans([{"start": 3.0, "end": 4.0}]) == [(3.0, 4.0)]  # a line without word timings


def test_cues_set_the_expression_from_their_time_on():
    cues = [
        {"at": 2.0, "expression": "surprised"},
        {"at": 1.0, "expression": "thinking"},
        {"at": 1.5, "expression": "x"},
    ]
    assert [expression_at(cues, t, "happy") for t in (0.5, 1.2, 1.7, 3.0)] == [
        "happy",
        "thinking",
        "thinking",
        "surprised",
    ]


def test_the_singer_dances_and_the_explainer_does_not():
    tone = 0.2 * np.sin(np.arange(3 * RATE) / RATE * 2 * np.pi * 220).astype(np.float32)
    music = click_track(100, 0.0, seconds=3.0) + tone
    words = [{"start": 1.0, "end": 2.0}]
    song = plan_performance(TEMPO, music, RATE, 30, words=words, bpm=100)
    assert song.beat_phase is not None and len(song.expressions) == 90 and set(song.expressions) == {"happy"}
    assert song.mouth[:25].max() == 0.0 and song.mouth[35:55].min() > 0.3  # music alone does not move the mouth
    speech = plan_performance(BYTE, music, RATE, 30)
    assert speech.beat_phase is None and speech.mouth.max() > 0.5


def test_an_mp4_needs_a_background_colour(tmp_path):
    with pytest.raises(SystemExit):
        output_arguments(tmp_path / "x.mp4", None)
    with pytest.raises(SystemExit):
        output_arguments(tmp_path / "x.gif", "#000000")
    assert "prores_ks" in output_arguments(tmp_path / "x.mov", None)


@needs_ffmpeg
@pytest.mark.parametrize(("name", "background", "alpha"), [("tempo.webm", None, True), ("tempo.mp4", "#101018", False)])
def test_a_performance_is_written_with_its_audio(tmp_path, name, background, alpha):
    audio = tmp_path / "song.wav"
    audio.write_bytes(voice.to_wav(click_track(120, 0.1, seconds=1.0), RATE))
    out = tmp_path / name
    main(["--teammate", "tempo", "--audio", str(audio), "--out", str(out), "--size", "180x240"]
         + (["--background", background] if background else []))  # fmt: skip
    entries = "stream=codec_type,width:stream_tags=alpha_mode"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    streams = json.loads(probe.stdout)["streams"]
    assert {stream["codec_type"] for stream in streams} == {"video", "audio"}
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert video["width"] == 180 and (video.get("tags", {}).get("alpha_mode") == "1") is alpha
