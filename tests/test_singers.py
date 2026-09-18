import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from humanoid_companion import voice
from humanoid_companion.perform import RATE, main, performance_frames, plan_performance
from humanoid_companion.singers import SINGERS, SingerRenderer, vowel_track, word_vowels
from humanoid_companion.teammates import ALESIA, BYTE, MAKS, TEMPO, all_teammates, teammate_from_file

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
REFERENCE = Path(__file__).with_name("data") / "robot_frames.npz"


def click_track(bpm: float, offset_s: float, seconds: float) -> np.ndarray:
    samples = np.zeros(int(seconds * RATE), np.float32)
    click = 0.8 * np.sin(np.linspace(0, 2 * np.pi * 60, int(0.03 * RATE))).astype(np.float32)
    for beat in np.arange(offset_s, seconds - 0.05, 60.0 / bpm):
        start = int(beat * RATE)
        samples[start : start + len(click)] += click
    return samples


def music(seconds: float) -> np.ndarray:
    tone = 0.2 * np.sin(np.arange(int(seconds * RATE)) / RATE * 2 * np.pi * 220).astype(np.float32)
    return click_track(100, 0.0, seconds) + tone


def test_the_singers_are_teammates_with_their_own_names_voices_and_characters():
    mates = all_teammates()
    assert mates["alesia"] is ALESIA and mates["maks"] is MAKS
    assert (ALESIA.display_name, MAKS.display_name) == ("Алеся / Alesia", "Максім / Maks")
    assert ALESIA.voice in {"af_bella", "af_sky"} and MAKS.voice in {"am_michael", "am_adam"}
    assert ALESIA.dances and MAKS.dances and {ALESIA.character, MAKS.character} == set(SINGERS)
    assert BYTE.character == TEMPO.character == "robot"
    for singer in (ALESIA, MAKS):
        text = singer.persona("Alex")
        assert text.startswith(f"You are {singer.name} ({singer.native_name}), an animated singer teammate:")
        assert "Belarus" in text and "never invent lyrics" in text and "not a real person" in text
        assert "call Alex by name" in text and "Answer only with the JSON object" in text


def test_a_teammate_file_based_on_a_singer_is_drawn_as_that_singer(tmp_path):
    path = tmp_path / "lesia.toml"
    path.write_text('name = "Lesia"\nbased_on = "alesia"\nvoice = "af_sky"\n')
    lesia = teammate_from_file(path)
    assert (lesia.character, lesia.dances, lesia.voice, lesia.native_name) == ("alesia", True, "af_sky", "")


@pytest.mark.parametrize(
    ("letter", "shape"),
    [("а", "a"), ("я", "a"), ("э", "e"), ("е", "e"), ("і", "e"), ("ы", "e"), ("о", "o"), ("у", "o"), ("ё", "o"), ("ю", "o")],
)
def test_every_belarusian_vowel_has_a_mouth_shape(letter, shape):
    assert word_vowels(letter) == [shape] and word_vowels(letter.upper()) == [shape]


def test_a_word_is_split_between_its_vowels_and_short_u_is_not_a_vowel():
    assert word_vowels("ўсё") == ["o"] and word_vowels("дахамі") == ["a", "a", "e"] and word_vowels("ў") == ["a"]
    lines = [{"start": 1.0, "end": 1.6, "words": [{"word": "далей", "start": 1.0, "end": 1.6}]}]
    track = vowel_track(lines, np.array([0.5, 1.1, 1.45, 2.0]))
    assert track == [None, "a", "e", None]


def test_the_mouth_is_closed_while_no_word_is_sung():
    words = [{"text": "мы едзем", "start": 0.5, "end": 1.2, "words": [{"word": "мы", "start": 0.5, "end": 0.7},
                                                                     {"word": "едзем", "start": 0.8, "end": 1.2}]}]  # fmt: skip
    performance = plan_performance(ALESIA, music(2.5), RATE, 30, words=words, bpm=100)
    assert performance.vowels[27] == "e" and performance.vowels[10] is None
    renderer = SingerRenderer(SINGERS["alesia"], 90, 120, fps=30)
    heights = []
    for i, expression in enumerate(performance.expressions):
        pose = renderer.plan(expression, float(performance.mouth[i]), float(performance.energy[i]),
                             float(performance.beat_phase[i]), performance.vowels[i])  # fmt: skip
        heights.append(pose.mouth[1])
    assert max(heights[:12]) < 2.0  # music plays, nothing sung: closed
    assert max(heights[18:36]) > 10.0  # singing
    assert max(heights[50:]) < 2.0  # closed again after the last word


@pytest.mark.parametrize("key", sorted(SINGERS))
def test_no_hand_or_microphone_leaves_the_frame_across_a_beat_sweep(key):
    renderer = SingerRenderer(SINGERS[key], 180, 240, supersample=1)
    for beats in np.arange(0.0, 8.0, 0.25):
        for sing in (1.0, 0.0):  # the microphone up at the mouth, and lowered
            alpha = renderer.still("happy", beats=float(beats), sing=sing, energy=1.0)[..., 3]
            edges = (alpha[:, :2], alpha[:, -2:], alpha[:2, :])
            assert max(int(edge.max()) for edge in edges) == 0, (key, beats, sing)


def test_the_dancing_robot_renders_exactly_as_before_the_singers():
    """Frames recorded from the code before the singers were wired into humanoid-perform. Byte's speaking
    motion was redesigned since (tests/test_motion.py), so only the dancing robot, Tempo, is pinned here."""
    words = [{"text": "one two", "start": 0.5, "end": 1.4, "words": [{"word": "one", "start": 0.5, "end": 0.8},
                                                                    {"word": "two", "start": 1.0, "end": 1.4}]}]  # fmt: skip
    tone = 0.2 * np.sin(np.arange(2 * RATE) / RATE * 2 * np.pi * 220).astype(np.float32)
    audio = click_track(100, 0.0, 2.0) + tone
    reference = np.load(REFERENCE)
    for teammate, options in ((TEMPO, {"words": words, "bpm": 100}),):
        frames = list(performance_frames(plan_performance(teammate, audio, RATE, 30, **options), teammate, (180, 240)))
        got = np.stack([frames[i] for i in (10, 35, 59)]).astype(np.int16)
        expected = reference[teammate.key].astype(np.int16)
        assert got.shape == expected.shape
        # 4x4 blocks: the same picture, allowing only sub-pixel edge differences between platforms
        blocks = np.abs(got - expected).reshape(3, 60, 4, 45, 4, 4).mean(axis=(2, 4))
        assert blocks.max() <= 6.0, teammate.key


@needs_ffmpeg
@pytest.mark.parametrize("key", sorted(SINGERS))
def test_a_singer_performance_is_written_with_alpha_and_audio(tmp_path, key):
    audio = tmp_path / "song.wav"
    audio.write_bytes(voice.to_wav(music(1.0), RATE))
    words = tmp_path / "times.json"
    words.write_text(json.dumps([{"text": "а о", "start": 0.2, "end": 0.8, "words": [{"word": "ао", "start": 0.2, "end": 0.8}]}]))
    out = tmp_path / f"{key}.webm"
    main(["--teammate", key, "--audio", str(audio), "--words", str(words), "--bpm", "100", "--size", "120x160", "--out", str(out)])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:stream_tags=alpha_mode", "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    streams = json.loads(probe.stdout)["streams"]
    assert {stream["codec_type"] for stream in streams} == {"video", "audio"}
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == (120, 160) and video.get("tags", {}).get("alpha_mode") == "1"
