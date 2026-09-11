"""The talk runtime with every outside service faked: speech, model, planner."""

import argparse
import json
import shutil
import subprocess

import numpy as np
import pytest

from humanoid_companion import brain, voice
from humanoid_companion.brain import Command
from humanoid_companion.conversation import Conversation

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
TONE = voice.to_wav(0.3 * np.sin(np.linspace(0, 2 * np.pi * 200, 2400)).astype(np.float32), 24000)  # 0.1 s


class Stand:
    def __init__(self):
        from humanoid_companion.export import spec_from_env

        self.spec = spec_from_env("Op3Joystick")

    def act(self, obs):
        return np.zeros(self.spec.action_size, np.float32)


def make_robot(tmp_path, monkeypatch, record=True, body=True, reply=None):
    from humanoid_companion import talk
    from humanoid_companion.run_sim import training_model

    spoken = []
    monkeypatch.setattr(voice, "speak", lambda text, lang="en": spoken.append((text, lang) if lang != "en" else text) or TONE)
    monkeypatch.setattr(brain, "plan", lambda instruction, **kw: brain.Plan(
        [Command(0.3, 0, 0, 0.5), Command(0, 0, 0.5, 0.5)], "model", routing={"backend": "local"}))
    args = argparse.Namespace(port=0, record=tmp_path if record else None, policy=None, speed_sweep=None,
                              speed_cap=0.5, mic=False, speaker=False)
    robot = talk.Robot(args)
    robot.conv = Conversation(lambda m, s: json.dumps(reply or {
        "say": "Walking now!", "expression": "happy", "action": {"kind": "walk", "instruction": "walk then turn"}}))
    if body:
        robot.body = talk.Body(Stand(), training_model(), robot.face, keep_frames=record)
    return robot, spoken


@needs_ffmpeg
def test_a_walking_turn_speaks_and_walks_together_and_is_recorded(tmp_path, monkeypatch):
    robot, spoken = make_robot(tmp_path, monkeypatch)
    try:
        robot.turn("please walk", None)
    finally:
        robot.face.stop()
    c = json.loads((tmp_path / "conversation.json").read_text())
    t = c["turns"][0]
    assert c["name"] == "" and spoken == ["Walking now!"]
    assert t["say"] == "Walking now!" and (tmp_path / t["audio"]).read_bytes() == TONE
    w = t["walk"]
    assert w["instruction"] == "walk then turn" and w["plan_routing"] == {"backend": "local"} and w["safety_stops"] == []
    assert w["seconds"] == 1.5   # the 1 s walk plan + 0.5 s settle; the 0.1 s speech overlaps it
    assert t["gesture"] == "none" and t["walk_delay_s"] == 0.0
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width:format=duration",
                            "-of", "json", str(tmp_path / t["video"])], capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
    assert [s["width"] for s in info["streams"] if s["codec_type"] == "video"] == [1280]   # face | body
    assert 2.5 <= float(info["format"]["duration"]) <= 3.0   # 1.2 s thinking beat + 1.5 s turn


def test_the_farewell_names_the_person_only_when_the_belarusian_name_is_known():
    from humanoid_companion.talk import farewell_lines

    assert farewell_lines("", "") == {"belarusian": "Дзякуй! Да сустрэчы!",
                                      "caption": "Дзякуй! Да сустрэчы! (Thank you! See you soon!)",
                                      "phonetic": "Dzyakuy! Da sustrechy!"}
    f = farewell_lines("Yauhen", "Яўген")
    assert f["caption"] == "Дзякуй, Яўген! Да сустрэчы! (Thank you, Yauhen! See you soon!)"
    assert farewell_lines("Yauhen", "")["belarusian"] == "Дзякуй! Да сустрэчы!"


@needs_ffmpeg
def test_the_farewell_is_spoken_by_the_belarusian_voice(tmp_path, monkeypatch):
    from humanoid_companion import talk

    monkeypatch.setattr(talk, "FAREWELL", talk.farewell_lines("Yauhen", "Яўген"))
    robot, spoken = make_robot(tmp_path, monkeypatch)
    try:
        robot.farewell()
    finally:
        robot.face.stop()
    t = json.loads((tmp_path / "conversation.json").read_text())["turns"][0]
    assert t["say"].startswith("Дзякуй, Яўген! Да сустрэчы!") and "Thank you, Yauhen" in t["say"]
    assert spoken == [("Дзякуй, Яўген! Да сустрэчы!", "be")] and t["routing"] == {"voice": "omnivoice-be"}
    assert t["gesture"] == "wave"   # waves goodbye
    assert t["source"] == "scripted" and "walk" not in t


@needs_ffmpeg
def test_without_the_belarusian_service_the_farewell_falls_back_and_says_so(tmp_path, monkeypatch):
    from humanoid_companion import talk

    monkeypatch.setattr(talk, "FAREWELL", talk.farewell_lines("Yauhen", "Яўген"))
    robot, spoken = make_robot(tmp_path, monkeypatch)

    def speak(text, lang="en"):
        if lang == "be":
            raise ConnectionRefusedError("refused")
        spoken.append(text)
        return TONE

    monkeypatch.setattr(voice, "speak", speak)
    try:
        robot.farewell()
    finally:
        robot.face.stop()
    t = json.loads((tmp_path / "conversation.json").read_text())["turns"][0]
    assert spoken == ["Dzyakuy, Yauhen! Da sustrechy!"] and t["routing"] == {"voice": "kokoro-phonetic"}
    assert "unavailable" in t["error"]


def test_without_a_speech_server_the_robot_still_answers_silently(tmp_path, monkeypatch):
    robot, _ = make_robot(tmp_path, monkeypatch, record=False, body=False)

    def down(text, lang="en"):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(voice, "speak", down)
    try:
        robot.turn("please walk", None)
        robot.turn("again", None)
    finally:
        robot.face.stop()
    assert [t["say"] for t in robot.turns] == ["Walking now!", "Walking now!"] and robot.silent
    assert robot.turns[0]["speech_s"] == 1.0   # silence as long as the caption takes to read


def test_without_a_body_a_walk_request_is_only_spoken(tmp_path, monkeypatch):
    robot, spoken = make_robot(tmp_path, monkeypatch, record=False, body=False)
    try:
        robot.turn("please walk", None)
    finally:
        robot.face.stop()
    assert spoken == ["Walking now!"] and "walk" not in robot.turns[0]


@needs_ffmpeg
def test_an_arm_gesture_comes_before_the_walk_never_during_it(tmp_path, monkeypatch):
    robot, _ = make_robot(tmp_path, monkeypatch, reply={"say": "Here I go!", "expression": "happy", "gesture": "wave",
                                                         "action": {"kind": "walk", "instruction": "walk then turn"}})
    try:
        robot.turn("walk please", None)
    finally:
        robot.face.stop()
    t = json.loads((tmp_path / "conversation.json").read_text())["turns"][0]
    assert t["gesture"] == "wave" and t["walk_delay_s"] == 2.6
    assert t["walk"]["seconds"] == 4.1 and t["walk"]["safety_stops"] == []   # 2.6 s wave + 1 s walk + 0.5 s


@needs_ffmpeg
def test_compose_joins_the_turn_videos(tmp_path, monkeypatch):
    from humanoid_companion.talk import compose

    robot, _ = make_robot(tmp_path, monkeypatch, reply={"say": "Hi!", "expression": "happy",
                                                         "action": {"kind": "none", "instruction": ""}})
    try:
        robot.turn("hello", None)
        robot.turn("again", None)
    finally:
        robot.face.stop()
    out = compose(tmp_path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(out)],
                           capture_output=True, text=True, check=True)
    assert 3.0 <= float(json.loads(probe.stdout)["format"]["duration"]) <= 4.0   # 2 x (1.2 + 0.5)
    assert not list(tmp_path.glob("_*"))
