import json

import numpy as np

from humanoid_companion import voice
from humanoid_companion.conversation import ACTIONS, Conversation, parse_reply
from humanoid_companion.songs import caption_at, find_song, load_library, repertoire

LINES = [
    {"text": "Сонца свеціць нам у вочы, | The sun shines in our eyes,", "start": 0.2, "end": 1.0},
    {"text": "Едзем мы да мора.", "start": 1.2, "end": 2.0},
]


def make_library(root, tone_seconds=2.2):
    for key, title in (("sun-shines", "Sun shines"), ("night-city", "")):
        folder = root / key
        folder.mkdir(parents=True)
        tone = 0.2 * np.sin(np.arange(int(24000 * tone_seconds)) / 24000 * 2 * np.pi * 330).astype(np.float32)
        (folder / "audio.wav").write_bytes(voice.to_wav(tone, 24000))
        (folder / "times.json").write_text(json.dumps(LINES, ensure_ascii=False))
        if title:
            (folder / "song.json").write_text(json.dumps({"title": title}))
    (root / "unfinished").mkdir()  # no audio, no timings: not a song yet
    return load_library(root)


def test_the_library_holds_finished_songs_only(tmp_path):
    library = make_library(tmp_path)
    assert list(library) == ["night-city", "sun-shines"]
    assert library["sun-shines"].title == "Sun shines" and library["night-city"].title == "night-city"
    assert library["sun-shines"].first_line == "Сонца свеціць нам у вочы,"  # the sung half of a bilingual line


def test_a_song_is_found_by_its_id_or_unambiguous_words(tmp_path):
    library = make_library(tmp_path)
    assert find_song(library, "sun-shines").key == "sun-shines"
    assert find_song(library, "Night city").key == "night-city"
    assert find_song(library, "a song about the moon") is None


def test_the_caption_follows_the_sung_line():
    assert [caption_at(LINES, t) for t in (0.1, 0.5, 1.1, 1.5)] == [
        "",
        "Сонца свеціць нам у вочы,",
        "",
        "Едзем мы да мора.",
    ]


def test_the_repertoire_lists_the_songs_and_how_to_ask_for_one(tmp_path):
    text = repertoire(make_library(tmp_path))
    assert "'sun-shines'" in text and "'night-city'" in text and "action.kind to 'sing'" in text
    assert "no songs" in repertoire({})


def test_only_a_singer_conversation_accepts_the_sing_action():
    raw = json.dumps({"say": "Here it is!", "expression": "happy", "gesture": "dance",
                      "action": {"kind": "sing", "instruction": "sun-shines"}})  # fmt: skip
    assert parse_reply(raw).action == {"kind": "none", "instruction": ""}
    assert parse_reply(raw, ACTIONS + ("sing",)).action == {"kind": "sing", "instruction": "sun-shines"}
    singer = Conversation(lambda m, s: raw, actions=ACTIONS + ("sing",))
    assert "sing" in singer.schema["properties"]["action"]["properties"]["kind"]["enum"]
    assert singer.respond("sing please").action["kind"] == "sing"
