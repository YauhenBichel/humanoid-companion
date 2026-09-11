import json

import pytest

from humanoid_companion.brain import Completion
from humanoid_companion.conversation import MAX_HISTORY, MAX_SAY, SCHEMA, Conversation, parse_reply, trim_spoken


def reply(say="Hello!", expression="happy", kind="none", instruction=""):
    return json.dumps({"say": say, "expression": expression, "action": {"kind": kind, "instruction": instruction}})


def test_a_reply_carries_speech_face_action_and_routing():
    seen = {}

    def complete(messages, schema):
        seen.update(messages=messages, schema=schema)
        return Completion(reply("Sure, walking now.", "happy", "walk", "walk forward for three seconds"), {"backend": "local"})

    r = Conversation(complete).respond("walk please")
    assert (r.say, r.expression, r.action, r.source) == (
        "Sure, walking now.", "happy", {"kind": "walk", "instruction": "walk forward for three seconds"}, "model")
    assert r.routing == {"backend": "local"} and seen["schema"] is SCHEMA
    assert seen["messages"][0]["role"] == "system" and seen["messages"][-1] == {"role": "user", "content": "walk please"}


def test_with_a_name_the_persona_uses_it_and_never_says_owner():
    from humanoid_companion.conversation import persona

    p = persona("Alex")
    assert "Alex is building" in p and "call Alex by name" in p
    assert "never say 'my owner'" in p and "its owner" not in p


def test_without_a_name_the_persona_names_nobody_and_still_never_says_owner():
    from humanoid_companion.conversation import persona

    p = persona("")
    assert "the person you talk with is building" in p and "Never call anyone your owner" in p
    assert " , " not in p and "  " not in p


def test_the_system_prompt_is_sent_first():
    seen = {}
    Conversation(lambda m, s: seen.setdefault("m", m) and reply(), system="be brief").respond("hi")
    assert seen["m"][0] == {"role": "system", "content": "be brief"}


def test_history_is_kept_and_bounded():
    conv = Conversation(lambda m, s: reply())
    for i in range(10):
        conv.respond(f"message {i}")
    assert len(conv.history) == MAX_HISTORY and conv.history[-2] == {"role": "user", "content": "message 9"}


@pytest.mark.parametrize("said", ["stop!", "Please halt", "freeze right there"])
def test_stop_words_stop_without_the_model(said):
    def never(m, s):
        raise AssertionError("must not ask the model")

    r = Conversation(never).respond(said)
    assert r.stop and r.action["kind"] == "none" and r.source == "stop-word"


def test_the_persona_is_upbeat_and_uses_its_body():
    from humanoid_companion.conversation import PERSONA, SCHEMA

    assert "cheerful" in PERSONA and "Mostly look happy" in PERSONA and "'wave'" in PERSONA
    assert set(SCHEMA["properties"]["gesture"]["enum"]) == {"none", "wave", "nod", "celebrate", "dance", "look_around"}


def test_a_gesture_is_kept_and_an_unknown_one_becomes_none():
    ok = parse_reply(json.dumps({"say": "Hi!", "expression": "happy", "gesture": "wave",
                                 "action": {"kind": "none", "instruction": ""}}))
    bad = parse_reply(json.dumps({"say": "Hi!", "expression": "happy", "gesture": "backflip",
                                  "action": {"kind": "none", "instruction": ""}}))
    assert ok.gesture == "wave" and bad.gesture == "none"


def test_unknown_expression_and_action_are_made_safe():
    r = parse_reply(json.dumps({"say": "Hi", "expression": "furious", "action": {"kind": "fly", "instruction": "up"}}))
    assert r.expression == "neutral" and r.action == {"kind": "none", "instruction": ""}
    r = parse_reply(reply(kind="hand", instruction="spin the cube"))   # no hand in this robot
    assert r.action == {"kind": "none", "instruction": ""}
    r = parse_reply(reply(kind="walk", instruction="   "))
    assert r.action["kind"] == "none"  # an action with no instruction is no action


@pytest.mark.parametrize("raw", ["not json", json.dumps({"say": ""}), json.dumps(["hi"]), json.dumps({"expression": "happy"})])
def test_unusable_answers_get_a_motionless_apology(raw):
    r = Conversation(lambda m, s: raw).respond("hello")
    assert r.source == "fallback" and r.action["kind"] == "none" and r.say and r.error


def test_model_server_down_gets_a_motionless_apology():
    def down(m, s):
        raise ConnectionRefusedError("refused")

    r = Conversation(down).respond("hello")
    assert r.source == "fallback" and r.expression == "sad" and r.action["kind"] == "none"


def test_long_speech_is_cut_at_a_sentence():
    text = "This is a sentence that goes on. " * 20
    out = trim_spoken(text)
    assert len(out) <= MAX_SAY and out.endswith(".")
    assert trim_spoken("short") == "short"
