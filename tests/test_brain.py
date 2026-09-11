import json

import pytest

from humanoid_companion.brain import MAX_COMMANDS, SCHEMA, STOP, plan


def answer(*commands):
    return lambda messages, schema: json.dumps({"commands": [dict(zip(("vx", "vy", "yaw_rate", "duration_s"), c)) for c in commands]})


def never_called(messages, schema):
    raise AssertionError("the model must not be asked")


def test_a_good_answer_becomes_commands_and_the_model_gets_the_schema():
    seen = {}

    def complete(messages, schema):
        seen.update(messages=messages, schema=schema)
        return answer((0.2, 0.0, 0.0, 3.0), (0.0, 0.0, 0.5, 3.0))(messages, schema)

    p = plan("walk forward slowly, then turn left", complete)
    assert p.source == "model" and not p.clamped
    assert [c.as_array() + [c.duration_s] for c in p.commands] == [[0.2, 0, 0, 3.0], [0, 0, 0.5, 3.0]]
    assert seen["schema"] is SCHEMA and seen["messages"][-1]["content"].startswith("walk forward")


def test_out_of_range_values_are_clamped_and_reported():
    p = plan("run!", answer((3.0, -2.0, 5.0, 60.0)), speed_cap=0.5)
    c = p.commands[0]
    assert (c.vx, c.vy, c.yaw_rate, c.duration_s) == (0.5, -0.5, 0.7, 10.0)
    assert len(p.clamped) == 4


def test_backwards_is_limited_by_the_trained_range_and_the_cap():
    assert plan("back", answer((-5.0, 0, 0, 1)), speed_cap=1.0).commands[0].vx == -0.6
    assert plan("back", answer((-5.0, 0, 0, 1)), speed_cap=0.3).commands[0].vx == -0.3


def test_slow_commands_are_raised_to_the_policys_slowest_walk_but_zero_stays_zero():
    p = plan("amble", answer((0.2, 0, 0, 3), (0.0, 0, 0.5, 3), (-0.1, 0, 0, 1)), min_speed=0.3)
    assert [c.vx for c in p.commands] == [0.3, 0.0, -0.3]
    assert sum("slowest walk" in n for n in p.clamped) == 2


def test_the_minimum_never_exceeds_the_speed_cap():
    assert plan("amble", answer((0.1, 0, 0, 1)), speed_cap=0.2, min_speed=0.3).commands[0].vx == 0.2


def test_too_many_commands_are_cut():
    p = plan("dance", answer(*[(0.1, 0, 0, 1)] * 25))
    assert len(p.commands) == MAX_COMMANDS and any("25 commands" in n for n in p.clamped)


@pytest.mark.parametrize("instruction", ["stop", "Halt now!", "FREEZE", "please stop walking", ""])
def test_stop_words_and_empty_input_never_reach_the_model(instruction):
    p = plan(instruction, never_called)
    assert p.commands == [STOP] and p.source == "stop-word"


@pytest.mark.parametrize("raw", [
    "not json at all",
    json.dumps({"commands": []}),
    json.dumps({"commands": [{"vx": 0.2, "vy": 0, "yaw_rate": 0}]}),              # missing duration
    json.dumps({"commands": [{"vx": "fast", "vy": 0, "yaw_rate": 0, "duration_s": 1}]}),
    '{"commands": [{"vx": NaN, "vy": 0, "yaw_rate": 0, "duration_s": 1}]}',
    json.dumps({"commands": [{"vx": True, "vy": 0, "yaw_rate": 0, "duration_s": 1}]}),
    json.dumps(["vx", 0.2]),
])
def test_unusable_answers_fall_back_to_standing_still(raw):
    p = plan("walk", lambda m, s: raw)
    assert p.commands == [STOP] and p.source == "fallback" and p.error


def test_routing_is_kept_with_the_plan():
    from humanoid_companion.brain import Completion

    route = {"backend": "local", "rule": "", "model": "llama3.1:8b"}
    ok = plan("walk", lambda m, s: Completion(answer((0.2, 0, 0, 1))(m, s), route))
    bad = plan("walk", lambda m, s: Completion("nope", route))
    assert ok.routing == route and ok.source == "model"
    assert bad.routing == route and bad.source == "fallback"


def test_gateway_down_falls_back_to_standing_still():
    def down(messages, schema):
        raise ConnectionRefusedError(61, "Connection refused")

    p = plan("walk forward", down)
    assert p.commands == [STOP] and p.source == "fallback" and "Connection refused" in p.error


def test_complete_json_speaks_the_openai_chat_wire(monkeypatch):
    """Against a local fake server: the schema, model and key go out; the answer and routing come back."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from humanoid_companion.brain import complete_json

    seen = {}

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["auth"] = self.path, self.headers.get("authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            out = json.dumps({"model": "llama3.1:8b", "choices": [{"message": {"content": '{"commands": []}'}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("x-router-backend", "local")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("HUMANOID_LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1/")
    monkeypatch.setenv("HUMANOID_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("HUMANOID_LLM_API_KEY", "test-key")
    try:
        c = complete_json([{"role": "user", "content": "walk"}], SCHEMA, name="walk_plan")
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "llama3.1:8b" and seen["body"]["response_format"]["json_schema"]["schema"] == SCHEMA
    assert c.text == '{"commands": []}' and c.routing == {"model": "llama3.1:8b", "backend": "local"}
