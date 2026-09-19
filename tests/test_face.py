import json
import threading
import urllib.error
import urllib.request

import pytest

from humanoid_companion.face import EXPRESSIONS, FaceServer


@pytest.fixture
def face():
    f = FaceServer(port=0).start()
    yield f
    f.stop()


def get(face, path):
    with urllib.request.urlopen(face.url + path, timeout=5) as r:
        return r.status, r.headers["content-type"], r.read()


def post(face, path, data):
    req = urllib.request.Request(face.url + path, data=json.dumps(data).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def subscribe(face, n):
    """Read the first n events from /events in a thread."""
    events, ready = [], threading.Event()

    def run():
        with urllib.request.urlopen(face.url + "/events", timeout=10) as r:
            for line in r:
                if line.startswith(b"data: "):
                    events.append(json.loads(line[6:]))
                    ready.set()
                    if len(events) >= n:
                        return

    t = threading.Thread(target=run, daemon=True)
    t.start()
    ready.wait(5)
    return events, t


def test_the_page_has_a_canvas_and_every_expression(face):
    status, ctype, body = get(face, "/")
    assert status == 200 and ctype.startswith("text/html") and b"<canvas" in body
    assert all(f'"{e}":'.encode() in body for e in EXPRESSIONS)   # the shared table was injected
    assert b"/*EXPRESSIONS*/" not in body


def test_the_kiosk_url_of_the_head_display_is_the_page_too(face):
    assert b"<canvas" in get(face, "/?kiosk=1")[2]


def test_a_new_viewer_gets_the_current_state_then_updates(face):
    face.set_expression("thinking", caption="hmm")
    events, t = subscribe(face, 3)
    assert events[0] == {"type": "state", "expression": "thinking", "caption": "hmm"}
    assert post(face, "/state", {"expression": "happy"}) == 204
    uid = face.say(b"RIFF-fake-wav", caption="Hello!", expression="happy")
    t.join(5)
    assert events[1]["expression"] == "happy"
    assert events[2] == {"type": "say", "id": uid, "audio": f"/audio/{uid}.wav", "caption": "Hello!", "expression": "happy"}


def test_utterance_audio_is_served_and_played_is_signalled(face):
    uid = face.say(b"RIFF-fake-wav", caption="hi")
    assert get(face, f"/audio/{uid}.wav")[2] == b"RIFF-fake-wav"
    assert not face.wait_played(uid, 0.05)
    assert post(face, "/played", {"id": uid}) == 204
    assert face.wait_played(uid, 1)


def test_body_frames_stream_as_mjpeg(face):
    got = []

    def read():
        with urllib.request.urlopen(face.url + "/body.mjpg", timeout=10) as r:
            assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
            buf = b""
            while len(got) < 2:
                buf += r.read1(4096)
                while b"\xff\xd9" in buf:     # end of a JPEG
                    start = buf.index(b"\xff\xd8")
                    end = buf.index(b"\xff\xd9") + 2
                    got.append(buf[start:end])
                    buf = buf[end:]

    t = threading.Thread(target=read, daemon=True)
    t.start()
    import time
    for i in range(20):
        face.push_frame(b"\xff\xd8frame%d\xff\xd9" % i)
        time.sleep(0.05)
        if len(got) >= 2:
            break
    t.join(5)
    assert len(got) >= 2 and all(g.startswith(b"\xff\xd8frame") for g in got)


def test_bad_input_is_rejected(face):
    assert post(face, "/state", {"expression": "angry-robot"}) == 400
    assert post(face, "/played", {"id": "nope"}) == 404
    with pytest.raises(urllib.error.HTTPError):
        get(face, "/audio/nope.wav")
    with pytest.raises(ValueError):
        face.set_expression("evil")
    assert face.say(b"x", expression="evil")  # an unknown expression while speaking falls back to neutral
