"""voice.py against a local fake of the OpenAI audio wire (what Kokoro-FastAPI and Speaches speak)."""

import email.parser
import email.policy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from humanoid_companion import voice


@pytest.fixture
def fake_server(monkeypatch):
    seen = []
    tone = voice.to_wav(np.sin(np.linspace(0, 440 * 2 * np.pi, 24000)).astype(np.float32) * 0.3, 24000)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            seen.append((self.path, self.headers["content-type"], body))
            if self.path == "/v1/audio/speech":
                out, ctype = tone, "audio/wav"
            else:
                out, ctype = json.dumps({"text": " Hello, I am your humanoid. "}).encode(), "application/json"
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("HUMANOID_TTS_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("HUMANOID_STT_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1/")
    yield seen, tone
    srv.shutdown()


def test_speak_sends_the_openai_speech_request(fake_server):
    seen, tone = fake_server
    assert voice.speak("Hi there") == tone
    path, ctype, body = seen[0]
    assert path == "/v1/audio/speech" and ctype == "application/json"
    assert json.loads(body) == {"model": "tts-1", "input": "Hi there", "voice": "af_heart", "response_format": "wav"}


def test_transcribe_uploads_a_multipart_wav_and_trims_the_text(fake_server):
    seen, tone = fake_server
    assert voice.transcribe(tone) == "Hello, I am your humanoid."
    path, ctype, body = seen[0]
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {ctype}\r\n\r\n".encode() + body)
    parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
    assert path == "/v1/audio/transcriptions"
    assert parts["model"].get_content().strip() == "whisper-1"
    assert parts["file"].get_content_type() == "audio/wav" and parts["file"].get_content() == tone


def test_wav_round_trip_and_duration():
    x = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    samples, rate = voice.from_wav(voice.to_wav(x, 16000))
    assert rate == 16000 and np.abs(samples - x).max() < 1e-4
    assert voice.duration_s(voice.to_wav(x, 16000)) == pytest.approx(1.0)


def test_roundtrip_matches_ignoring_case_and_punctuation(fake_server, tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "REPORT", tmp_path / "voice.md")
    row = voice.roundtrip("hello I am your Humanoid")
    assert row["match"] and row["audio_s"] == pytest.approx(1.0)
    voice.append_report(row, "fake")
    assert "| match | fake |" in (tmp_path / "voice.md").read_text()
    assert not voice.roundtrip("goodbye")["match"]


def test_silence_lasts_about_as_long_as_the_text_takes_to_say():
    assert voice.duration_s(voice.silence_for("x" * 45)) == pytest.approx(3.0)
    assert voice.duration_s(voice.silence_for("Hi")) == pytest.approx(1.0)
    samples, _ = voice.from_wav(voice.silence_for("Hello there"))
    assert not samples.any()
