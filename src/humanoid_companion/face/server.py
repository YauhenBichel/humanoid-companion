"""The face server: serves the face page and pushes expressions and speech to it (stdlib only).

    GET  /                  the face (face.html)
    GET  /events            server-sent events: {"type": "state"|"say", ...}
    GET  /audio/<id>.wav    the audio of utterance <id>
    GET  /body.mjpg         the simulated body's camera as an MJPEG stream (push_frame)
    POST /state             {"expression": "...", "caption": "..."}  (manual control, tests)
    POST /played            {"id": "..."}  sent by the page when an utterance has finished

In Python: FaceServer().start(); face.set_expression("happy"); uid = face.say(wav, "Hello!");
face.wait_played(uid, timeout_s); face.push_frame(jpeg_bytes).
Open `/?view=robot` for face and body side by side; `/` is the face alone (the head display).
"""

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

TABLE_FILE = Path(__file__).with_name("expressions.json")
TABLE = {k: v for k, v in json.loads(TABLE_FILE.read_text()).items() if not k.startswith("_")}
EXPRESSIONS = tuple(TABLE)
PAGE = Path(__file__).with_name("face.html")
MAX_AUDIO = 32  # utterances kept for late or reconnecting pages


def page_html() -> bytes:
    """face.html with the shared expression table put in."""
    return PAGE.read_text().replace("/*EXPRESSIONS*/{}", json.dumps(TABLE)).encode()


class FaceServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self._subscribers: list[queue.Queue] = []
        self._audio: dict[str, bytes] = {}
        self._played: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._frame: bytes | None = None
        self._frame_cond = threading.Condition()
        self._frame_seq = 0
        self.last_state = {"type": "state", "expression": "neutral", "caption": ""}
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FaceServer":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # --- what the runtime calls ---
    def set_expression(self, expression: str, caption: str | None = None) -> None:
        if expression not in EXPRESSIONS:
            raise ValueError(f"unknown expression {expression!r}; one of {EXPRESSIONS}")
        msg = {"type": "state", "expression": expression}
        if caption is not None:
            msg["caption"] = caption
        self.last_state = {**self.last_state, **msg}
        self._publish(msg)

    def say(self, wav: bytes, caption: str = "", expression: str = "neutral") -> str:
        if expression not in EXPRESSIONS:
            expression = "neutral"
        uid = uuid.uuid4().hex[:12]
        with self._lock:
            self._audio[uid] = wav
            self._played[uid] = threading.Event()
            while len(self._audio) > MAX_AUDIO:
                old = next(iter(self._audio))
                self._audio.pop(old)
                self._played.pop(old, None)
        self._publish({"type": "say", "id": uid, "audio": f"/audio/{uid}.wav", "caption": caption,
                       "expression": expression})
        return uid

    def push_frame(self, jpeg: bytes) -> None:
        """The body's latest camera image (JPEG), for /body.mjpg viewers."""
        with self._frame_cond:
            self._frame, self._frame_seq = jpeg, self._frame_seq + 1
            self._frame_cond.notify_all()

    def wait_played(self, uid: str, timeout_s: float) -> bool:
        ev = self._played.get(uid)
        return bool(ev and ev.wait(timeout_s))

    @property
    def viewers(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def _publish(self, msg: dict) -> None:
        with self._lock:
            for q in self._subscribers:
                q.put(msg)

    # --- HTTP ---
    def _handler(self):
        face = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(body)))
                self.send_header("cache-control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json_body(self) -> dict | None:
                try:
                    data = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    return None

            def do_GET(self):
                path = urlsplit(self.path).path  # "/?kiosk=1" is the page too
                if path == "/":
                    self._send(200, page_html(), "text/html; charset=utf-8")
                elif path.startswith("/audio/") and path.endswith(".wav"):
                    wav = face._audio.get(path[len("/audio/"):-len(".wav")])
                    self._send(200, wav, "audio/wav") if wav else self._send(404, b"no such utterance", "text/plain")
                elif path == "/events":
                    self._events()
                elif path == "/body.mjpg":
                    self._body()
                else:
                    self._send(404, b"not found", "text/plain")

            def _body(self):
                self.send_response(200)
                self.send_header("content-type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("cache-control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with face._frame_cond:
                            face._frame_cond.wait_for(lambda: face._frame_seq != seen, timeout=15)
                            jpeg, seen = face._frame, face._frame_seq
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def _events(self):
                q: queue.Queue = queue.Queue()
                with face._lock:
                    face._subscribers.append(q)
                q.put(face.last_state)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-store")
                self.end_headers()
                try:
                    while True:
                        try:
                            msg = q.get(timeout=15)
                            self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with face._lock:
                        face._subscribers.remove(q)

            def do_POST(self):
                data = self._json_body()
                if data is None:
                    return self._send(400, b"expected a JSON object", "text/plain")
                if self.path == "/state":
                    try:
                        face.set_expression(data.get("expression", ""), data.get("caption"))
                    except ValueError as e:
                        return self._send(400, str(e).encode(), "text/plain")
                    return self._send(204, b"", "text/plain")
                if self.path == "/played":
                    ev = face._played.get(str(data.get("id", "")))
                    if ev is None:
                        return self._send(404, b"no such utterance", "text/plain")
                    ev.set()
                    return self._send(204, b"", "text/plain")
                self._send(404, b"not found", "text/plain")

        return Handler
