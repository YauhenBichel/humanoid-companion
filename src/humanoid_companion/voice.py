"""The robot's ears and mouth: any OpenAI-compatible speech and transcription servers.

    python -m humanoid_companion.voice --say "Hello, I am your humanoid."          # speak through the speaker
    python -m humanoid_companion.voice --listen 5                                   # record 5 s, print the text
    python -m humanoid_companion.voice --roundtrip "Hello, I am your humanoid."    # speak, transcribe back, time both
                                                                                    # -> voice-latency.md

Speech (/v1/audio/speech): HUMANOID_TTS_BASE_URL, default http://127.0.0.1:8880/v1 (Kokoro-FastAPI;
the default voice `af_heart` is a Kokoro voice, HUMANOID_TTS_VOICE picks another).
Transcription (/v1/audio/transcriptions): HUMANOID_STT_BASE_URL, default http://127.0.0.1:8000/v1
(Speaches, faster-whisper). Belarusian speech: HUMANOID_BE_TTS_URL, default
http://127.0.0.1:11810/v1 (github.com/YauhenBichel/belarusian-tts).

HTTP is stdlib only. Microphone and speaker use `sounddevice` (optional extra `voice`), imported
only by record() and play(), so the rest works on machines without audio.
"""

import argparse
import io
import json
import os
import re
import time
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

DEFAULT_VOICE = os.environ.get("HUMANOID_TTS_VOICE", "af_heart")
REPORT = Path("voice-latency.md")


def tts_url() -> str:
    return os.environ.get("HUMANOID_TTS_BASE_URL", "http://127.0.0.1:8880/v1").rstrip("/")


def stt_url() -> str:
    return os.environ.get("HUMANOID_STT_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def be_tts_url() -> str:
    """The Belarusian speech server (github.com/YauhenBichel/belarusian-tts: OmniVoice with a
    native Belarusian reference voice)."""
    return os.environ.get("HUMANOID_BE_TTS_URL", "http://127.0.0.1:11810/v1").rstrip("/")


def speak(text: str, voice: str = DEFAULT_VOICE, timeout_s: float = 60.0, lang: str = "en") -> bytes:
    """Text -> WAV bytes (OpenAI /v1/audio/speech wire). English: the speech server (Kokoro).
    Belarusian (lang="be"): OmniVoice with a native Belarusian reference voice; slow on CPU the
    first time a sentence is said (cached after that), hence the longer timeout."""
    if lang == "be":
        base, timeout_s = be_tts_url(), max(timeout_s, 300.0)
        body = json.dumps({"input": text, "response_format": "wav"}).encode()
    else:
        base = tts_url()
        body = json.dumps({"model": "tts-1", "input": text, "voice": voice, "response_format": "wav"}).encode()
    req = urllib.request.Request(f"{base}/audio/speech", data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.read()


def multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"humanoid-{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                 f"Content-Type: {content_type}\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(wav: bytes, timeout_s: float = 120.0, language: str | None = None) -> str:
    """WAV bytes -> text (OpenAI /v1/audio/transcriptions wire). `language`: ISO code hint, e.g. "be"."""
    fields = {"model": "whisper-1", **({"language": language} if language else {})}
    body, ctype = multipart(fields, "file", "speech.wav", wav, "audio/wav")
    req = urllib.request.Request(f"{stt_url()}/audio/transcriptions", data=body, headers={"content-type": ctype})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.load(r)["text"].strip()


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def from_wav(wav: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
        return pcm, w.getframerate()


def silence_for(text: str, rate: int = 24000) -> bytes:
    """A silent WAV about as long as `text` takes to say (~15 characters a second): the robot
    still shows its caption and moves its body when no speech server is running."""
    return to_wav(np.zeros(int(max(1.0, len(text) / 15) * rate), np.float32), rate)


def duration_s(wav: bytes) -> float:
    samples, rate = from_wav(wav)
    return len(samples) / rate


def record(seconds: float, rate: int = 16000) -> bytes:
    import sounddevice as sd

    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    return to_wav(audio[:, 0], rate)


def play(wav: bytes) -> None:
    import sounddevice as sd

    samples, rate = from_wav(wav)
    sd.play(samples, rate)
    sd.wait()


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def roundtrip(text: str) -> dict:
    t = time.monotonic()
    wav = speak(text)
    tts_s = time.monotonic() - t
    t = time.monotonic()
    heard = transcribe(wav)
    stt_s = time.monotonic() - t
    return {"text": text, "transcript": heard, "audio_s": round(duration_s(wav), 2),
            "tts_s": round(tts_s, 2), "stt_s": round(stt_s, 2), "match": normalise(text) == normalise(heard)}


def append_report(row: dict, note: str = "") -> None:
    if not REPORT.exists():
        REPORT.write_text(
            "# Voice round trips\n\n`python -m humanoid_companion.voice --roundtrip \"...\"`: Kokoro speaks the text, Whisper "
            f"transcribes it back (speech: {tts_url()}, transcription: {stt_url()}).\n\n"
            "| time | text | transcript | audio s | speak s | transcribe s | result | note |\n"
            "|---|---|---|---|---|---|---|---|\n")
    with REPORT.open("a") as f:
        f.write(f"| {time.strftime('%Y-%m-%d %H:%M')} | {row['text']} | {row['transcript']} | {row['audio_s']} | "
                f"{row['tts_s']} | {row['stt_s']} | {'match' if row['match'] else 'differs'} | {note} |\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--say")
    g.add_argument("--listen", type=float, metavar="SECONDS")
    g.add_argument("--roundtrip")
    p.add_argument("--note", default="", help="context for the report row, e.g. 'GPU busy training'")
    args = p.parse_args(argv)
    if args.say:
        play(speak(args.say))
    elif args.listen:
        print(transcribe(record(args.listen)))
    else:
        row = roundtrip(args.roundtrip)
        append_report(row, args.note)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
