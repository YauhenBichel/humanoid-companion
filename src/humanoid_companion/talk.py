"""Talk to the robot: face, voice and walking together.

    python -m humanoid_companion.talk --open                         # face + live body side by side; type to it
    python -m humanoid_companion.talk --open --mic                   # press Enter, speak for --listen seconds
    python -m humanoid_companion.talk --text "Hi!" --text "Walk forward a little, then turn left." \
        --record demo/                                    # scripted, recorded as one split-screen video
    python -m humanoid_companion.talk --open --teammate byte          # Byte, who explains computer science
    python -m humanoid_companion.talk --open --teammate tempo --songs my-songs/   # Tempo sings from a song folder
    python -m humanoid_companion.talk --open --teammate alesia --songs my-songs/  # so do the singers Alesia and Maks

Per turn: face "thinking" -> conversation reply (your LLM) -> in parallel the speech (Kokoro) and,
when asked to walk, the walking plan -> the robot speaks *while* its body acts: the face page plays
the voice with lip sync and shows the body's camera (`/?view=robot`); the body runs the 50 Hz
control loop in C MuJoCo in real time. The body persists for the whole conversation (it stays where
it walked and keeps balancing between turns). Every conversation ends with a short Belarusian
farewell. Without a speech server the robot still shows its captions and moves; it is just silent. With --record, each turn becomes a video: face (drawn by humanoid_companion.face.render, mouth
following the voice) | body camera, with the voice; talk.mp4 joins them.

A teammate (humanoid_companion.teammates) brings its persona, voice and face colours. With --songs
(humanoid_companion.songs) the robot can sing: asked for a song, it answers, then plays the song on
the face page with each sung line as the caption while the body dances standing.
"""

import argparse
import io
import json
import os
import subprocess
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from humanoid_companion import voice
from humanoid_companion.conversation import ACTIONS, NAME, Conversation, Reply, persona
from humanoid_companion.face import FaceServer
from humanoid_companion.face.look import DEFAULT_LOOK
from humanoid_companion.gestures import GESTURES, ease, envelope, overlay, talking_head
from humanoid_companion.songs import Song, caption_at, find_song, load_library, repertoire
from humanoid_companion.teammates import all_teammates

FPS = 25
PRELUDE_S = 1.2   # recorded "thinking" beat before each answer, showing what was said

POLICIES = Path(__file__).parent / "policies"

# The person's name in Belarusian (the vocative, e.g. "Яўген" for Yauhen), for the farewell.
NAME_BE = os.environ.get("HUMANOID_USER_NAME_BE", "").strip()


def farewell_lines(name: str = NAME, name_be: str = NAME_BE) -> dict:
    """Spoken by the Belarusian voice (github.com/YauhenBichel/belarusian-tts: OmniVoice, a native
    reference speaker). Fallback when that server is unreachable: the English voice reading a
    phonetic Latin spelling, which sounds foreign, so it is only a fallback."""
    be = f"Дзякуй, {name_be}! Да сустрэчы!" if name_be else "Дзякуй! Да сустрэчы!"
    en = f"Thank you, {name}! See you soon!" if name_be and name else "Thank you! See you soon!"
    return {"belarusian": be, "caption": f"{be} ({en})",
            "phonetic": f"Dzyakuy, {name}! Da sustrechy!" if name else "Dzyakuy! Da sustrechy!"}


FAREWELL = farewell_lines()


def jpeg(img: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=80)
    return buf.getvalue()


class Body:
    """The simulated body for a whole conversation: one MuJoCo state, a torso-tracking camera."""

    def __init__(self, policy, model, face: FaceServer, keep_frames: bool):
        from humanoid_companion.robot_io import MujocoRobotIO
        from humanoid_companion.run_sim import Recorder

        self.policy, self.face, self.keep_frames = policy, face, keep_frames
        self.io = MujocoRobotIO(model, ctrl_dt=policy.spec.ctrl_dt)
        self.cam = Recorder(self.io)
        # Warm up now: the first render takes ~0.3 s (GL start-up). Inside the real-time control
        # loop that stall is ~14 periods late, and the loop's watchdog rightly stops the robot.
        jpeg(self.view())

    def view(self) -> np.ndarray:
        self.cam.renderer.update_scene(self.io.data, camera=self.cam.camera)
        return self.cam.renderer.render()

    def run(self, command_at, seconds: float, realtime: bool, gesture_at=None) -> tuple[dict, list[np.ndarray]]:
        """Run the control loop (with head/arm gestures layered on, humanoid_companion.gestures); every 2nd
        period (25 fps) push the camera to viewers and keep it."""
        from humanoid_companion.run_sim import simulate

        frames, body = [], self

        class Sink:
            def capture(self, i):
                if i % 2:
                    return
                img = body.view()
                if body.keep_frames:
                    frames.append(img)
                if body.face.viewers:
                    body.face.push_frame(jpeg(img))

        return simulate(self.policy, command_at, seconds, io=self.io, realtime=realtime, recorder=Sink(),
                        gesture_at=gesture_at), frames


class Robot:
    def __init__(self, args):
        self.args = args
        self.teammate = all_teammates().get(getattr(args, "teammate", None) or "")
        songs = getattr(args, "songs", None)
        self.library = load_library(songs) if songs else {}
        look = self.teammate.look if self.teammate else DEFAULT_LOOK
        title = f"{self.teammate.name}, humanoid teammate" if self.teammate else "Humanoid face"
        self.face = FaceServer(port=args.port, look=look, title=title).start()
        self.conv = Conversation(system=self.system_prompt(), actions=ACTIONS + (("sing",) if self.library else ()))
        # Only a teammate names its voice; otherwise voice.speak's default (HUMANOID_TTS_VOICE) applies.
        self.voice_options = {"voice": self.teammate.voice} if self.teammate else {}
        self.look = look
        self.record = args.record
        self.turns: list[dict] = []
        self.body = None
        self.silent = False
        if args.policy:
            from humanoid_companion.policy import NumpyPolicy
            from humanoid_companion.run_sim import training_model

            self.body = Body(NumpyPolicy.load(args.policy), training_model(), self.face, keep_frames=bool(args.record))
        self.min_speed = 0.0
        if args.speed_sweep:
            usable = json.loads(args.speed_sweep.read_text())["usable_range"]
            self.min_speed = usable[0] if usable else 0.0
        if self.record:
            from humanoid_companion.face.render import FaceRenderer

            self.record.mkdir(parents=True, exist_ok=True)
            self.painter = FaceRenderer(640, 480, FPS, look=self.look)

    def system_prompt(self) -> str:
        songs = repertoire(self.library) if self.library else ""
        return self.teammate.persona(NAME, more_role=songs) if self.teammate else persona(NAME, role=songs)

    # --- senses ---
    def hear(self) -> tuple[str, str | None]:
        """(what was said, what Whisper heard) — typed input has no transcript."""
        self.face.set_expression("listening", caption="")
        if self.args.mic:
            input(f"press Enter and speak for {self.args.listen:g} s… ")
            heard = voice.transcribe(voice.record(self.args.listen))
            print(f"(heard) {heard}")
            return heard, heard
        return input("you> ").strip(), None

    # --- one turn ---
    def turn(self, said: str, heard: str | None) -> None:
        self.face.set_expression("thinking", caption="")
        reply = self.conv.respond(said)
        print(f"robot [{reply.expression}, {reply.gesture}]> {reply.say}")
        walking = reply.action["kind"] == "walk" and self.body is not None
        with ThreadPoolExecutor(2) as pool:          # the voice and the walk plan at the same time
            wav_job = pool.submit(self._speak, reply.say)
            plan_job = pool.submit(self._plan, reply.action["instruction"]) if walking else None
            wav, plan = wav_job.result(), (plan_job.result() if plan_job else None)
        self.perform(said, heard, reply, reply.say, wav, plan)
        if reply.action["kind"] == "sing":
            song = find_song(self.library, reply.action["instruction"])
            if song:
                self.sing(song)
            else:
                print(f"  (no song {reply.action['instruction']!r} in the library)")

    def sing(self, song: Song) -> None:
        """Play a song from the library: the face page plays it with the sung line as the caption, the body
        dances standing (the dance gesture's arms, over and over; no steps)."""
        from humanoid_companion.perform import RATE, decode_audio

        samples = decode_audio(song.audio, RATE)
        wav, seconds = voice.to_wav(samples, RATE), len(samples) / RATE
        print(f"  singing {song.key} ({seconds:.0f} s)")
        live = bool(self.face.viewers)
        uid = self.face.say(wav, caption=song.first_line, expression="happy")
        finished = threading.Event()

        def follow_the_lines() -> None:
            started, shown = time.monotonic(), None
            while not finished.wait(0.1):
                line = caption_at(song.lines, time.monotonic() - started)
                if line != shown:
                    self.face.set_expression("happy", caption=line)
                    shown = line

        captions = threading.Thread(target=follow_the_lines, daemon=True)
        if live:
            captions.start()
        player = None
        if not live and self.args.speaker:
            player = threading.Thread(target=voice.play, args=(wav,))
            player.start()
        result = {}
        if self.body:
            dance = GESTURES["dance"]
            arms = overlay(self.body.policy.spec.actuator_names, lambda t: dance.offsets(t % dance.duration))
            standing = np.zeros(3, np.float32)
            result, _ = self.body.run(lambda t: standing, seconds, realtime=live or self.args.speaker, gesture_at=arms)
        if live:
            self.face.wait_played(uid, seconds + 5)
        if player:
            player.join()
        finished.set()
        self.face.set_expression("happy", caption="")
        self.turns.append({"n": len(self.turns) + 1, "song": song.key, "title": song.title, "seconds": round(seconds, 2),
                           "safety_stops": result.get("safety_stops", [])})
        if self.record:
            (self.record / "conversation.json").write_text(json.dumps({"name": NAME, "turns": self.turns}, indent=2))

    def _speak(self, text: str) -> bytes:
        """The voice, or silence as long as the text when no speech server answers."""
        try:
            return voice.speak(text, **self.voice_options)
        except OSError as e:
            if not self.silent:
                print(f"(no speech server at {voice.tts_url()}: {e}; captions only)")
                self.silent = True
            return voice.silence_for(text)

    def farewell(self) -> None:
        reply = Reply(say=FAREWELL["caption"], expression="happy", gesture="wave", source="scripted")
        try:
            wav, reply.routing = voice.speak(FAREWELL["belarusian"], lang="be"), {"voice": "omnivoice-be"}
        except OSError as e:   # Belarusian server down: say it anyway, with the accented fallback
            wav, reply.routing = self._speak(FAREWELL["phonetic"]), {"voice": "kokoro-phonetic"}
            reply.error = f"belarusian voice unavailable: {e}"
        print(f"robot [happy]> {reply.say}  ({reply.routing['voice']})")
        self.perform("", None, reply, FAREWELL["caption"], wav, None)

    def _plan(self, instruction: str):
        from humanoid_companion.brain import plan

        return plan(instruction, speed_cap=self.args.speed_cap, min_speed=self.min_speed)

    def perform(self, said: str, heard: str | None, reply: Reply, caption: str, wav: bytes, plan) -> None:
        """Speak and act at the same time; record the turn as one video if asked."""
        from humanoid_companion.run_sim import schedule

        n = len(self.turns) + 1
        speech_s = voice.duration_s(wav)
        samples, rate = voice.from_wav(wav)
        g = GESTURES.get(reply.gesture)
        # Arm gestures and dance are measured unsafe while walking: they come first, then the walk.
        delay = g.duration if (g and plan is not None and (not g.walk_ok or g.command)) else 0.0
        walk_at, walk_s = schedule(plan.commands) if plan is not None else (None, 0.0)

        def command_at(t):
            if g and g.command and t < g.duration:
                return np.asarray(g.command(t), np.float32)
            if walk_at and delay <= t < delay + walk_s:
                return walk_at(t - delay)
            return np.zeros(3, np.float32)

        names = self.body.policy.spec.actuator_names if self.body else ()
        parts = [talking_head(envelope(samples, rate))] + ([g.offsets] if g else [])
        gesture_at = overlay(names, *parts) if self.body else None
        total = max(speech_s + 0.4, delay + walk_s + 0.5 if plan is not None else 0.0, g.duration + 0.3 if g else 0.0)

        prelude_frames = []
        if self.record and self.body and said:        # the "thinking" beat: head tilted, body balancing
            think = overlay(names, lambda t: {"head_tilt_act": -0.15 * ease(t, PRELUDE_S), "head_pan_act": 0.2 * ease(t, PRELUDE_S)})
            _, prelude_frames = self.body.run(lambda t: np.zeros(3, np.float32), PRELUDE_S, realtime=False, gesture_at=think)

        live = bool(self.face.viewers)
        uid = self.face.say(wav, caption=caption, expression=reply.expression)
        player = None
        if not live and self.args.speaker:
            player = threading.Thread(target=voice.play, args=(wav,))
            player.start()
        result, frames = (self.body.run(command_at, total, realtime=live or self.args.speaker, gesture_at=gesture_at)
                          if self.body else ({}, []))
        if live:
            self.face.wait_played(uid, speech_s + 5)
        if player:
            player.join()
        self.face.set_expression("neutral", caption="")

        row = {"n": n, "said": said, "heard": heard, "say": caption, "expression": reply.expression,
               "gesture": reply.gesture, "walk_delay_s": delay, "action": reply.action, "stop": reply.stop, "source": reply.source, "error": reply.error,
               "latency_s": reply.latency_s, "routing": reply.routing, "speech_s": round(speech_s, 2)}
        if plan is not None:
            row["walk"] = {"instruction": reply.action["instruction"], "plan_source": plan.source,
                           "plan_routing": plan.routing, "plan_clamped": plan.clamped,
                           "commands": [asdict(c) for c in plan.commands], "seconds": result.get("seconds"),
                           "forward_distance_m": result.get("forward_distance_m"),
                           "yaw_change_rad": result.get("yaw_change_rad"), "safety_stops": result.get("safety_stops", [])}
            print(f"  walked {row['walk']['forward_distance_m']} m, turned {row['walk']['yaw_change_rad']} rad")
        if self.record:
            row["audio"] = f"turn-{n:02d}.wav"
            (self.record / row["audio"]).write_bytes(wav)
            row["video"] = self._record_turn(n, said, reply.expression, caption, wav, prelude_frames, frames, total)
        self.turns.append(row)
        if self.record:
            (self.record / "conversation.json").write_text(json.dumps({"name": NAME, "turns": self.turns}, indent=2))

    def _record_turn(self, n, said, expression, caption, wav, prelude_frames, frames, total) -> str:
        """Face (drawn, mouth from the voice) | body camera, with the voice after the prelude."""
        import mediapy

        from humanoid_companion.face.render import mouth_track

        samples, rate = voice.from_wav(wav)
        body = prelude_frames + frames or [np.zeros((480, 640, 3), np.uint8)] * int(total * FPS)
        pre = len(prelude_frames)
        mouth = mouth_track(samples, rate, FPS, len(body) - pre)
        speaking_frames = int((len(samples) / rate + 0.4) * FPS)
        out = []
        for k, b in enumerate(body):
            if k < pre:
                face = self.painter.frame("thinking", 0.0, f"“{said}”")
            else:
                j = k - pre
                face = self.painter.frame(expression, float(mouth[j]), caption if j < speaking_frames else "")
            out.append(np.concatenate([face, b], axis=1))
        silent = self.record / f"_turn-{n:02d}-video.mp4"
        padded = self.record / f"_turn-{n:02d}-audio.wav"
        mediapy.write_video(silent, out, fps=FPS)
        padded.write_bytes(voice.to_wav(np.concatenate([np.zeros(int(pre / FPS * rate), np.float32), samples]), rate))
        name = f"turn-{n:02d}.mp4"
        _ffmpeg("-i", str(silent), "-i", str(padded), "-filter_complex", "[1:a]apad[a]", "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-ar", "24000", "-ac", "1", "-shortest", str(self.record / name))
        silent.unlink()
        padded.unlink()
        return name


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def compose(record: Path) -> Path:
    """talk.mp4: the recorded turns' videos (each with its voice), in order."""
    turns = json.loads((record / "conversation.json").read_text())["turns"]
    listing = record / "_list.txt"
    listing.write_text("".join(f"file '{t['video']}'\n" for t in turns if t.get("video")))
    out = record / "talk.mp4"
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out))
    listing.unlink()
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", action="append", help="scripted things to say (repeatable); otherwise interactive")
    p.add_argument("--mic", action="store_true", help="speak instead of typing (needs the 'voice' extra)")
    p.add_argument("--listen", type=float, default=5.0, help="seconds recorded per spoken turn")
    p.add_argument("--speaker", action="store_true", help="play speech on this machine when no face page is open")
    p.add_argument("--open", action="store_true", help="open face + body (/?view=robot) in the default browser")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--policy", type=Path, default=POLICIES / "op3_walk.npz", help="default: the bundled OP3 walking policy")
    p.add_argument("--speed-sweep", type=Path, default=POLICIES / "op3_walk.speed-sweep.json",
                   help="the policy's measured speed range (humanoid_companion.speed_sweep)")
    p.add_argument("--speed-cap", type=float, default=0.5)
    p.add_argument("--no-farewell", action="store_true", help="skip the Belarusian goodbye at the end")
    p.add_argument("--record", type=Path, help="save transcript, audio and a split-screen video per turn here")
    p.add_argument("--teammate", choices=sorted(all_teammates()),
                   help="talk to a teammate: byte (computer science), tempo, alesia or maks (songs), or one of your own")
    p.add_argument("--songs", type=Path, help="a song library folder (humanoid_companion.songs); "
                   "default: the teammate's [songs] entry in settings.toml")
    args = p.parse_args(argv)
    if args.teammate and args.songs is None:
        from humanoid_companion.settings import song_library

        args.songs = song_library(args.teammate)
    args.policy = args.policy if args.policy.exists() else None
    args.speed_sweep = args.speed_sweep if args.speed_sweep.exists() else None

    robot = Robot(args)
    print(f"face + body: {robot.face.url}/?view=robot  (click it once so it may speak)")
    if args.open:
        webbrowser.open(f"{robot.face.url}/?view=robot")
        time.sleep(4)
    try:
        if args.text:
            for said in args.text:
                print(f"you> {said}")
                robot.turn(said, None)
        else:
            while True:
                said, heard = robot.hear()
                if said.lower() in ("quit", "exit", "bye", "goodbye"):
                    break
                robot.turn(said, heard)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if not args.no_farewell:
            robot.farewell()
        robot.face.set_expression("sleeping", caption="")
        time.sleep(0.5)
        robot.face.stop()
    if args.record and robot.turns:
        print(f"wrote {compose(args.record)}")


if __name__ == "__main__":
    main()
