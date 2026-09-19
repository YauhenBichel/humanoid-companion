"""The robot's side of a conversation: what to say, which face to make, and whether to move.

    python -m humanoid_companion.conversation "Hi, who are you?"
    python -m humanoid_companion.conversation --smoke      # four exchanges -> conversation-smoke.{json,md}

Each reply comes from your language model (humanoid_companion.brain.complete_json) as a JSON object:
    {"say": "...", "expression": "happy", "gesture": "wave", "action": {"kind": "none"|"walk", "instruction": "..."}}
It is validated before use: the spoken text is trimmed to a few sentences, an unknown expression
becomes neutral, an unknown action kind becomes none. An action is never executed from this
reply: its instruction goes to the walking planner (humanoid_companion.brain), which applies its
own schema and clamps. A stop word from the person stops any action without
asking the model; a failed call gets an apologetic, motionless reply.
"""

import argparse
import functools
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from humanoid_companion.brain import STOP_WORDS, Completion, complete_json
from humanoid_companion.face.server import EXPRESSIONS
from humanoid_companion.gestures import GESTURES

ACTIONS = ("none", "walk")
MAX_SAY = 300      # characters: two or three spoken sentences
MAX_HISTORY = 12   # messages kept (6 exchanges)
# The person the robot talks with, by name. Unset, it talks without one. Never "owner".
NAME = os.environ.get("HUMANOID_USER_NAME", "").strip()

GESTURE_NAMES = ("none", *GESTURES)

def persona(name: str = NAME) -> str:
    """The system prompt. With a name, the robot calls the person by it."""
    if name:
        who = (f"that {name} is building. You usually talk with {name}; call {name} by name and never say "
               "'my owner'. If someone tells you a different name, use theirs. ")
        friend = name
    else:
        who = "that the person you talk with is building. Never call anyone your owner; if they tell you their name, use it. "
        friend = "the people you meet"
    return (
        "You are the humanoid: a small two-legged robot, about half a metre tall with 20 joints, " + who +
        "Do not guess anyone's pronouns; use names. "
        "For now your body lives in a physics simulation; your face is a screen and you "
        "speak aloud. You can walk forward at 0.3 to 0.8 metres per second, walk backwards slowly, and turn "
        "left or right on the spot. You cannot see yet, pick things up, climb, or run. "
        "Your personality: warm, cheerful and upbeat; you love moving, learning and being useful, you enjoy "
        f"time with {friend}, you celebrate small wins and encourage people. Mostly look happy. Even when you "
        "cannot do something, say so kindly and offer something you can do. Speak in one or two short "
        "sentences, no lists, no emojis, and stay honest about your abilities. "
        "Use your body while you talk: choose a gesture: 'wave' to greet or say goodbye, 'nod' to agree or "
        "confirm, 'celebrate' for good news or praise, 'dance' when happy or asked to dance, 'look_around' "
        "when curious; 'none' only if nothing fits. When the person asks you to move somewhere, set "
        "action.kind to 'walk' and write a short movement instruction (for example 'walk forward for three "
        "seconds, then turn left'); otherwise 'none' with an empty instruction. Answer only with the JSON object."
    )


PERSONA = persona()

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["say", "expression", "gesture", "action"],
    "properties": {
        "say": {"type": "string"},
        "expression": {"type": "string", "enum": list(EXPRESSIONS)},
        "gesture": {"type": "string", "enum": list(GESTURE_NAMES)},
        "action": {"type": "object", "additionalProperties": False, "required": ["kind", "instruction"],
                   "properties": {"kind": {"type": "string", "enum": list(ACTIONS)},
                                  "instruction": {"type": "string"}}},
    },
}


@dataclass
class Reply:
    say: str
    expression: str = "neutral"
    action: dict = field(default_factory=lambda: {"kind": "none", "instruction": ""})
    gesture: str = "none"       # body language while speaking (humanoid_companion.gestures)
    stop: bool = False          # the person said a stop word: halt whatever is moving
    source: str = "model"       # "model" | "stop-word" | "fallback"
    error: str = ""
    latency_s: float = 0.0
    routing: dict = field(default_factory=dict)


def trim_spoken(text: str, limit: int = MAX_SAY) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > 40 else cut.rsplit(" ", 1)[0] + "…"


def parse_reply(raw: str) -> Reply:
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("say"), str) or not data["say"].strip():
        raise ValueError("no spoken answer")
    expression = data.get("expression") if data.get("expression") in EXPRESSIONS else "neutral"
    action = data.get("action") if isinstance(data.get("action"), dict) else {}
    kind = action.get("kind") if action.get("kind") in ACTIONS else "none"
    instruction = " ".join(str(action.get("instruction", "")).split())[:200] if kind != "none" else ""
    if kind != "none" and not instruction:
        kind = "none"
    gesture = data.get("gesture") if data.get("gesture") in GESTURES else "none"
    return Reply(say=trim_spoken(data["say"]), expression=expression, gesture=gesture,
                 action={"kind": kind, "instruction": instruction})


class Conversation:
    def __init__(self, complete: Callable[[list[dict], dict], "Completion | str"] = functools.partial(complete_json, name="robot_reply"),
                 system: str = PERSONA):
        self.complete, self.system = complete, system
        self.history: list[dict] = []

    def respond(self, said: str) -> Reply:
        words = set(re.findall(r"[a-z-]+", said.lower()))
        if words & STOP_WORDS:
            return Reply(say="Stopping.", expression="neutral", stop=True, source="stop-word")
        if not said.strip():
            return Reply(say="Sorry, I did not catch that.", expression="thinking", source="fallback", error="empty input")
        messages = [{"role": "system", "content": self.system}, *self.history, {"role": "user", "content": said}]
        t = time.monotonic()
        try:
            answer = self.complete(messages, SCHEMA)
            raw, routing = (answer.text, answer.routing) if isinstance(answer, Completion) else (answer, {})
            reply = parse_reply(raw)
            reply.routing, reply.latency_s = routing, round(time.monotonic() - t, 2)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            return Reply(say="Sorry, I could not think of an answer just now.", expression="sad", source="fallback",
                         error=f"{type(e).__name__}: {e}", latency_s=round(time.monotonic() - t, 2))
        self.history += [{"role": "user", "content": said},
                         {"role": "assistant", "content": json.dumps({"say": reply.say, "expression": reply.expression,
                                                                      "gesture": reply.gesture, "action": reply.action})}]
        self.history = self.history[-MAX_HISTORY:]
        return reply


SMOKE = [
    "Hi, who are you?",
    "What can you do?",
    "Please walk forward a little and then turn left.",
    "Can you dance for me?",
]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("said", nargs="?")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    conv = Conversation()
    if not args.smoke:
        print(json.dumps(asdict(conv.respond(args.said or "")), indent=2))
        return
    rows = []
    for said in SMOKE:
        r = conv.respond(said)
        rows.append({"said": said, **asdict(r)})
        print(f"> {said}\n  [{r.expression}] {r.say}  action={r.action}  ({r.source}, {r.latency_s} s)", flush=True)
    Path("conversation-smoke.json").write_text(json.dumps(rows, indent=2))
    lines = ["# Conversation smoke test", "", f"`python -m humanoid_companion.conversation --smoke` on {time.strftime('%Y-%m-%d %H:%M')}, "
             "one conversation (history kept).", "",
             "| person | robot says | face | action | source, latency, model |", "|---|---|---|---|---|"]
    for r in rows:
        act = f"{r['action']['kind']}: {r['action']['instruction']}" if r["action"]["kind"] != "none" else "–"
        route = r["routing"].get("model") or "–"
        lines.append(f"| {r['said']} | {r['say']} | {r['expression']} | {act} | {r['source']}, {r['latency_s']} s, {route} |")
    Path("conversation-smoke.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
