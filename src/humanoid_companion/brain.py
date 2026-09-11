"""Plain language in, safe walking commands out, answered by a language model you run yourself.

    python -m humanoid_companion.brain "walk forward slowly for three seconds, then turn left"
    python -m humanoid_companion.brain --smoke          # three instructions -> brain-smoke.md

The model is asked for a JSON object matching SCHEMA (a JSON-schema answer format, which small
local models follow more reliably than native tool calls). Its answer is never trusted: it is
parsed, validated and clamped to the command ranges the policy was trained on and to a speed cap.
Stop words never reach the model, and any failure (server down, malformed or non-finite answer)
becomes "stand still". The stop check is deliberately crude: any instruction containing a stop
word ("...then stop turning") means stand still.

Any OpenAI-compatible chat endpoint works: Ollama, llama.cpp's server, vLLM, LM Studio, a hosted
API. Environment: HUMANOID_LLM_BASE_URL (default http://127.0.0.1:11434/v1, Ollama),
HUMANOID_LLM_MODEL (the model name; Ollama needs one, e.g. llama3.1:8b), HUMANOID_LLM_API_KEY
(optional, sent as a bearer token).
"""

import argparse
import json
import math
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# Command ranges Op3Joystick is trained on (lin_vel_x, lin_vel_y, ang_vel_yaw).
TRAINED = {"vx": (-0.6, 1.5), "vy": (-0.8, 0.8), "yaw_rate": (-0.7, 0.7)}
DURATION = (0.5, 10.0)
MAX_COMMANDS = 10
STOP_WORDS = {"stop", "halt", "freeze", "estop", "e-stop"}

SYSTEM_PROMPT = (
    "You turn walking instructions for a small humanoid robot into a list of velocity commands, "
    "executed one after another. vx: forward speed in m/s (negative walks backwards). "
    "vy: sideways speed in m/s (positive = left). yaw_rate: turning speed in rad/s "
    "(positive = turn left). duration_s: how long to hold the command, in seconds. "
    "The robot walks between about 0.3 m/s ('slowly') and 0.5 m/s ('quickly'); use 0 to stand. "
    "A turn of 90 degrees at 0.5 rad/s takes about 3 seconds. Answer only with the JSON object."
)

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["commands"],
    "properties": {"commands": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["vx", "vy", "yaw_rate", "duration_s"],
        "properties": {k: {"type": "number"} for k in ("vx", "vy", "yaw_rate", "duration_s")},
    }}},
}


@dataclass(frozen=True)
class Command:
    vx: float
    vy: float
    yaw_rate: float
    duration_s: float
    stop: bool = False

    def as_array(self):
        return [self.vx, self.vy, self.yaw_rate]


STOP = Command(0.0, 0.0, 0.0, 1.0, stop=True)


@dataclass
class Plan:
    commands: list[Command]
    source: str                 # "model" | "stop-word" | "fallback"
    raw: str = ""
    error: str = ""
    clamped: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    routing: dict = field(default_factory=dict)  # which model answered (and the router's headers, if any)


@dataclass(frozen=True)
class Completion:
    text: str
    routing: dict


class PlanError(ValueError):
    pass


def _clip(name: str, x: float, lo: float, hi: float, notes: list[str]) -> float:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        raise PlanError(f"{name} is not a finite number: {x!r}")
    y = min(max(float(x), lo), hi)
    if y != x:
        notes.append(f"{name} {x} -> {y}")
    return y


def parse_plan(text: str, speed_cap: float = 0.5, min_speed: float = 0.0) -> tuple[list[Command], list[str]]:
    """Validate and clamp a model answer. Raises PlanError if it cannot be used at all.

    `min_speed`: the slowest forward/backward speed the policy actually walks at (measured by
    humanoid_companion.speed_sweep). A non-zero vx below it is raised to it: a slower command would make
    the robot stand still instead of walking slowly."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanError(f"not JSON: {e}") from None
    items = data.get("commands") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise PlanError("no commands")
    notes = []
    if len(items) > MAX_COMMANDS:
        notes.append(f"{len(items)} commands -> {MAX_COMMANDS}")
        items = items[:MAX_COMMANDS]
    out = []
    for i, c in enumerate(items):
        if not isinstance(c, dict):
            raise PlanError(f"command {i} is not an object")
        try:
            vx = _clip(f"[{i}].vx", c["vx"], max(TRAINED["vx"][0], -speed_cap), min(TRAINED["vx"][1], speed_cap), notes)
            vy = _clip(f"[{i}].vy", c["vy"], max(TRAINED["vy"][0], -speed_cap), min(TRAINED["vy"][1], speed_cap), notes)
            yaw = _clip(f"[{i}].yaw_rate", c["yaw_rate"], *TRAINED["yaw_rate"], notes)
            dur = _clip(f"[{i}].duration_s", c["duration_s"], *DURATION, notes)
        except KeyError as e:
            raise PlanError(f"command {i} misses {e}") from None
        if 0.0 < abs(vx) < min_speed:
            raised = math.copysign(min(min_speed, speed_cap), vx)
            notes.append(f"[{i}].vx {vx} -> {raised} (policy's slowest walk)")
            vx = raised
        out.append(Command(vx, vy, yaw, dur))
    return out, notes


def base_url() -> str:
    return os.environ.get("HUMANOID_LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")


def complete_json(messages: list[dict], schema: dict, timeout_s: float = 60.0, name: str = "walk_plan") -> Completion:
    """One chat completion with a JSON-schema answer format."""
    body = {
        "temperature": 0,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
    }
    if model := os.environ.get("HUMANOID_LLM_MODEL"):
        body["model"] = model
    headers = {"content-type": "application/json"}
    if key := os.environ.get("HUMANOID_LLM_API_KEY"):
        headers["authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{base_url()}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.load(r)
        # A router in front of several models (e.g. a local one and a hosted one) may say which it
        # chose in x-*-backend / x-*-rule headers; they are kept with the plan when present.
        routing = {"model": data.get("model", "")}
        for k, v in r.headers.items():
            if k.lower().startswith("x-") and k.lower().endswith(("-backend", "-rule")):
                routing[k.lower().rsplit("-", 1)[1]] = v
    return Completion(data["choices"][0]["message"]["content"], routing)


def plan(instruction: str, complete: Callable[[list[dict], dict], "Completion | str"] = complete_json,
         speed_cap: float = 0.5, min_speed: float = 0.0) -> Plan:
    words = set(re.findall(r"[a-z-]+", instruction.lower()))
    if not instruction.strip() or words & STOP_WORDS:
        return Plan([STOP], "stop-word")
    t = time.monotonic()
    raw, routing = "", {}
    try:
        answer = complete([{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": instruction}], SCHEMA)
        raw, routing = (answer.text, answer.routing) if isinstance(answer, Completion) else (answer, {})
        commands, notes = parse_plan(raw, speed_cap, min_speed)
        return Plan(commands, "model", raw=raw, clamped=notes, latency_s=round(time.monotonic() - t, 2), routing=routing)
    except (PlanError, OSError, KeyError, IndexError, TypeError, ValueError) as e:
        return Plan([STOP], "fallback", raw=raw, error=f"{type(e).__name__}: {e}",
                    latency_s=round(time.monotonic() - t, 2), routing=routing)


SMOKE = [
    "walk forward slowly for three seconds, then turn left",
    "take a few steps backwards",
    "turn right on the spot for two seconds, then walk forward quickly",
]


def smoke_report(plans: list[tuple[str, Plan]]) -> str:
    lines = ["# Brain smoke test", "", f"Model server `{base_url()}` "
             f"(model: {os.environ.get('HUMANOID_LLM_MODEL') or 'server default'}), speed cap 0.5 m/s. "
             f"Generated by `python -m humanoid_companion.brain --smoke` on {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    for instruction, p in plans:
        r = p.routing
        route = f", routed: backend {r.get('backend') or '?'}, rule {r.get('rule') or 'default'}, model {r.get('model') or '?'}" if r else ""
        lines += [f"## {instruction}", "", f"source: {p.source}, latency {p.latency_s} s{route}"
                  + (f", clamped: {'; '.join(p.clamped)}" if p.clamped else "")
                  + (f", error: {p.error}" if p.error else ""), "",
                  "| vx | vy | yaw_rate | duration_s |", "|---|---|---|---|",
                  *[f"| {c.vx} | {c.vy} | {c.yaw_rate} | {c.duration_s} |" for c in p.commands], ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("instruction", nargs="?")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--speed-cap", type=float, default=0.5)
    args = p.parse_args(argv)
    if args.smoke:
        plans = [(i, plan(i, speed_cap=args.speed_cap)) for i in SMOKE]
        out = Path("brain-smoke.md")
        out.write_text(smoke_report(plans))
        print(out.read_text())
    elif args.instruction is not None:
        r = plan(args.instruction, speed_cap=args.speed_cap)
        print(json.dumps({**asdict(r), "commands": [asdict(c) for c in r.commands]}, indent=2))
    else:
        p.error("give an instruction or --smoke")


if __name__ == "__main__":
    main()
