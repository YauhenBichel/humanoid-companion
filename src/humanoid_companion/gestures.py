"""Body language for the OP3: head and arm gestures layered over the walking policy.

A gesture is a function of time returning target *offsets from the default pose* for head and
arm joints only (actuator names). The control loop replaces the policy's targets for those joints
while a gesture runs; the legs stay with the policy, which keeps the robot balanced. Every gesture
eases in and out (no jumps) and is tested for stability in simulation (tests/test_gestures.py):
a gesture that made the robot fall would not be allowed here.

Joint conventions (checked by rendering each joint): right shoulder
pitch + raises the right arm, right shoulder roll + swings it outward, elbow + bends it; the left
arm mirrors the right (signs flip); head_tilt + nods down, head_pan + turns the head.
Dance also moves the whole body: it returns a command pattern (turning left-right) for the legs.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

HEAD_ARMS = ("head_pan_act", "head_tilt_act", "l_sho_pitch_act", "l_sho_roll_act", "l_el_act",
             "r_sho_pitch_act", "r_sho_roll_act", "r_el_act")


def ease(t: float, duration: float, ramp: float = 0.35) -> float:
    """0 -> 1 -> 0 envelope with smooth (cosine) ramps at both ends."""
    if t <= 0 or t >= duration:
        return 0.0
    r = min(ramp, duration / 2)
    if t < r:
        return 0.5 - 0.5 * math.cos(math.pi * t / r)
    if t > duration - r:
        return 0.5 - 0.5 * math.cos(math.pi * (duration - t) / r)
    return 1.0


@dataclass(frozen=True)
class Gesture:
    name: str
    duration: float
    offsets: Callable[[float], dict[str, float]]                   # t -> {actuator: offset rad}
    command: Callable[[float], tuple[float, float, float]] | None = None  # optional leg command pattern
    walk_ok: bool = False   # safe while walking (measured: docs/gesture-stability.md)


def _wave(t):
    # The left arm counter-balances: without it, one raised arm shifts the weight sideways and the
    # robot tips over even standing (every variant without it fell; 11 September measurements).
    e = ease(t, 2.6)
    return {"r_sho_pitch_act": 2.0 * e, "r_sho_roll_act": 0.35 * e,
            "r_el_act": e * (0.5 + 0.45 * math.sin(2 * math.pi * 2.0 * t)), "head_tilt_act": -0.12 * e,
            "l_sho_roll_act": 0.4 * e}


def _nod(t):
    e = ease(t, 1.4, 0.2)
    return {"head_tilt_act": e * (0.12 + 0.18 * math.sin(2 * math.pi * 1.6 * t))}


def _celebrate(t):
    e = ease(t, 2.2)
    shake = 0.25 * math.sin(2 * math.pi * 2.5 * t)
    return {"r_sho_pitch_act": 2.3 * e, "l_sho_pitch_act": -2.3 * e,
            "r_el_act": e * (0.3 + shake), "l_el_act": -e * (0.3 + shake), "head_tilt_act": -0.2 * e}


def _look_around(t):
    e = ease(t, 2.4)
    return {"head_pan_act": 0.6 * e * math.sin(2 * math.pi * t / 2.4), "head_tilt_act": -0.08 * e}


def _dance_arms(t):
    e = ease(t, 3.2)
    swing = math.sin(2 * math.pi * t / 1.6)
    return {"r_sho_pitch_act": e * (1.2 + 0.5 * swing), "l_sho_pitch_act": -e * (1.2 - 0.5 * swing),
            "head_pan_act": 0.25 * e * swing}


def _dance_legs(t):
    # turn left, right, left, right on the spot (0.8 s each): trained yaw range, above the dead zone
    return (0.0, 0.0, 0.6 if int(t / 0.8) % 2 == 0 else -0.6) if 0 <= t < 3.2 else (0.0, 0.0, 0.0)


GESTURES = {
    "wave": Gesture("wave", 2.6, _wave),
    "nod": Gesture("nod", 1.4, _nod, walk_ok=True),
    "celebrate": Gesture("celebrate", 2.2, _celebrate),          # both arms up: falls if walking
    "look_around": Gesture("look_around", 2.4, _look_around, walk_ok=True),
    "dance": Gesture("dance", 3.2, _dance_arms, _dance_legs),
}


def talking_head(loudness: Callable[[float], float]) -> Callable[[float], dict[str, float]]:
    """Small head bobs while speaking, following the voice's loudness (0..1)."""
    return lambda t: {"head_tilt_act": 0.10 * loudness(t) * math.sin(2 * math.pi * 1.7 * t),
                      "head_pan_act": 0.06 * loudness(t) * math.sin(2 * math.pi * 0.6 * t)}


def overlay(actuator_names: tuple[str, ...], *parts: Callable[[float], dict[str, float]]) -> Callable[[float], dict[int, float]]:
    """Combine gesture parts (summed) into {actuator index: offset}; only head and arm joints."""
    index = {n: i for i, n in enumerate(actuator_names)}
    bad = [n for p in parts for n in p(0.5) if n not in HEAD_ARMS]
    if bad:
        raise ValueError(f"gestures may only move head and arm joints, not {bad}")

    def at(t: float) -> dict[int, float]:
        out: dict[int, float] = {}
        for p in parts:
            for name, v in p(t).items():
                out[index[name]] = out.get(index[name], 0.0) + v
        return out

    return at


def envelope(samples: np.ndarray, rate: int, window_s: float = 0.05) -> Callable[[float], float]:
    """Loudness 0..1 of a speech signal at time t (RMS x 6, clipped), for talking_head."""
    hop = max(1, int(rate * window_s))
    rms = np.array([math.sqrt(float(np.mean(samples[i:i + hop] ** 2))) for i in range(0, len(samples), hop)] or [0.0])
    levels = np.clip(rms * 6, 0, 1)
    return lambda t: float(levels[int(t / window_s)]) if 0 <= t < len(levels) * window_s else 0.0


def check(policy_path: str, out: str = "gesture-stability.md") -> list[dict]:
    """Every gesture standing (twice: from rest and after 1 s), walk_ok ones also at 0.4 m/s:
    safety stops, worst tilt, drift. Writes the report the gesture flags are based on."""
    from pathlib import Path

    from humanoid_companion.policy import NumpyPolicy
    from humanoid_companion.robot_io import MujocoRobotIO
    from humanoid_companion.run_sim import simulate, training_model

    pol, model = NumpyPolicy.load(policy_path), training_model()
    rows = []
    for name, g in GESTURES.items():
        cases = [("standing, from rest", 0.0, 0.0), ("standing, after 1 s", 0.0, 1.0)]
        if g.walk_ok:
            cases.append(("walking 0.4 m/s", 0.4, 0.0))
        for label, vx, settle in cases:
            io = MujocoRobotIO(model, ctrl_dt=pol.spec.ctrl_dt)
            if settle:
                simulate(pol, [0, 0, 0], settle, io=io)
            cmd = (lambda t, g=g: np.array(g.command(t), np.float32)) if g.command else [vx, 0.0, 0.0]
            r = simulate(pol, cmd, g.duration + 1.0, io=io, gesture_at=overlay(pol.spec.actuator_names, g.offsets))
            rows.append({"gesture": name, "case": label, "safety_stops": r["safety_stops"],
                         "min_up": round(min(t["up_z"] for t in r["trace"]), 3),
                         "lateral_m": r["lateral_distance_m"], "forward_m": r["forward_distance_m"]})
    lines = ["# Gesture stability", "", f"`python -m humanoid_companion.gestures --check --policy {policy_path}`: each gesture layered "
             "over the walking policy in C MuJoCo. A gesture is allowed only if it never triggers a safety stop; "
             "`walk_ok` gestures are also run while walking.", "",
             "| gesture | case | safety stop | worst tilt (up-vector z) | sideways m | forward m |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['gesture']} | {r['case']} | {', '.join(r['safety_stops']) or '-'} | {r['min_up']} | {r['lateral_m']} | {r['forward_m']} |" for r in rows]
    lines += ["", "Found while tuning (11 September): a raised right arm tips the robot over even standing unless the left "
              "arm counter-balances; both arms up (celebrate) is stable standing but tips it over while walking.", ""]
    Path(out).write_text("\n".join(lines))
    return rows


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Gesture stability check")
    p.add_argument("--check", action="store_true", required=True)
    p.add_argument("--policy", default=str(Path(__file__).parent / "policies" / "op3_walk.npz"))
    a = p.parse_args()
    for r in check(a.policy):
        print(r)
