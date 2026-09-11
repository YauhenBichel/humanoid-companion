"""Drive an exported policy through the deployment control loop in plain (CPU) MuJoCo.

    python -m humanoid_companion.run_sim --vx 0.5 --seconds 10
    bin/viewer --vx 0.4                                         # macOS viewer (mjpython + uv's libpython)

The robot model is Playground's OP3 model with the training PD gains and timestep; the physics
engine is MuJoCo C, not MJX, so this is also a sim-to-sim test. Writes sim-in-the-loop.json.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from humanoid_companion.control_loop import ControlLoop, SafetyLimits
from humanoid_companion.policy import NumpyPolicy
from humanoid_companion.robot_io import MujocoRobotIO


def training_model(env_name: str = "Op3Joystick"):
    """The MuJoCo model exactly as the policy was trained on (nominal gains, damping, timestep)."""
    from humanoid_companion.envs import load_env

    return load_env(env_name).mj_model


def schedule(commands) -> tuple[Callable[[float], np.ndarray], float]:
    """Brain commands (each held for its duration) -> command_at(t) and the total time."""
    ends = np.cumsum([c.duration_s for c in commands])

    def command_at(t: float) -> np.ndarray:
        i = min(int(np.searchsorted(ends, t, side="right")), len(commands) - 1)
        return np.asarray(commands[i].as_array(), np.float32)

    return command_at, float(ends[-1])


class Recorder:
    """Offscreen frames from a camera that follows the torso, written with mediapy."""

    def __init__(self, io: MujocoRobotIO, every: int = 2, height: int = 480, width: int = 640):
        import mujoco

        self.io, self.every, self.frames = io, every, []
        self.renderer = mujoco.Renderer(io.model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = io.model.body("body_link").id
        self.camera.distance, self.camera.elevation, self.camera.azimuth = 1.4, -15.0, 135.0

    def capture(self, i: int) -> None:
        if i % self.every == 0:
            self.renderer.update_scene(self.io.data, camera=self.camera)
            self.frames.append(self.renderer.render())

    def save(self, path: Path, ctrl_dt: float) -> None:
        import mediapy

        path.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(path, self.frames, fps=int(round(1.0 / (ctrl_dt * self.every))))


def simulate(policy: NumpyPolicy, command, seconds: float, io: MujocoRobotIO | None = None, viewer=None,
             realtime: bool = False, limits: SafetyLimits = SafetyLimits(), recorder: Recorder | None = None,
             gesture_at=None) -> dict:
    """`command`: a fixed [vx, vy, yaw_rate] or a function of simulated time. `gesture_at`: t -> head/arm
    target offsets (humanoid_companion.gestures.overlay)."""
    io = io or MujocoRobotIO(training_model(), ctrl_dt=policy.spec.ctrl_dt)
    command_at = command if callable(command) else (lambda t, c=np.asarray(command, np.float32): c)
    start = io.base_position
    heading = io.data.xmat[io.model.body("body_link").id].reshape(3, 3)[:, 0].copy()
    heading[2] = 0.0
    heading /= np.linalg.norm(heading)

    def on_step(i, s):
        if viewer is not None:
            viewer.sync()
        if recorder is not None:
            recorder.capture(i)
        if i % 25 == 0:
            p = io.base_position
            return {"t": round(s.time, 3), "x": round(float(p[0]), 4), "y": round(float(p[1]), 4),
                    "z": round(float(p[2]), 4), "up_z": round(float(s.upvector[2]), 4)}
        return None

    loop = ControlLoop(io, policy, limits=limits)
    result = loop.run(command_at, seconds, realtime=realtime, on_step=on_step, gesture_at=gesture_at)
    end = io.base_position
    x_end = io.data.xmat[io.model.body("body_link").id].reshape(3, 3)[:, 0]
    yaw_change = float(np.arctan2(np.cross(heading, x_end)[2], np.dot(heading[:2], x_end[:2])))
    return {
        "command": None if callable(command) else [float(c) for c in command],
        "seconds": round(result.steps * policy.spec.ctrl_dt, 3),
        "requested_seconds": seconds,
        "steps": result.steps,
        "forward_distance_m": round(float(np.dot(end - start, heading)), 4),
        "lateral_distance_m": round(float(np.cross(heading, end - start)[2]), 4),
        "yaw_change_rad": round(yaw_change, 4),  # positive = turned left
        "final_height_m": round(float(end[2]), 4),
        "safety_stops": [result.stop_reason] if result.stop_reason else [],
        "trace": result.log,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", type=Path, default=Path(__file__).parent / "policies" / "op3_walk.npz",
                   help="default: the bundled OP3 walking policy")
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--viewer", action="store_true", help="MuJoCo passive viewer, real time (mjpython on macOS)")
    p.add_argument("--instruction", help="plain language, planned by the brain (your LLM, humanoid_companion.brain)")
    p.add_argument("--speed-cap", type=float, default=0.5, help="brain speed cap, m/s")
    p.add_argument("--speed-sweep", type=Path,
                   help="a speed-sweep-<name>.json for this policy: the brain raises slower commands to its slowest usable walk")
    p.add_argument("--record", type=Path, help="write an mp4 of the run")
    p.add_argument("--report", type=Path, help="default: sim-in-the-loop.json, or demo.json with --instruction")
    args = p.parse_args(argv)

    policy = NumpyPolicy.load(args.policy)
    io = MujocoRobotIO(training_model(), ctrl_dt=policy.spec.ctrl_dt)
    extra = {}
    if args.instruction:
        from dataclasses import asdict

        from humanoid_companion.brain import plan

        usable = json.loads(args.speed_sweep.read_text())["usable_range"] if args.speed_sweep else None
        min_speed = usable[0] if usable else 0.0
        brain_plan = plan(args.instruction, speed_cap=args.speed_cap, min_speed=min_speed)
        command, seconds = schedule(brain_plan.commands)
        extra = {"instruction": args.instruction, "brain_min_speed": min_speed,
                 "brain_source": brain_plan.source, "brain_error": brain_plan.error,
                 "brain_routing": brain_plan.routing, "brain_clamped": brain_plan.clamped,
                 "brain_latency_s": brain_plan.latency_s, "commands": [asdict(c) for c in brain_plan.commands]}
    else:
        command, seconds = [args.vx, args.vy, args.yaw], args.seconds
    recorder = Recorder(io) if args.record else None
    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(io.model, io.data) as v:
            report = simulate(policy, command, seconds, io=io, viewer=v, realtime=True, recorder=recorder)
            time.sleep(1.0)
    else:
        report = simulate(policy, command, seconds, io=io, recorder=recorder)
    if recorder:
        recorder.save(args.record, policy.spec.ctrl_dt)
        report["video"] = str(args.record)
    report.update(extra, policy=str(args.policy))
    args.report = args.report or Path("demo.json" if args.instruction else "sim-in-the-loop.json")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
