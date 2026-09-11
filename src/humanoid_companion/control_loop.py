"""The 50 Hz loop that will run on the robot: sensors -> observation -> policy -> joint targets.

Every period: read sensors, check safety, build the observation, run the policy, turn the action
into joint targets exactly as in training (default pose + action_scale * action, clipped to the
actuator range), write them. A safety stop calls io.stop() and ends the loop; it never writes
another target.
"""

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from humanoid_companion.observation import ObservationBuilder, Sensors
from humanoid_companion.policy import NumpyPolicy
from humanoid_companion.robot_io import RobotIO

Command = np.ndarray  # [vx m/s, vy m/s, yaw rate rad/s]


@dataclass(frozen=True)
class SafetyLimits:
    min_up_z: float = 0.85         # same as training's fall termination (up-vector z)
    joint_margin: float = 0.05     # rad beyond the actuator range before stopping
    max_missed_deadlines: int = 5  # consecutive overruns of ctrl_dt (real time only)


@dataclass
class RunResult:
    steps: int = 0
    stop_reason: str | None = None       # None: ran to the end
    log: list[dict] = field(default_factory=list)


class ControlLoop:
    def __init__(self, io: RobotIO, policy: NumpyPolicy, limits: SafetyLimits = SafetyLimits(),
                 estop: Callable[[], bool] = lambda: False):
        self.io, self.policy, self.limits, self.estop = io, policy, limits, estop
        s = policy.spec
        self.builder = ObservationBuilder(s.default_pose, s.obs_history)
        self.lower, self.upper = s.ctrl_lower, s.ctrl_upper

    def safety_reason(self, s: Sensors) -> str | None:
        if self.estop():
            return "estop"
        if s.upvector[2] < self.limits.min_up_z:
            return "tilt"
        m = self.limits.joint_margin
        if np.any(s.joint_pos < self.lower - m) or np.any(s.joint_pos > self.upper + m):
            return "joint_limit"
        return None

    def targets(self, action: np.ndarray) -> np.ndarray:
        s = self.policy.spec
        return np.clip(s.default_pose + s.action_scale * action, self.lower, self.upper)

    def step(self, command: Command, gesture: dict[int, float] | None = None) -> tuple[str | None, Sensors]:
        """One control period. Returns (stop reason or None, the sensors it acted on).

        `gesture`: {actuator index: offset from the default pose} for head and arm joints; those
        targets replace the policy's (humanoid_companion.gestures). The observation still records the
        policy's own action, so what the policy sees keeps its training meaning."""
        s = self.io.read()
        reason = self.safety_reason(s)
        if reason:
            self.io.stop()
            return reason, s
        action = self.policy.act(self.builder.update(s, command))
        targets = self.targets(action)
        if gesture:
            for i, offset in gesture.items():
                targets[i] = self.policy.spec.default_pose[i] + offset
            targets = np.clip(targets, self.lower, self.upper)
        self.io.write(targets)
        self.builder.record_action(action)
        return None, s

    def run(self, command_at: Callable[[float], Command], seconds: float, realtime: bool = False,
            clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
            on_step: Callable[[int, Sensors], dict | None] | None = None,
            gesture_at: Callable[[float], dict[int, float]] | None = None) -> RunResult:
        dt = self.policy.spec.ctrl_dt
        result, missed = RunResult(), 0
        n_steps = int(round(seconds / dt))
        next_tick = clock()
        for i in range(n_steps):
            reason, s = self.step(np.asarray(command_at(i * dt), np.float32), gesture_at(i * dt) if gesture_at else None)
            if on_step and (row := on_step(i, s)):
                result.log.append(row)
            if reason:
                result.stop_reason = reason
                break
            result.steps = i + 1
            if realtime:
                next_tick += dt
                late = clock() - next_tick
                missed = missed + 1 if late > 0 else 0
                if missed >= self.limits.max_missed_deadlines:
                    self.io.stop()
                    result.stop_reason = "deadline"
                    break
                if late < 0:
                    sleep(-late)
        return result
