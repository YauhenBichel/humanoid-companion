"""The one observation builder, shared by the simulator loop and the robot.

Mirrors mujoco_playground Op3 Joystick `_get_obs` (without training noise):
per frame  gyro (3) | IMU up-vector in world frame (3) | command (3) |
           joint positions - default pose (n) | action (n)   -> 9 + 2n = 49 for OP3
clipped to +-100, stacked `history` frames deep, newest frame first.

Playground builds the observation inside `step` *before* it records the action just applied,
so the action in the frame that follows action a_t is a_{t-1}. `record_action` keeps that
two-deep memory; tests/test_observation_parity.py pins it against Playground's env.step.
"""

from dataclasses import dataclass

import numpy as np

CLIP = 100.0


@dataclass(frozen=True)
class Sensors:
    gyro: np.ndarray       # rad/s, IMU frame
    upvector: np.ndarray   # IMU z-axis expressed in the world frame (R_world_imu[:, 2])
    joint_pos: np.ndarray  # rad, actuator order
    time: float


class ObservationBuilder:
    def __init__(self, default_pose: np.ndarray, history: int):
        self.default_pose = np.asarray(default_pose, np.float32)
        self.n = self.default_pose.size
        self.frame_size = 9 + 2 * self.n
        self.history = history
        self.reset()

    def reset(self) -> None:
        self._stack = np.zeros(self.history * self.frame_size, np.float32)
        self._action_in_obs = np.zeros(self.n, np.float32)   # what the next frame reports
        self._last_action = np.zeros(self.n, np.float32)     # the action applied most recently

    def frame(self, s: Sensors, command: np.ndarray) -> np.ndarray:
        f = np.concatenate([s.gyro, s.upvector, command, s.joint_pos - self.default_pose, self._action_in_obs])
        return np.clip(f.astype(np.float32), -CLIP, CLIP)

    def update(self, s: Sensors, command: np.ndarray) -> np.ndarray:
        """Push a new frame and return the full observation (a copy)."""
        f = self.frame(s, np.asarray(command, np.float32))
        self._stack = np.roll(self._stack, f.size)
        self._stack[: f.size] = f
        return self._stack.copy()

    def record_action(self, action: np.ndarray) -> None:
        """Call after applying `action`; the next frame will report the action before it."""
        self._action_in_obs = self._last_action
        self._last_action = np.asarray(action, np.float32).copy()
