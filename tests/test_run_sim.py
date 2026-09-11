"""The whole deployment stack in C MuJoCo, without a trained policy."""

import numpy as np
import pytest

from humanoid_companion.run_sim import simulate, training_model
from humanoid_companion.export import spec_from_env


class ZeroPolicy:
    """Action 0 = hold the training default pose: the PD controllers alone keep OP3 standing."""

    def __init__(self, spec):
        self.spec = spec

    def act(self, obs):
        assert obs.shape == (self.spec.obs_size,)
        return np.zeros(self.spec.action_size, np.float32)


class FlailPolicy(ZeroPolicy):
    """Full-scale alternating actions: should end in a safety stop, never a crash."""

    def __init__(self, spec):
        super().__init__(spec)
        self.i = 0

    def act(self, obs):
        self.i += 1
        return np.full(self.spec.action_size, 1.0 if (self.i // 10) % 2 else -1.0, np.float32)


@pytest.fixture(scope="module")
def spec():
    return spec_from_env("Op3Joystick")


def test_standing_still_for_two_seconds_stays_upright_and_in_place(spec):
    r = simulate(ZeroPolicy(spec), [0.0, 0.0, 0.0], seconds=2.0)
    assert r["safety_stops"] == [] and r["steps"] == 100 and r["seconds"] == 2.0
    assert abs(r["forward_distance_m"]) < 0.05 and abs(r["lateral_distance_m"]) < 0.05
    assert abs(r["yaw_change_rad"]) < 0.05
    assert r["final_height_m"] > 0.21  # training's fall threshold for the torso height
    assert r["trace"] and all(row["up_z"] > 0.85 for row in r["trace"])


def test_a_brain_plan_becomes_a_command_schedule():
    from humanoid_companion.brain import Command
    from humanoid_companion.run_sim import schedule

    command_at, total = schedule([Command(0.2, 0, 0, 3.0), Command(0, 0, 0.5, 2.0)])
    assert total == 5.0
    assert list(command_at(0.0)) == [0.2, 0, 0] and list(command_at(2.98)) == [0.2, 0, 0]
    assert list(command_at(3.0)) == pytest.approx([0, 0, 0.5]) and list(command_at(99)) == pytest.approx([0, 0, 0.5])


def test_a_time_varying_command_reaches_the_policy_and_frames_are_recorded(spec):
    from humanoid_companion.robot_io import MujocoRobotIO
    from humanoid_companion.run_sim import Recorder

    seen = []

    class Spy(ZeroPolicy):
        def act(self, obs):
            seen.append(obs[6:9].copy())  # the command slot of the newest frame
            return super().act(obs)

    io = MujocoRobotIO(training_model(), ctrl_dt=spec.ctrl_dt)
    rec = Recorder(io, every=5, height=64, width=64)
    r = simulate(Spy(spec), lambda t: np.array([0.3 if t < 0.5 else 0.0, 0.0, 0.0]), 1.0, io=io, recorder=rec)
    assert r["command"] is None and r["steps"] == 50
    assert seen[0][0] == pytest.approx(0.3) and seen[-1][0] == 0.0
    assert len(rec.frames) == 10 and rec.frames[0].shape == (64, 64, 3)


def test_a_flailing_policy_is_stopped_by_safety(spec):
    r = simulate(FlailPolicy(spec), [0.5, 0.0, 0.0], seconds=10.0)
    assert r["safety_stops"] and r["safety_stops"][0] in ("tilt", "joint_limit")
    assert r["steps"] < 500
