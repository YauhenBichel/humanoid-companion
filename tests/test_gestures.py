from pathlib import Path

import numpy as np
import pytest

from humanoid_companion.gestures import GESTURES, HEAD_ARMS, ease, envelope, overlay, talking_head

NAMES = ("head_pan_act", "head_tilt_act", "l_sho_pitch_act", "l_sho_roll_act", "l_el_act", "r_sho_pitch_act",
         "r_sho_roll_act", "r_el_act", "l_hip_yaw_act")
POLICY = Path(__file__).parents[1] / "src/humanoid_companion/policies/op3_walk.npz"


def test_ease_starts_and_ends_at_rest():
    assert ease(0, 2) == 0 and ease(2, 2) == 0 and ease(1, 2) == 1
    assert 0 < ease(0.1, 2) < 1 and 0 < ease(1.9, 2) < 1


@pytest.mark.parametrize("name", list(GESTURES))
def test_every_gesture_moves_only_head_and_arms_and_returns_to_rest(name):
    g = GESTURES[name]
    assert set(g.offsets(g.duration / 2)) <= set(HEAD_ARMS)
    assert all(abs(v) < 1e-9 for v in g.offsets(g.duration).values())
    assert all(abs(v) < 1e-9 for v in g.offsets(0.0).values())


def test_overlay_refuses_leg_joints_and_sums_parts():
    with pytest.raises(ValueError, match="head and arm"):
        overlay(NAMES, lambda t: {"l_hip_yaw_act": 0.1})
    at = overlay(NAMES, lambda t: {"head_tilt_act": 0.1}, lambda t: {"head_tilt_act": 0.2, "r_el_act": 0.3})
    assert at(0.0) == pytest.approx({1: 0.3, 7: 0.3})


def test_talking_head_is_still_when_the_voice_is_silent():
    rate = 24000
    loud = envelope(np.concatenate([np.zeros(rate), 0.3 * np.ones(rate)]).astype(np.float32), rate)
    head = talking_head(loud)
    assert all(v == 0 for v in head(0.5).values())
    assert loud(1.5) > 0.5 and any(v != 0 for v in head(1.33).values())


@pytest.mark.skipif(not POLICY.exists(), reason="needs the bundled walking policy")
@pytest.mark.parametrize("name", list(GESTURES))
def test_every_gesture_keeps_the_robot_upright(name):
    from humanoid_companion.policy import NumpyPolicy
    from humanoid_companion.robot_io import MujocoRobotIO
    from humanoid_companion.run_sim import simulate, training_model

    pol, g = NumpyPolicy.load(POLICY), GESTURES[name]
    cases = [0.0] + ([0.4] if g.walk_ok else [])
    for vx in cases:
        io = MujocoRobotIO(training_model(), ctrl_dt=pol.spec.ctrl_dt)
        cmd = (lambda t: np.array(g.command(t), np.float32)) if g.command else [vx, 0.0, 0.0]
        r = simulate(pol, cmd, g.duration + 1.0, io=io, gesture_at=overlay(pol.spec.actuator_names, g.offsets))
        assert r["safety_stops"] == [], (name, vx)
        assert min(t["up_z"] for t in r["trace"]) > 0.95, (name, vx)
