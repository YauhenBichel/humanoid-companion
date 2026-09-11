"""Every stop condition fires, and the loop never writes a target outside the actuator range."""

import numpy as np

from humanoid_companion.control_loop import ControlLoop, SafetyLimits
from humanoid_companion.observation import Sensors
from humanoid_companion.policy import PolicySpec

N = 4
UPRIGHT = np.array([0.0, 0.0, 1.0])


class FakeIO:
    def __init__(self, sensors):
        self.sensors = list(sensors)
        self.writes, self.stops = [], 0

    def read(self):
        return self.sensors.pop(0) if len(self.sensors) > 1 else self.sensors[0]

    def write(self, targets):
        self.writes.append(np.array(targets))

    def stop(self):
        self.stops += 1


class FakePolicy:
    def __init__(self, action, default=None):
        self.action = np.asarray(action, np.float32)
        self.spec = PolicySpec(obs_size=3 * (9 + 2 * N), action_size=N, action_scale=0.3,
                               default_pose=np.zeros(N, np.float32) if default is None else np.asarray(default, np.float32),
                               ctrl_lower=np.full(N, -1.0, np.float32), ctrl_upper=np.full(N, 1.0, np.float32),
                               ctrl_dt=0.02, actuator_names=tuple("abcd"), obs_history=3)

    def act(self, obs):
        assert obs.shape == (self.spec.obs_size,)
        return self.action


def sensors(up=UPRIGHT, joints=np.zeros(N)):
    return Sensors(gyro=np.zeros(3), upvector=np.asarray(up, float), joint_pos=np.asarray(joints, float), time=0.0)


def run(io, policy, seconds=0.2, **kw):
    loop = ControlLoop(io, policy, **{k: v for k, v in kw.items() if k in ("limits", "estop")})
    return loop.run(lambda t: np.array([0.5, 0.0, 0.0]), seconds,
                    **{k: v for k, v in kw.items() if k in ("realtime", "clock", "sleep")})


def test_upright_run_goes_to_the_end_and_writes_every_period():
    io = FakeIO([sensors()])
    r = run(io, FakePolicy(np.zeros(N)))
    assert (r.stop_reason, r.steps, len(io.writes), io.stops) == (None, 10, 10, 0)


def test_tilt_stops_before_writing():
    io = FakeIO([sensors(), sensors(), sensors(up=[0.6, 0.0, 0.8])])
    r = run(io, FakePolicy(np.zeros(N)))
    assert (r.stop_reason, r.steps, len(io.writes), io.stops) == ("tilt", 2, 2, 1)


def test_joint_beyond_limit_plus_margin_stops():
    io = FakeIO([sensors(joints=[0, 0, 1.04, 0]), sensors(joints=[0, 0, 1.06, 0])])
    r = run(io, FakePolicy(np.zeros(N)))
    assert (r.stop_reason, r.steps) == ("joint_limit", 1)


def test_estop_stops_immediately():
    io = FakeIO([sensors()])
    r = run(io, FakePolicy(np.zeros(N)), estop=lambda: True)
    assert (r.stop_reason, r.steps, io.writes, io.stops) == ("estop", 0, [], 1)


def test_targets_are_clipped_to_the_actuator_range():
    io = FakeIO([sensors()])
    run(io, FakePolicy(np.array([1.0, -1.0, 1.0, -1.0]), default=[0.9, -0.9, 0.0, 0.0]))
    w = np.array(io.writes)
    assert w.max() <= 1.0 and w.min() >= -1.0
    np.testing.assert_allclose(w[0], [1.0, -1.0, 0.3, -0.3])


def test_missed_deadlines_stop_a_realtime_run():
    now = [0.0]

    def clock():
        now[0] += 0.03  # every period takes longer than ctrl_dt = 0.02
        return now[0]

    io = FakeIO([sensors()])
    r = run(io, FakePolicy(np.zeros(N)), seconds=1.0, realtime=True, clock=clock, sleep=lambda s: None,
            limits=SafetyLimits(max_missed_deadlines=5))
    assert (r.stop_reason, io.stops) == ("deadline", 1)
    assert r.steps == 5
