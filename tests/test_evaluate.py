import numpy as np
import pytest

from humanoid_companion.evaluate import episode_metrics


def test_a_robot_counts_until_its_first_fall_and_not_after():
    # 4 steps, 2 episodes: episode 0 walks at 0.5 throughout; episode 1 walks at 0.3, falls at step 2
    vx = np.array([[0.5, 0.3], [0.5, 0.3], [0.5, 9.9], [0.5, 9.9]])
    done = np.array([[0, 0], [0, 0], [0, 1], [0, 0]])  # done can flip back after a fall; still counts as fallen
    m = episode_metrics(vx, done, command_vx=0.5)
    assert m["fall_rate"] == 0.5
    assert m["forward_speed_error"] == pytest.approx((0 * 4 + 0.2 * 2) / 6)
    assert m["mean_forward_speed"] == pytest.approx((0.5 * 4 + 0.3 * 2) / 6)
    assert m["mean_time_up_fraction"] == pytest.approx(6 / 8)


def test_nobody_up_does_not_divide_by_zero():
    m = episode_metrics(np.ones((3, 2)), np.ones((3, 2)), command_vx=0.5)
    assert m["fall_rate"] == 1.0 and m["forward_speed_error"] == 0.0
