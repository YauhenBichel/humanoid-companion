"""Our observation builder and simulated IO == Playground's Op3Joystick, step by step."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from humanoid_companion.observation import ObservationBuilder, Sensors
from humanoid_companion.robot_io import MujocoRobotIO

STEPS = 8


@pytest.fixture(scope="module")
def rollout():
    from mujoco_playground import registry

    cfg = registry.get_default_config("Op3Joystick")
    cfg.impl = "jax"
    cfg.obs_noise = 0.0  # Playground adds obs_noise * U(-1, 1); zero noise leaves only the clip
    env = registry.load("Op3Joystick", config=cfg)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    actions = np.random.default_rng(1).uniform(-0.5, 0.5, size=(STEPS, env.action_size)).astype(np.float32)

    states = [reset(jax.random.PRNGKey(3))]
    for a in actions:
        states.append(step(states[-1], jp.asarray(a)))
    return env, states, actions


def env_sensors(env, data) -> Sensors:
    return Sensors(gyro=np.asarray(env.get_gyro(data)), upvector=np.asarray(env.get_gravity(data)),
                   joint_pos=np.asarray(data.qpos[7:]), time=float(data.time))


def test_observations_match_env_including_history_and_action_memory(rollout):
    env, states, actions = rollout
    builder = ObservationBuilder(np.asarray(env._default_pose), history=env._config.obs_history_size)
    command = np.asarray(states[0].info["command"])

    for t, state in enumerate(states):
        if t > 0:
            builder.record_action(actions[t - 1])
        ours = builder.update(env_sensors(env, state.data), command)
        np.testing.assert_allclose(ours, np.asarray(state.obs), atol=1e-6, err_msg=f"step {t}")


def test_the_action_in_a_frame_is_the_one_before_the_last_applied(rollout):
    env, states, actions = rollout
    n = env.action_size
    obs = np.asarray(states[3].obs)  # after applying actions[0..2]
    np.testing.assert_allclose(obs[49 - n:49], actions[1], atol=1e-6)


def test_mujoco_io_reads_the_same_sensors_as_the_env_for_the_same_state(rollout):
    """Sensor addressing: C MuJoCo and MJX evaluate the sensors at an identical state."""
    from mujoco import mjx

    env, states, _ = rollout
    io = MujocoRobotIO(env.mj_model, ctrl_dt=env._config.ctrl_dt)
    assert io.n_substeps == 5
    fwd = jax.jit(lambda d: mjx.forward(env.mjx_model, d))
    for state in states[1:]:
        io.data.qpos[:] = np.asarray(state.data.qpos)
        io.data.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(io.model, io.data)
        ours, theirs = io.read(), env_sensors(env, fwd(state.data))
        np.testing.assert_allclose(ours.gyro, theirs.gyro, atol=1e-4)
        np.testing.assert_allclose(ours.upvector, theirs.upvector, atol=1e-5)
        np.testing.assert_allclose(ours.joint_pos, theirs.joint_pos, atol=1e-6)


def test_one_control_period_in_c_mujoco_matches_one_env_step(rollout):
    """Timing: after a step, sensors lag the state by one substep in both engines (MuJoCo
    computes sensors before integrating). Same start, same targets -> close readings.

    What remains is the C-vs-MJX engine gap (contact handling at the feet), measured on
    11 September over these 7 periods: joints 1.8e-3 rad, up-vector 8.2e-4, gyro 5.3e-2 rad/s.
    A one-substep timing error gives gyro differences of 0.06-0.12 on the first period alone.
    Tolerances are the measurements with ~2x margin."""
    env, states, actions = rollout
    io = MujocoRobotIO(env.mj_model, ctrl_dt=env._config.ctrl_dt)
    for before, after in zip(states[1:-1], states[2:]):
        io.data.qpos[:] = np.asarray(before.data.qpos)
        io.data.qvel[:] = np.asarray(before.data.qvel)
        mujoco.mj_forward(io.model, io.data)
        io.write(np.asarray(after.info["motor_targets"]))
        ours, theirs = io.read(), env_sensors(env, after.data)
        np.testing.assert_allclose(ours.joint_pos, theirs.joint_pos, atol=4e-3)
        np.testing.assert_allclose(ours.upvector, theirs.upvector, atol=2e-3)
        np.testing.assert_allclose(ours.gyro, theirs.gyro, atol=0.1)


def test_mujoco_io_starts_in_the_training_pose(rollout):
    env, states, _ = rollout
    io = MujocoRobotIO(env.mj_model, ctrl_dt=env._config.ctrl_dt)
    np.testing.assert_allclose(io.read().joint_pos, np.asarray(env._default_pose), atol=1e-6)
    np.testing.assert_allclose(io.data.qpos, np.asarray(states[0].data.qpos), atol=1e-6)
