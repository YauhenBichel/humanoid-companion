import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_companion.envs import RANGES, default_config, load_env, make_randomizer

N = 64


@pytest.fixture(scope="module")
def env():
    return load_env("Op3JoystickHardened")


@pytest.fixture(scope="module")
def randomized(env):
    model, in_axes = make_randomizer(env.mj_model)(env.mjx_model, jax.random.split(jax.random.PRNGKey(0), N))
    return env.mjx_model, model, in_axes


def within(x, lo, hi):
    return bool(np.all(x >= lo - 1e-6) and np.all(x <= hi + 1e-6))


def test_parameters_vary_across_envs_and_stay_in_range(env, randomized):
    base, m, _ = randomized
    floor, torso = env.mj_model.geom("floor").id, env.mj_model.body("body_link").id
    f = np.asarray(m.geom_friction[:, floor, 0])
    assert f.std() > 0.05 and within(f, *RANGES["floor_friction"])

    kp = np.asarray(m.actuator_gainprm[:, :, 0]) / np.asarray(base.actuator_gainprm[:, 0])
    assert kp.std() > 0.05 and within(kp, *RANGES["kp_scale"])
    kd = np.asarray(m.dof_damping[:, 6:]) / np.asarray(base.dof_damping[6:])
    assert kd.std() > 0.05 and within(kd, *RANGES["kd_scale"])

    mass = np.asarray(m.body_mass)
    others = np.delete(np.arange(env.mj_model.nbody), [0, torso])  # body 0 is the world, mass 0
    assert within(mass[:, others] / np.asarray(base.body_mass)[others], *RANGES["mass_scale"])
    # torso: scaled like every body, then a mass added
    t0 = float(np.asarray(base.body_mass)[torso])
    lo = t0 * RANGES["mass_scale"][0] + RANGES["torso_mass_add"][0]
    hi = t0 * RANGES["mass_scale"][1] + RANGES["torso_mass_add"][1]
    assert within(mass[:, torso], lo, hi) and mass[:, torso].std() > 0.05


def test_position_servo_stays_consistent_bias_is_minus_gain(randomized):
    _, m, _ = randomized
    np.testing.assert_allclose(np.asarray(m.actuator_biasprm[:, :, 1]), -np.asarray(m.actuator_gainprm[:, :, 0]))


def test_only_the_randomized_fields_are_batched(randomized):
    base, m, in_axes = randomized
    # Static int fields (na, neq, ...) keep their values in in_axes; only array fields can be batched.
    batched = {k for k, v in vars(in_axes).items()
               if isinstance(v, int) and v == 0 and hasattr(getattr(base, k), "shape")}
    assert batched == {"geom_friction", "dof_frictionloss", "dof_armature", "body_mass",
                       "actuator_gainprm", "actuator_biasprm", "dof_damping"}
    np.testing.assert_array_equal(np.asarray(m.geom_friction[:, 1:]),
                                  np.broadcast_to(np.asarray(base.geom_friction[1:]), m.geom_friction[:, 1:].shape))


def _hardened(**overrides):
    cfg = default_config("Op3JoystickHardened")
    cfg.obs_noise = 0.0
    for k, v in overrides.items():
        cfg[k] = v
    return load_env("Op3JoystickHardened", cfg)


def test_with_additions_off_the_hardened_step_is_playgrounds_step():
    """Guards the copied step: same state, same actions -> same physics, obs and reward."""
    from humanoid_companion.envs import load_env as load

    base_cfg = default_config("Op3Joystick")
    base_cfg.obs_noise = 0.0
    base = load("Op3Joystick", base_cfg)
    ours = _hardened(pushes=False, latency_prob=0.0, joint_offset_range=0.0)
    s_base = jax.jit(base.reset)(jax.random.PRNGKey(5))
    s_ours = s_base.replace(info={**s_base.info, "joint_offset": jp.zeros(20), "delayed": jp.array(False),
                                  "pending_action": jp.zeros(20)})
    step_base, step_ours = jax.jit(base.step), jax.jit(ours.step)
    for a in np.random.default_rng(0).uniform(-0.5, 0.5, (5, 20)).astype(np.float32):
        s_base, s_ours = step_base(s_base, jp.asarray(a)), step_ours(s_ours, jp.asarray(a))
        np.testing.assert_allclose(np.asarray(s_ours.obs), np.asarray(s_base.obs), atol=1e-6)
        np.testing.assert_allclose(np.asarray(s_ours.data.qpos), np.asarray(s_base.data.qpos), atol=1e-6)
        assert float(s_ours.reward) == pytest.approx(float(s_base.reward), abs=1e-6)


def test_latency_delays_the_servos_but_not_the_observed_actions():
    env = _hardened(pushes=False, latency_prob=1.0, joint_offset_range=0.0)
    step = jax.jit(env.step)
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert bool(s.info["delayed"])
    acts = np.random.default_rng(1).uniform(-0.5, 0.5, (3, 20)).astype(np.float32)
    targets, obs = [], []
    for a in acts:
        s = step(s, jp.asarray(a))
        targets.append(np.asarray(s.info["motor_targets"]))
        obs.append(np.asarray(s.obs))
    default = np.asarray(env._default_pose)
    np.testing.assert_allclose(targets[0], np.clip(default, env._lowers, env._uppers), atol=1e-6)  # zeros first
    np.testing.assert_allclose(targets[1], np.clip(default + 0.3 * acts[0], env._lowers, env._uppers), atol=1e-6)
    np.testing.assert_allclose(obs[2][29:49], acts[1], atol=1e-6)  # observation semantics unchanged


def test_encoder_offsets_are_constant_per_episode_and_in_range():
    env = _hardened(pushes=False, latency_prob=0.0, joint_offset_range=0.03)
    s = jax.jit(env.reset)(jax.random.PRNGKey(2))
    off = np.asarray(s.info["joint_offset"])
    assert np.abs(off).max() <= 0.03 and off.std() > 0.005
    s = jax.jit(env.step)(s, jp.zeros(20))
    joints_seen = np.asarray(s.obs)[9:29]
    np.testing.assert_allclose(joints_seen, np.asarray(s.data.qpos[7:]) + off - np.asarray(env._default_pose), atol=1e-6)
    np.testing.assert_allclose(np.asarray(s.info["joint_offset"]), off)


def _max_push_force(pushes: bool) -> float:
    cfg = default_config("Op3JoystickHardened")
    cfg.pushes = pushes
    cfg.kick_wait_times = [0.1, 0.2]  # push within a few steps instead of 1-3 s
    env = load_env("Op3JoystickHardened", cfg)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    peak = 0.0
    for _ in range(25):
        state = step(state, jp.zeros(env.action_size))
        peak = max(peak, float(jp.abs(state.data.xfrc_applied).max()))
    return peak


def test_pushes_apply_a_force_to_the_torso_only_when_enabled():
    assert _max_push_force(True) > 1.0
    assert _max_push_force(False) == 0.0
