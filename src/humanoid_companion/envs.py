"""Our environments on top of MuJoCo Playground, and one way to load any of them.

Op3JoystickHardened = Playground's Op3Joystick plus, for sim-to-real:
- pushes: the perturbation Playground ships but leaves commented out in `step`;
- action latency: per episode, with probability `latency_prob`, the servos receive the previous
  action instead of the current one (one 20 ms control period late);
- encoder offsets: per episode, a constant calibration error U(-r, r) on every observed joint
  angle (`joint_offset_range`, rad).
Latency changes only what the servos receive. The observation keeps Playground's semantics
(the frame after action a_t reports a_{t-1} as the commanded action), so the deployment loop is
unchanged. `step` is Playground's Joystick.step (Apache-2.0, mujoco_playground 0.2.0) with those
additions; tests/test_randomize.py pins it to the original when the additions are off.

Domain randomization is a separate function (Brax applies it per environment when training):
floor friction, joint friction and armature, link masses, torso mass and the PD gains. Ranges
follow Playground's Berkeley Humanoid randomizer, scaled for a 3 kg robot.
"""

from typing import Callable

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.op3 import joystick as op3_joystick

OUR_ENVS = ("Op3JoystickHardened",)
BASE_ENV = {"Op3JoystickHardened": "Op3Joystick"}


class Op3JoystickHardened(op3_joystick.Joystick):
    def reset(self, rng):
        rng, offset_rng, latency_rng, noise_rng = jax.random.split(rng, 4)
        state = super().reset(rng)
        r = self._config.joint_offset_range
        state.info["joint_offset"] = jax.random.uniform(offset_rng, (self.action_size,), minval=-r, maxval=r)
        state.info["delayed"] = jax.random.bernoulli(latency_rng, self._config.latency_prob)
        state.info["pending_action"] = jp.zeros(self.action_size)
        # Recompute the first observation now that the offset exists.
        obs = self._get_obs(state.data, state.info, jp.zeros_like(state.obs), noise_rng)
        return state.replace(obs=obs)

    def _get_obs(self, data, info, obs_history, rng):
        offset = info.get("joint_offset", jp.zeros(self.action_size))
        obs = jp.concatenate([
            self.get_gyro(data),
            self.get_gravity(data),
            info["command"],
            data.qpos[7:] + offset - self._default_pose,
            info["last_act"],
        ])
        if self._config.obs_noise >= 0.0:
            noise = self._config.obs_noise * jax.random.uniform(rng, obs.shape, minval=-1.0, maxval=1.0)
            obs = jp.clip(obs, -100.0, 100.0) + noise
        return jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def step(self, state, action):
        cfg = self._config
        if cfg.pushes:
            rng, push_rng = jax.random.split(state.info["rng"])
            state.info["rng"] = rng
            state = self._maybe_apply_perturbation(state, push_rng)

        rng, cmd_rng, noise_rng = jax.random.split(state.info["rng"], 3)
        applied = jp.where(state.info["delayed"], state.info["pending_action"], action)
        motor_targets = jp.clip(self._default_pose + applied * cfg.action_scale, self._lowers, self._uppers)
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        obs = self._get_obs(data, state.info, state.obs, noise_rng)
        done = self._get_termination(data)
        rewards = self._get_reward(data, action, state.info, state.metrics, done)
        rewards = {k: v * cfg.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["pending_action"] = action
        state.info["motor_targets"] = motor_targets
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["last_vel"] = data.qvel[6:]
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["command"] = jp.where(state.info["step"] > 500, self.sample_command(cmd_rng), state.info["command"])
        state.info["step"] = jp.where(done | (state.info["step"] > 500), 0, state.info["step"])
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        return state.replace(data=data, obs=obs, reward=reward, done=jp.float32(done))


def hardened_config() -> config_dict.ConfigDict:
    cfg = op3_joystick.default_config()
    cfg.pushes = True
    cfg.latency_prob = 0.5          # half the episodes one control period late: robust to both
    cfg.joint_offset_range = 0.03   # rad (~1.7 deg), a plausible Dynamixel horn calibration error
    return cfg


def default_config(name: str) -> config_dict.ConfigDict:
    from mujoco_playground import registry

    return hardened_config() if name == "Op3JoystickHardened" else registry.get_default_config(name)


def load_env(name: str, config: config_dict.ConfigDict | None = None, impl: str = "jax"):
    """Playground's registry for its envs, ours for ours; impl defaults to MJX-JAX (not Warp)."""
    from mujoco_playground import registry

    cfg = config if config is not None else default_config(name)
    cfg.impl = impl
    if name == "Op3JoystickHardened":
        return Op3JoystickHardened(config=cfg)
    return registry.load(name, config=cfg)


def config_from_run(name: str, saved: dict) -> config_dict.ConfigDict:
    """An env config rebuilt from a run's config.json (reward scales included)."""
    cfg = default_config(name)
    cfg.update({k: v for k, v in saved.items() if k in cfg and k != "reward_config"})
    if "reward_config" in saved:
        cfg.reward_config.update(saved["reward_config"])
    return cfg


# Ranges: (low, high). Multiplicative unless named *_add.
RANGES = {
    "floor_friction": (0.4, 1.0),
    "frictionloss_scale": (0.9, 1.1),
    "armature_scale": (1.0, 1.05),
    "mass_scale": (0.9, 1.1),
    "torso_mass_add": (-0.3, 0.3),
    "kp_scale": (0.8, 1.2),
    "kd_scale": (0.8, 1.2),
}


def make_randomizer(mj_model, ranges: dict = RANGES) -> Callable:
    """Brax randomization_fn(mjx_model, rng[num_envs]) -> (batched model, in_axes)."""
    floor = mj_model.geom("floor").id
    torso = mj_model.body("body_link").id
    nu = mj_model.nu
    ndof = mj_model.nv - 6

    def randomize(model, rng):
        @jax.vmap
        def sample(key):
            k = jax.random.split(key, 7)
            u = lambda i, name, shape=(): jax.random.uniform(k[i], shape, minval=ranges[name][0], maxval=ranges[name][1])
            friction = model.geom_friction.at[floor, 0].set(u(0, "floor_friction"))
            frictionloss = model.dof_frictionloss.at[6:].multiply(u(1, "frictionloss_scale", (ndof,)))
            armature = model.dof_armature.at[6:].multiply(u(2, "armature_scale", (ndof,)))
            mass = model.body_mass * u(3, "mass_scale", (model.nbody,))
            mass = mass.at[torso].add(u(4, "torso_mass_add"))
            kp = model.actuator_gainprm[:, 0] * u(5, "kp_scale", (nu,))
            gainprm = model.actuator_gainprm.at[:, 0].set(kp)
            biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
            damping = model.dof_damping.at[6:].multiply(u(6, "kd_scale", (ndof,)))
            return friction, frictionloss, armature, mass, gainprm, biasprm, damping

        values = sample(rng)
        fields = ("geom_friction", "dof_frictionloss", "dof_armature", "body_mass",
                  "actuator_gainprm", "actuator_biasprm", "dof_damping")
        in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({f: 0 for f in fields})
        return model.tree_replace(dict(zip(fields, values))), in_axes

    return randomize


def randomizer_for(name: str, env) -> Callable | None:
    return make_randomizer(env.mj_model) if name in OUR_ENVS else None
