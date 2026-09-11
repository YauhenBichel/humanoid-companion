"""NumPy policy == Brax's deterministic inference, for the network shape Playground trains for OP3."""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from humanoid_companion.export import policy_from_params
from humanoid_companion.policy import NumpyPolicy, PolicySpec

OBS, ACT = 147, 20


@pytest.fixture(scope="module")
def brax_policy():
    net = ppo_networks.make_ppo_networks(
        OBS, ACT, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128,) * 4, value_hidden_layer_sizes=(256,) * 5,
    )
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(7), 4)
    policy_params = net.policy_network.init(k1)
    value_params = net.value_network.init(k2)
    normaliser = running_statistics.init_state(specs.Array((OBS,), jp.float32))
    normaliser = normaliser.replace(mean=jax.random.normal(k3, (OBS,)),
                                    std=jp.exp(jax.random.normal(k4, (OBS,)) * 0.5))
    infer = ppo_networks.make_inference_fn(net)((normaliser, policy_params), deterministic=True)
    return (normaliser, policy_params, value_params), jax.jit(infer)


def spec():
    return PolicySpec(obs_size=OBS, action_size=ACT, action_scale=0.3, default_pose=np.zeros(ACT, np.float32),
                      ctrl_lower=-np.ones(ACT, np.float32), ctrl_upper=np.ones(ACT, np.float32), ctrl_dt=0.02,
                      actuator_names=tuple(f"j{i}" for i in range(ACT)), obs_history=3)


def test_numpy_policy_matches_brax_inference(brax_policy, tmp_path):
    params, infer = brax_policy
    policy = policy_from_params(jax.tree.map(np.asarray, params), spec())
    policy.save(tmp_path / "policy.npz")
    loaded = NumpyPolicy.load(tmp_path / "policy.npz")

    obs = np.random.default_rng(0).normal(size=(1000, OBS)).astype(np.float32) * 2.0
    want, _ = infer(jp.asarray(obs), jax.random.PRNGKey(0))
    got = loaded.act(obs)
    worst = float(np.abs(np.asarray(want) - got).max())
    assert worst < 1e-5, f"max abs diff {worst:.2e}"
    assert np.abs(got).max() > 0.5, "the test should exercise the non-linear part of tanh"


def test_single_observation_gives_one_action(brax_policy):
    params, _ = brax_policy
    policy = policy_from_params(jax.tree.map(np.asarray, params), spec())
    assert policy.act(np.zeros(OBS, np.float32)).shape == (ACT,)


def test_load_rejects_another_format_version(brax_policy, tmp_path):
    params, _ = brax_policy
    policy_from_params(jax.tree.map(np.asarray, params), spec()).save(tmp_path / "p.npz")
    z = dict(np.load(tmp_path / "p.npz"))
    z["format_version"] = 99
    np.savez(tmp_path / "old.npz", **z)
    with pytest.raises(ValueError, match="format 99"):
        NumpyPolicy.load(tmp_path / "old.npz")
