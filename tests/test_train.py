import json

import numpy as np

import pytest

from humanoid_companion.train import apply_env_overrides, cpu_device_flags, latest_checkpoint, parse_args, to_jsonable


def op3_config():
    from humanoid_companion.envs import default_config

    return default_config("Op3JoystickHardened")


def test_ppo_settings_come_from_the_envs_family():
    from humanoid_companion.train import ppo_config_for

    walk, hand = ppo_config_for("Op3JoystickHardened"), ppo_config_for("LeapCubeRotateZAxis")
    assert walk.network_factory.value_obs_key == "state" and walk.unroll_length == 20
    assert hand.network_factory.value_obs_key == "privileged_state" and hand.unroll_length == 40
    assert hand.num_timesteps == 100_000_000


def test_env_overrides_set_nested_and_typed_values():
    cfg = op3_config()
    apply_env_overrides(cfg, ["reward_config.tracking_sigma=0.1", "lin_vel_x=[0.2, 1.0]", "pushes=false",
                              "reward_config.scales.zero_cmd=-1.5"])
    assert cfg.reward_config.tracking_sigma == 0.1 and list(cfg.lin_vel_x) == [0.2, 1.0]
    assert cfg.pushes is False and cfg.reward_config.scales.zero_cmd == -1.5


@pytest.mark.parametrize("bad", ["reward_config.tracking_sigmaa=0.1", "rewards.tracking_sigma=1", "pushes"])
def test_a_typo_is_an_error_not_a_silent_default(bad):
    with pytest.raises(SystemExit):
        apply_env_overrides(op3_config(), [bad])


def test_overrides_survive_the_round_trip_through_config_json():
    from humanoid_companion.envs import config_from_run

    cfg = op3_config()
    apply_env_overrides(cfg, ["reward_config.tracking_sigma=0.1", "latency_prob=0.0"])
    rebuilt = config_from_run("Op3JoystickHardened", json.loads(json.dumps(cfg.to_dict())))
    assert rebuilt.reward_config.tracking_sigma == 0.1 and rebuilt.latency_prob == 0.0


def test_cpu_device_flag_is_added_and_replaces_an_earlier_value():
    assert cpu_device_flags("", 8) == "--xla_force_host_platform_device_count=8"
    assert cpu_device_flags("--a=1 --xla_force_host_platform_device_count=2", 16) == \
        "--a=1 --xla_force_host_platform_device_count=16"
    assert cpu_device_flags("--a=1", None) == "--a=1"


def test_to_jsonable_turns_array_scalars_into_floats():
    row = to_jsonable({"eval/episode_reward": np.float32(1.5), "steps": np.array(3)})
    assert row == {"eval/episode_reward": 1.5, "steps": 3.0}
    json.dumps(row)


def test_to_jsonable_keeps_non_finite_values_visible_and_valid_json():
    row = to_jsonable({"loss": np.float32("nan"), "grad": np.inf})
    assert row == {"loss": "nan", "grad": "inf"}
    json.dumps(row, allow_nan=False)


def test_latest_checkpoint_is_the_highest_step_not_the_last_name(tmp_path):
    for step in ("000000009830400", "000000019660800", "000000002000000"):
        (tmp_path / step).mkdir()
    (tmp_path / "notes").mkdir()
    assert latest_checkpoint(tmp_path).name == "000000019660800"
    assert latest_checkpoint(tmp_path / "missing") is None


def test_defaults_target_op3_on_mjx_jax_not_warp():
    args = parse_args(["--name", "x"])
    assert (args.env, args.impl) == ("Op3Joystick", "jax")
