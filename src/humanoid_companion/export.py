"""Export a training run's policy to a NumPy file the robot can run.

    python -m humanoid_companion.export --run runs/baseline          # -> exports/baseline/policy.npz
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from humanoid_companion.policy import NumpyPolicy, PolicySpec


def spec_from_env(env_name: str, env_config: dict | None = None) -> PolicySpec:
    """Robot constants from the same env (and config) the policy was trained in."""
    from humanoid_companion.envs import config_from_run, default_config, load_env

    cfg = config_from_run(env_name, env_config) if env_config else default_config(env_name)
    env = load_env(env_name, cfg)
    m = env.mj_model
    return PolicySpec(
        obs_size=int(env.observation_size if isinstance(env.observation_size, int) else env.observation_size["state"][0]),
        action_size=int(env.action_size),
        action_scale=float(cfg.action_scale),
        default_pose=np.asarray(env._default_pose, np.float32),
        ctrl_lower=np.asarray(m.actuator_ctrlrange[:, 0], np.float32),
        ctrl_upper=np.asarray(m.actuator_ctrlrange[:, 1], np.float32),
        ctrl_dt=float(cfg.ctrl_dt),
        actuator_names=tuple(m.actuator(i).name for i in range(m.nu)),
        obs_history=int(cfg.obs_history_size),
    )


def policy_from_params(params, spec: PolicySpec) -> NumpyPolicy:
    """params = (normaliser state, policy params, value params) as saved by humanoid_companion.train."""
    normaliser, policy_params, _ = params
    layers = policy_params["params"]
    names = sorted(layers, key=lambda k: int(k.split("_")[1]))
    return NumpyPolicy(
        obs_mean=normaliser.mean, obs_std=normaliser.std,
        weights=[layers[n]["kernel"] for n in names], biases=[layers[n]["bias"] for n in names],
        spec=spec,
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="a runs/<name> directory with params.pkl and config.json")
    p.add_argument("--checkpoint", help="'latest' or a step: export runs/<name>/ckpt/<step> instead of params.pkl")
    p.add_argument("--out", type=Path, help="default: exports/<run name>[-<step>]/policy.npz")
    args = p.parse_args(argv)

    config = json.loads((args.run / "config.json").read_text())
    ppo_net = config["ppo"]["network_factory"]
    if ppo_net.get("policy_obs_key", "state") != "state":
        raise SystemExit("only flat 'state' observations are supported")
    name = args.run.name
    if args.checkpoint:
        from brax.training.agents.ppo import checkpoint

        from humanoid_companion.train import latest_checkpoint

        ckpt = latest_checkpoint(args.run / "ckpt") if args.checkpoint == "latest" else args.run / "ckpt" / f"{int(args.checkpoint):015d}"
        if ckpt is None or not ckpt.exists():
            raise SystemExit(f"no checkpoint {args.checkpoint} in {args.run / 'ckpt'}")
        params = checkpoint.load(ckpt.resolve())
        name = f"{name}-{int(ckpt.name)}"
    else:
        with (args.run / "params.pkl").open("rb") as f:
            params = pickle.load(f)
    policy = policy_from_params(params, spec_from_env(config["env"], config.get("env_config")))
    out = args.out or Path("exports") / name / "policy.npz"
    policy.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
