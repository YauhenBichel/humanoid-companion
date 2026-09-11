"""Train a walking policy with Brax PPO on a MuJoCo Playground environment.

    python -m humanoid_companion.train --name smoke-cpu --num-envs 64 --num-timesteps 200000 --num-evals 2

Writes to runs/<name>/: metrics.jsonl (one line per evaluation), config.json,
ckpt/<step>/ (Brax checkpoints, resumable with --restore) and params.pkl (the
final normaliser, policy and value parameters as NumPy arrays).
"""

import argparse
import functools
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np


def to_jsonable(metrics: dict) -> dict:
    """Brax metrics (JAX/NumPy scalars) as plain floats; non-finite values kept as strings."""
    out = {}
    for key, value in metrics.items():
        x = float(np.asarray(value))
        out[key] = x if math.isfinite(x) else str(x)
    return out


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    steps = sorted((p for p in ckpt_dir.glob("*") if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    return steps[-1] if steps else None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run name, the directory under --runs-dir")
    p.add_argument("--env", default="Op3Joystick", help="Playground environment name")
    p.add_argument("--impl", default="jax", help="MJX implementation; Playground's default 'warp' is NVIDIA-only")
    p.add_argument("--num-timesteps", type=int, help="override Playground's tuned PPO config")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-evals", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--restore", action="store_true", help="resume from the run's latest checkpoint")
    p.add_argument("--cpu-devices", type=int,
                   help="split the CPU into N JAX devices so Brax runs num_envs/N envs on each in parallel")
    p.add_argument("--env-set", action="append", default=[], metavar="KEY=VALUE",
                   help="override an env config value, e.g. reward_config.tracking_sigma=0.1 (JSON values; repeatable)")
    p.add_argument("--randomize", action="store_true",
                   help="use Playground's own domain randomizer for this env (ours are always on for our envs)")
    return p.parse_args(argv)


def ppo_config_for(env_name: str):
    """Playground's tuned PPO settings for the env's family (locomotion or manipulation)."""
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params, manipulation_params

    from humanoid_companion.envs import BASE_ENV

    base = BASE_ENV.get(env_name, env_name)
    if base in registry.manipulation.ALL_ENVS:
        return manipulation_params.brax_ppo_config(base)
    return locomotion_params.brax_ppo_config(base)


def apply_env_overrides(cfg, overrides: list[str]) -> None:
    """KEY=VALUE with dotted keys into an ml_collections ConfigDict. Unknown keys are errors, so a
    typo cannot silently train the default."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--env-set {item!r}: expected KEY=VALUE")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            if part not in node:
                raise SystemExit(f"--env-set {key}: no config section {part!r}")
            node = node[part]
        if leaf not in node:
            raise SystemExit(f"--env-set {key}: no such config key (have: {', '.join(sorted(node.keys()))})")
        node[leaf] = value


def cpu_device_flags(existing: str, n: int | None) -> str:
    """XLA_FLAGS with the host-device count set (replacing any earlier value)."""
    kept = [f for f in existing.split() if not f.startswith("--xla_force_host_platform_device_count=")]
    return " ".join(kept + ([f"--xla_force_host_platform_device_count={n}"] if n else []))


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.cpu_devices:
        # Must be set before JAX initialises its backends.
        os.environ["XLA_FLAGS"] = cpu_device_flags(os.environ.get("XLA_FLAGS", ""), args.cpu_devices)
    # Imported here so --help does not pay for JAX start-up.
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper

    from humanoid_companion.envs import default_config, load_env, randomizer_for

    run_dir = (Path(args.runs_dir) / args.name).resolve()
    ckpt_dir = run_dir / "ckpt"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = default_config(args.env)
    apply_env_overrides(env_cfg, args.env_set)
    env = load_env(args.env, env_cfg, impl=args.impl)
    eval_env = load_env(args.env, env_cfg, impl=args.impl)
    randomization_fn = randomizer_for(args.env, env)
    if randomization_fn is None and args.randomize:
        randomization_fn = registry.get_domain_randomizer(args.env)
        if randomization_fn is None:
            raise SystemExit(f"--randomize: Playground has no domain randomizer for {args.env}")

    ppo_cfg = ppo_config_for(args.env)
    for key in ("num_timesteps", "num_envs", "num_evals"):
        if getattr(args, key) is not None:
            ppo_cfg[key] = getattr(args, key)
    ppo_kwargs = dict(ppo_cfg)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_kwargs.pop("network_factory"))

    restore = latest_checkpoint(ckpt_dir) if args.restore else None
    if args.restore and restore is None:
        raise SystemExit(f"--restore: no checkpoint in {ckpt_dir}")

    (run_dir / "config.json").write_text(json.dumps({
        "env": args.env,
        "env_config": env_cfg.to_dict(),
        "ppo": {**ppo_kwargs, "network_factory": dict(ppo_cfg.network_factory)},
        "seed": args.seed,
        "env_overrides": args.env_set,
        "randomized": randomization_fn is not None,
        "pushes": bool(env_cfg.get("pushes", False)),
        "restored_from": str(restore) if restore else None,
        "jax_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }, indent=2, default=str))

    metrics_file = run_dir / "metrics.jsonl"
    started = time.monotonic()

    def progress(num_steps: int, metrics: dict) -> None:
        row = {"step": int(num_steps), "wall_s": round(time.monotonic() - started, 1), **to_jsonable(metrics)}
        with metrics_file.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"step {row['step']:>12,}  reward {row.get('eval/episode_reward', float('nan')):>9.3f}  "
              f"{row['wall_s']:>8.1f}s", flush=True)

    _, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomization_fn,
        network_factory=network_factory,
        progress_fn=progress,
        seed=args.seed,
        save_checkpoint_path=str(ckpt_dir),
        restore_checkpoint_path=str(restore) if restore else None,
        **ppo_kwargs,
    )

    with (run_dir / "params.pkl").open("wb") as f:
        pickle.dump(jax.tree.map(np.asarray, params), f)
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
