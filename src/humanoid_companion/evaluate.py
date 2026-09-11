"""Evaluate a trained run in MJX: fixed forward command, many episodes, one rollout video.

    python -m humanoid_companion.evaluate --run runs/baseline               # -> runs/baseline/eval.json, rollout.mp4
    MUJOCO_GL=egl python -m humanoid_companion.evaluate --run runs/baseline  # headless rendering on a Linux server

Metrics over `--episodes` parallel episodes of `--seconds`, command [vx, 0, 0] held fixed:
forward_speed_error  mean |v_forward - vx| over every step a robot is still up
fall_rate            fraction of episodes that hit the env's termination (fall or joint limit)
"""

import argparse
import functools
import json
import pickle
from pathlib import Path

import numpy as np


def episode_metrics(vx: np.ndarray, done: np.ndarray, command_vx: float) -> dict:
    """vx, done: [steps, episodes]. A robot counts until (not including) its first done step."""
    fell = done.astype(bool)
    alive = ~np.logical_or.accumulate(fell, axis=0)
    alive_steps = alive.sum()
    err = np.abs(vx - command_vx)
    return {
        "forward_speed_error": float((err * alive).sum() / max(alive_steps, 1)),
        "mean_forward_speed": float((vx * alive).sum() / max(alive_steps, 1)),
        "fall_rate": float(fell.any(axis=0).mean()),
        "mean_time_up_fraction": float(alive.mean()),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--video-only", action="store_true", help="only render rollout.mp4; leave eval.json alone")
    p.add_argument("--randomized", action="store_true", help="apply the run's domain randomization (train.py --randomize)")
    args = p.parse_args(argv)

    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import wrapper

    from humanoid_companion.envs import config_from_run, load_env, make_randomizer

    config = json.loads((args.run / "config.json").read_text())
    env_cfg = config_from_run(config["env"], config["env_config"])
    env = load_env(config["env"], env_cfg)

    with (args.run / "params.pkl").open("rb") as f:
        normaliser, policy_params, _ = pickle.load(f)
    net_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in config["ppo"]["network_factory"].items()}
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize, **net_cfg)
    policy = ppo_networks.make_inference_fn(net)((normaliser, policy_params), deterministic=True)

    command = jp.array([args.vx, 0.0, 0.0])
    steps = int(round(args.seconds / env.dt))

    def with_command(state):
        info = {**state.info, "command": jp.broadcast_to(command, state.info["command"].shape)}
        return state.replace(info=info)

    keys = jax.random.split(jax.random.PRNGKey(0), args.episodes)
    if args.randomized:
        # One randomized model per episode, exactly as Brax does in training. The wrapper also
        # auto-resets fallen robots; metrics only count each robot until its first done.
        rand = functools.partial(make_randomizer(env.mj_model), rng=jax.random.split(jax.random.PRNGKey(42), args.episodes))
        benv = wrapper.wrap_for_brax_training(env, episode_length=steps + 1, randomization_fn=rand)
        batch_reset, batch_step = benv.reset, benv.step
    else:
        batch_reset, batch_step = jax.vmap(env.reset), jax.vmap(env.step)

    @jax.jit
    def rollout(keys):
        state = with_command(batch_reset(keys))

        def body(state, key):
            action, _ = policy(state.obs, key)
            state = with_command(batch_step(state, action))
            vx = jax.vmap(env.get_local_linvel)(state.data)[:, 0]
            return state, (vx, state.done)

        _, (vx, done) = jax.lax.scan(body, state, jax.random.split(keys[0], steps))
        return vx, done

    report = {"run": str(args.run), "command_vx": args.vx, "episodes": args.episodes, "seconds": args.seconds,
              "randomized": args.randomized, "pushes": bool(env_cfg.get("pushes", False))}
    if not args.video_only:
        vx, done = rollout(keys)
        report.update(episode_metrics(np.asarray(vx), np.asarray(done), args.vx))

    if not args.no_video:
        import mediapy

        state = with_command(env.reset(jax.random.PRNGKey(1)))
        step = jax.jit(env.step)
        single_policy = jax.jit(functools.partial(policy))
        trajectory = [state]
        for i in range(steps):
            action, _ = single_policy(state.obs[None], jax.random.PRNGKey(i))
            state = with_command(step(state, action[0]))
            trajectory.append(state)
        frames = env.render(trajectory, height=480, width=640)
        mediapy.write_video(args.run / "rollout.mp4", frames, fps=int(round(1.0 / env.dt)))
        report["video"] = str(args.run / "rollout.mp4")

    if not args.video_only:
        (args.run / "eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
