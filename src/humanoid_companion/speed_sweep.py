"""How well does a policy track forward-speed commands through the deployment loop?

    python -m humanoid_companion.speed_sweep --policy src/humanoid_companion/policies/op3_walk.npz --name op3_walk
    -> speed-sweep-<name>.md (table) and speed-sweep-<name>.json

Each command is held for --seconds in C MuJoCo via run_sim.simulate. The achieved speed is
forward distance / time, so it includes the start-up from standing.
"""

import argparse
import json
from pathlib import Path

from humanoid_companion.policy import NumpyPolicy
from humanoid_companion.robot_io import MujocoRobotIO
from humanoid_companion.run_sim import simulate, training_model

DEFAULT_SPEEDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, -0.3]


def tracked(command: float, achieved: float) -> bool:
    """Within 25 % of the command, never tighter than 0.05 m/s. (A flat 0.1 m/s tolerance
    counted standing still as tracking a 0.1 m/s command.)"""
    return abs(achieved - command) <= max(0.05, 0.25 * abs(command))


def usable_range(rows: list[dict]) -> tuple[float, float] | None:
    """Smallest and largest positive command that is tracked, with no safety stop."""
    ok = [r["command_vx"] for r in rows
          if r["command_vx"] > 0 and not r["safety_stops"] and tracked(r["command_vx"], r["achieved_vx"])]
    return (min(ok), max(ok)) if ok else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    p.add_argument("--seconds", type=float, default=5.0)
    args = p.parse_args(argv)

    policy = NumpyPolicy.load(args.policy)
    model = training_model()
    rows = []
    for vx in args.speeds:
        r = simulate(policy, [vx, 0.0, 0.0], args.seconds, io=MujocoRobotIO(model, ctrl_dt=policy.spec.ctrl_dt))
        rows.append({"command_vx": vx, "achieved_vx": round(r["forward_distance_m"] / max(r["seconds"], 1e-9), 3),
                     "lateral_m": r["lateral_distance_m"], "yaw_change_rad": r["yaw_change_rad"],
                     "safety_stops": r["safety_stops"], "seconds": r["seconds"]})
        print(rows[-1], flush=True)

    rng = usable_range(rows)
    out = Path("reports") / f"speed-sweep-{args.name}"
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"policy": str(args.policy), "usable_range": rng, "rows": rows}, indent=2))
    lines = [f"# Speed sweep: {args.name}", "", f"`python -m humanoid_companion.speed_sweep --policy {args.policy}`; each command held "
             f"{args.seconds} s from standing, C MuJoCo, deployment loop.", "",
             "| command vx | achieved vx | lateral m | yaw change rad | safety stop |", "|---|---|---|---|---|"]
    lines += [f"| {r['command_vx']} | {r['achieved_vx']} | {r['lateral_m']} | {r['yaw_change_rad']} | "
              f"{', '.join(r['safety_stops']) or '-'} |" for r in rows]
    lines += ["", f"Usable forward range (achieved within 25 % of the command, at least ±0.05 m/s): "
              + (f"**{rng[0]}–{rng[1]} m/s**" if rng else "**none**"), ""]
    out.with_suffix(".md").write_text("\n".join(lines))
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
