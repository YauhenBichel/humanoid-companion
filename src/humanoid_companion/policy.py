"""The trained policy in NumPy, for the robot's computer: no JAX, no Brax.

Mirrors Brax PPO's deterministic inference for `make_ppo_networks` defaults:
normalise (obs - mean) / std, MLP with swish hidden layers and a linear output of 2 x action_size
(loc, scale), action = tanh(loc).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def swish(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), written with tanh so large |x| cannot overflow."""
    return 0.5 * x * (1.0 + np.tanh(0.5 * x))


@dataclass(frozen=True)
class PolicySpec:
    """Everything the control loop needs besides the network."""
    obs_size: int
    action_size: int
    action_scale: float
    default_pose: np.ndarray   # joint targets for action 0, actuator order
    ctrl_lower: np.ndarray     # actuator control range, also the joint limits used for safety
    ctrl_upper: np.ndarray
    ctrl_dt: float
    actuator_names: tuple[str, ...]
    obs_history: int


class NumpyPolicy:
    def __init__(self, obs_mean, obs_std, weights, biases, spec: PolicySpec):
        self.obs_mean = np.asarray(obs_mean, np.float32)
        self.obs_std = np.asarray(obs_std, np.float32)
        self.weights = [np.asarray(w, np.float32) for w in weights]
        self.biases = [np.asarray(b, np.float32) for b in biases]
        self.spec = spec
        if self.weights[-1].shape[1] != 2 * spec.action_size:
            raise ValueError(f"output layer {self.weights[-1].shape} is not 2 x action_size {spec.action_size}")

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action in [-1, 1]^action_size for one observation (or a batch)."""
        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = swish(x @ w + b)
        out = x @ self.weights[-1] + self.biases[-1]
        return np.tanh(out[..., : self.spec.action_size])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.spec
        np.savez(
            path,
            format_version=FORMAT_VERSION,
            obs_mean=self.obs_mean, obs_std=self.obs_std,
            n_layers=len(self.weights),
            **{f"w{i}": w for i, w in enumerate(self.weights)},
            **{f"b{i}": b for i, b in enumerate(self.biases)},
            obs_size=s.obs_size, action_size=s.action_size, action_scale=s.action_scale,
            default_pose=s.default_pose, ctrl_lower=s.ctrl_lower, ctrl_upper=s.ctrl_upper,
            ctrl_dt=s.ctrl_dt, actuator_names=np.array(s.actuator_names), obs_history=s.obs_history,
        )

    @classmethod
    def load(cls, path: Path) -> "NumpyPolicy":
        z = np.load(path, allow_pickle=False)
        if int(z["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"{path}: format {int(z['format_version'])}, expected {FORMAT_VERSION}")
        n = int(z["n_layers"])
        spec = PolicySpec(
            obs_size=int(z["obs_size"]), action_size=int(z["action_size"]),
            action_scale=float(z["action_scale"]), default_pose=z["default_pose"],
            ctrl_lower=z["ctrl_lower"], ctrl_upper=z["ctrl_upper"], ctrl_dt=float(z["ctrl_dt"]),
            actuator_names=tuple(str(a) for a in z["actuator_names"]), obs_history=int(z["obs_history"]),
        )
        return cls(z["obs_mean"], z["obs_std"], [z[f"w{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)], spec)
