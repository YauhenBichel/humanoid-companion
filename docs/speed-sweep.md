# Speed sweep: the bundled walking policy

`python -m humanoid_companion.speed_sweep --policy src/humanoid_companion/policies/op3_walk.npz --name op3_walk` (the bundled policy); each command held 5.0 s from standing, C MuJoCo, deployment loop.

| command vx | achieved vx | lateral m | yaw change rad | safety stop |
|---|---|---|---|---|
| 0.0 | -0.001 | 0.0001 | -0.0012 | - |
| 0.1 | -0.0 | -0.0001 | -0.0104 | - |
| 0.2 | 0.001 | -0.0004 | -0.0288 | - |
| 0.3 | 0.288 | -0.0081 | 0.0048 | - |
| 0.4 | 0.393 | -0.3528 | -0.1021 | - |
| 0.5 | 0.513 | -0.1048 | -0.1044 | - |
| 0.8 | 0.639 | 0.135 | -0.0652 | - |
| 1.0 | 0.65 | -0.1388 | -0.1633 | - |
| -0.3 | -0.267 | 0.0618 | -0.0019 | - |

Usable forward range (achieved within 25 % of the command, at least ±0.05 m/s): **0.3–0.8 m/s**
