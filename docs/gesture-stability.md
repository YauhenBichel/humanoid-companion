# Gesture stability

`python -m humanoid_companion.gestures --check` (the bundled walking policy): each gesture layered over the walking policy in C MuJoCo. A gesture is allowed only if it never triggers a safety stop; `walk_ok` gestures are also run while walking.

| gesture | case | safety stop | worst tilt (up-vector z) | sideways m | forward m |
|---|---|---|---|---|---|
| wave | standing, from rest | - | 0.992 | -0.0037 | -0.0256 |
| wave | standing, after 1 s | - | 0.993 | -0.0023 | -0.0146 |
| nod | standing, from rest | - | 1.0 | 0.0001 | -0.0047 |
| nod | standing, after 1 s | - | 0.999 | -0.0004 | 0.001 |
| nod | walking 0.4 m/s | - | 1.0 | -0.1242 | 0.8979 |
| celebrate | standing, from rest | - | 0.995 | -0.0338 | -0.0191 |
| celebrate | standing, after 1 s | - | 0.998 | -0.007 | -0.019 |
| look_around | standing, from rest | - | 0.999 | 0.0209 | 0.0101 |
| look_around | standing, after 1 s | - | 0.999 | 0.0202 | 0.0157 |
| look_around | walking 0.4 m/s | - | 0.998 | -0.0426 | 1.361 |
| dance | standing, from rest | - | 0.992 | -0.0145 | -0.0049 |
| dance | standing, after 1 s | - | 0.981 | -0.0204 | -0.0023 |

Found while tuning: a raised right arm tips the robot over even standing unless the left arm counter-balances; both arms up (celebrate) is stable standing but tips it over while walking.
