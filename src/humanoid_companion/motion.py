"""Motion that reads as alive: the building blocks the characters share.

A drawn character looks mechanical when its movement is a clock: a head rocking on a fixed sine, the
same arm poses in the same order, eyes fixed on one spot, a body frozen in every pause. People move
for reasons: a nod lands on a stressed syllable, the head settles into a new tilt when a new phrase
starts, the eyes look away and come back, and even a still person breathes and drifts a little.

    spring = Spring(frequency=2.2, damping=0.8)
    angle = spring.step(target, dt)          # follows the target with weight: eases in, settles, no jumps

    drift = Drift(seed=3, amplitude=1.2)     # slow, never-repeating wander (degrees, pixels, ...)
    drift.at(t)

    beats = EmphasisDetector(fps=30)         # the syllables the voice leans on, from the mouth track
    beats.step(mouth)                         # -> strength 0..1 on the frame an emphasis starts, else 0

    gaze = Gaze(seed=1)                      # holds a look, glances away quickly, mostly at the viewer
    gaze.step(dt, speaking=True)             # -> (x, y) offsets in -1..1

    Choreography(seed=0).pose_at(beats)      # which dance pose, and how far into the move

Every function is deterministic for a seed, so a video renders the same twice and tests can pin it.
"""

import math
from dataclasses import dataclass

import numpy as np


class Spring:
    """A damped spring on one value: `frequency` in Hz, `damping` 1 = critically damped (no overshoot).

    Moving a value through a spring instead of setting it gives the ease-in, the weight and the settle of a
    real movement; below 1, damping lets it overshoot a little and come back, like a nod that bounces."""

    def __init__(self, frequency: float = 2.0, damping: float = 1.0, value: float = 0.0):
        self.omega, self.damping = 2 * math.pi * frequency, damping
        self.value, self.velocity = value, 0.0

    def step(self, target: float, dt: float) -> float:
        substeps = max(1, math.ceil(dt * self.omega / 0.1))  # semi-implicit Euler, accurate at any fps
        h = dt / substeps
        for _ in range(substeps):
            acceleration = self.omega**2 * (target - self.value) - 2 * self.damping * self.omega * self.velocity
            self.velocity += acceleration * h
            self.value += self.velocity * h
        return self.value


class Drift:
    """Smooth wander around zero: three sines at unrelated slow frequencies with seeded phases.

    The sum never repeats within a video (the frequencies have no common period below minutes) and its
    speed stays low, so it reads as a body that is never quite still rather than as a movement."""

    FREQUENCIES = (0.071, 0.113, 0.187)  # Hz: one cycle in 14, 9 and 5 seconds
    WEIGHTS = (0.5, 0.33, 0.17)

    def __init__(self, seed: int = 0, amplitude: float = 1.0, speed: float = 1.0):
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0, 2 * math.pi, len(self.FREQUENCIES))
        self.amplitude, self.speed = amplitude, speed

    def at(self, t: float) -> float:
        total = sum(w * math.sin(2 * math.pi * f * self.speed * t + p)
                    for f, w, p in zip(self.FREQUENCIES, self.WEIGHTS, self.phases))  # fmt: skip
        return self.amplitude * total


class EmphasisDetector:
    """Finds the moments a voice leans on a syllable, one frame at a time, from the mouth track (0..1).

    An emphasis starts when the mouth opens clearly wider than it has been lately (its slow average), and
    only after `refractory` seconds since the last one: people nod on some stressed syllables, not on every
    syllable. `phrase_start` is True on the first emphasis after a pause of `pause` seconds."""

    def __init__(self, fps: float = 30.0, refractory: float = 0.55, pause: float = 0.45, threshold: float = 0.18):
        self.dt, self.refractory, self.pause, self.threshold = 1.0 / fps, refractory, pause, threshold
        self.slow, self.previous = 0.0, 0.0
        self.since_emphasis, self.since_sound = 10.0, 10.0
        self.phrase_start = False

    def step(self, mouth: float) -> float:
        """0 on most frames; on the frame an emphasis starts, its strength (0.3..1)."""
        self.since_emphasis += self.dt
        rising = mouth - self.previous
        strength = 0.0
        self.phrase_start = False
        if (mouth > 0.25 and rising > 0.04 and mouth - self.slow > self.threshold
                and self.since_emphasis >= self.refractory):  # fmt: skip
            strength = float(np.clip(0.3 + (mouth - self.slow) * 1.4, 0.3, 1.0))
            self.phrase_start = self.since_sound >= self.pause
            self.since_emphasis = 0.0
        self.since_sound = 0.0 if mouth >= 0.07 else self.since_sound + self.dt
        self.slow += (mouth - self.slow) * (1 - 0.92 ** (30 * self.dt))
        self.previous = mouth
        return strength


@dataclass
class Look:
    x: float
    y: float
    hold: float  # seconds


class Gaze:
    """Where the eyes point: held for a while, then a quick glance (a saccade) to a new spot.

    Speaking or singing, people look at their listener most of the time and glance aside briefly;
    in pauses they look aside a little more often. Offsets are in -1..1 of the eye's travel."""

    def __init__(self, seed: int = 0, at_viewer: float = 0.7):
        self.rng = np.random.default_rng(seed)
        self.at_viewer = at_viewer
        self.x, self.y = 0.0, 0.0
        self.look = Look(0.0, 0.0, 1.0 + self.rng.random())
        self.held = 0.0

    def step(self, dt: float, speaking: bool = True) -> tuple[float, float]:
        self.held += dt
        if self.held >= self.look.hold:
            self.held = 0.0
            if self.rng.random() < (self.at_viewer if speaking else self.at_viewer - 0.25):
                self.look = Look(self.rng.normal(0, 0.05), self.rng.normal(0, 0.04), 1.4 + self.rng.random() * 2.4)
            else:  # a glance aside: most often sideways and a little down, briefly
                side = 1 if self.rng.random() < 0.5 else -1
                self.look = Look(side * self.rng.uniform(0.35, 0.8), self.rng.uniform(-0.15, 0.45),
                                 0.5 + self.rng.random() * 1.1)  # fmt: skip
        k = 1 - 0.25 ** (30 * dt)  # a saccade: most of the way in about 60 ms, then settled
        self.x += (self.look.x - self.x) * k
        self.y += (self.look.y - self.y) * k
        return self.x, self.y


class Choreography:
    """A dance as a sequence of poses from a fixed set, chosen by a seed instead of cycled in order.

    Each move lasts 2 or 4 beats (2 more often), never repeats the pose
    before it, and now and then returns to `rest` for a move: a loop of four poses in one order, forever,
    is the clearest sign of a machine."""

    def __init__(self, poses: int, seed: int = 0, rest_every: float = 0.18, length: int = 512):
        rng = np.random.default_rng(seed)
        self.moves: list[tuple[float, int]] = []  # (start beat, pose index; -1 is rest)
        beat, previous = 0.0, None
        for _ in range(length):
            if previous is not None and previous != -1 and rng.random() < rest_every:
                pose = -1
            else:
                choices = [p for p in range(poses) if p != previous]
                pose = int(rng.choice(choices))
            self.moves.append((beat, pose))
            beat += 2.0 if rng.random() < 0.6 else 4.0
            previous = pose

    def pose_at(self, beats: float) -> tuple[int, int, float]:
        """(pose before, pose now, beats since the move began); poses are indices, -1 is rest."""
        beats = max(0.0, beats)
        starts = [start for start, _pose in self.moves]
        i = min(len(self.moves) - 1, int(np.searchsorted(starts, beats, side="right")) - 1)
        before = self.moves[i - 1][1] if i > 0 else -1
        return before, self.moves[i][1], beats - self.moves[i][0]
