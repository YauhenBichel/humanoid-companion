"""The characters move like people, not like clocks: measured, so a change that brings the clock back fails."""

import math

import numpy as np
import pytest

from humanoid_companion.character import CharacterRenderer
from humanoid_companion.motion import Choreography, Drift, EmphasisDetector, Gaze, Spring
from humanoid_companion.singers import DANCE_POSES, REST_ARM, SINGERS, Pose, SingerRenderer
from humanoid_companion.teammates import BYTE

FPS = 30


def speech_mouth(seconds: float, seed: int = 0) -> np.ndarray:
    """A mouth track like speech: syllables of varied length and loudness in phrases, with pauses."""
    rng = np.random.default_rng(seed)
    mouth, t = np.zeros(int(seconds * FPS)), 0.0
    while t < seconds - 1:
        phrase_end = t + rng.uniform(1.5, 4.0)
        while t < min(phrase_end, seconds - 0.3):
            length = rng.uniform(0.12, 0.3)
            start, end = int(t * FPS), int((t + length) * FPS)
            mouth[start:end] = rng.uniform(0.35, 1.0) * np.sin(np.linspace(0, np.pi, max(1, end - start)))
            t += length + rng.uniform(0.02, 0.1)
        t += rng.uniform(0.5, 1.4)  # a pause between phrases
    return mouth


def fast_part(signal: np.ndarray) -> np.ndarray:
    slow = np.convolve(signal, np.ones(FPS) / FPS, mode="same")
    return (signal - slow)[FPS:-FPS]


def strongest_rhythm(signal: np.ndarray) -> float:
    """The share of the fast motion's power in its single strongest frequency (0.5-5 Hz)."""
    x = fast_part(signal)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), 1 / FPS)
    band = (freq > 0.5) & (freq < 5)
    return float(power[band].max() / power[band].sum())


def byte_motion(mouth: np.ndarray):
    renderer = CharacterRenderer(BYTE, 300, 380, fps=FPS)
    out = []
    for value in mouth:
        out.append(renderer.motion(float(value), 0.0, None))
        renderer.t += renderer.dt
    return np.array(out)  # angle, dip, dx, breath per frame


def test_a_critically_damped_spring_eases_in_and_never_overshoots():
    spring, values = Spring(frequency=2.0, damping=1.0), []
    for _ in range(60):
        values.append(spring.step(1.0, 1 / FPS))
    assert values[0] < 0.1 and all(b >= a for a, b in zip(values, values[1:]))
    assert max(values) <= 1.0 + 1e-6 and values[-1] > 0.99


def test_drift_is_slow_bounded_and_does_not_repeat():
    drift = Drift(seed=3, amplitude=2.0)
    values = np.array([drift.at(t / FPS) for t in range(60 * FPS)])
    assert np.abs(values).max() <= 2.0 and np.abs(np.diff(values)).max() < 0.05
    first, second = values[: 20 * FPS], values[20 * FPS : 40 * FPS]
    assert np.abs(first - second).max() > 0.3


def test_emphasis_lands_on_some_syllables_and_marks_the_start_of_a_phrase():
    detector, mouth = EmphasisDetector(fps=FPS), speech_mouth(30, seed=1)
    hits = [(i, detector.step(m), detector.phrase_start) for i, m in enumerate(mouth)]
    emphases = [(i, phrase) for i, strength, phrase in hits if strength]
    syllables = sum(1 for a, b in zip(mouth, mouth[1:]) if a < 0.07 <= b)
    assert 0.2 * syllables < len(emphases) < 0.8 * syllables  # some stressed syllables, not every one
    assert all(b - a >= 0.55 * FPS - 1 for (a, _p), (b, _q) in zip(emphases, emphases[1:]))
    assert any(phrase for _i, phrase in emphases) and not all(phrase for _i, phrase in emphases)


def test_the_eyes_mostly_hold_on_the_viewer_and_glance_aside_now_and_then():
    gaze = Gaze(seed=4)
    looks = np.array([gaze.step(1 / FPS) for _ in range(60 * FPS)])
    at_viewer = np.mean(np.abs(looks[:, 0]) < 0.15)
    assert 0.5 < at_viewer < 0.95
    glances = np.sum((np.abs(looks[1:, 0]) >= 0.3) & (np.abs(looks[:-1, 0]) < 0.3))
    assert 3 <= glances <= 30  # a glance aside every 2 to 20 seconds
    again = Gaze(seed=4)
    assert np.array_equal(looks, np.array([again.step(1 / FPS) for _ in range(60 * FPS)]))  # the same every render


def test_the_dance_never_repeats_a_pose_twice_in_a_row_and_is_not_a_fixed_loop():
    dance = Choreography(len(DANCE_POSES), seed=0)
    poses = [pose for _start, pose in dance.moves[:64]]
    lengths = {round(b[0] - a[0], 3) for a, b in zip(dance.moves, dance.moves[1:])}
    assert all(a != b for a, b in zip(poses, poses[1:]))
    assert lengths == {2.0, 4.0}
    assert -1 in poses  # the arm rests now and then
    for period in range(1, 9):  # no loop of any short length
        assert any(poses[i] != poses[i + period] for i in range(len(poses) - period)), period


def test_byte_speaking_has_no_metronome_and_is_never_frozen():
    mouth = speech_mouth(40)
    motion = byte_motion(mouth)
    angle, dip = motion[:, 0], motion[:, 1]
    assert strongest_rhythm(angle) < 0.15  # the old head wobble put ~40% of its motion in one 1.7 Hz rhythm
    assert np.abs(np.diff(angle)).max() < 1.0 and np.abs(np.diff(dip)).max() < 3.0  # smooth: no jumps
    assert np.abs(angle).max() < 7.0
    still = [i for i in range(1, len(mouth)) if mouth[i] < 0.07 and mouth[i - 1] < 0.07]
    moving = np.abs(np.diff(angle))[np.array(still) - 1] + np.abs(np.diff(motion[:, 3]))[np.array(still) - 1]
    assert np.mean(moving > 0.003) > 0.6  # in the pauses it still breathes and drifts


def test_byte_nods_follow_the_voice():
    mouth = speech_mouth(40, seed=2)
    dip = byte_motion(mouth)[:, 1]
    nods = [i for i in range(1, len(dip)) if dip[i - 1] <= 0.5 < dip[i]]
    assert len(nods) >= 8
    silent = np.convolve(mouth >= 0.07, np.ones(FPS // 2), mode="same") == 0  # half a second of silence around
    assert not any(silent[i] for i in nods)


@pytest.mark.parametrize("key", sorted(SINGERS))
def test_every_move_between_two_dance_poses_stays_in_the_frame(key):
    """The seeded dance can go from any pose to any other (or to rest), with the forearm overshooting a little."""
    renderer = SingerRenderer(SINGERS[key], 180, 240, supersample=1)
    arms = [REST_ARM] + [arm for _name, arm in DANCE_POSES]
    for a0, b0, _h in arms:
        for a1, b1, hand in arms:
            for ease in (0.25, 0.5, 0.75):
                for overshoot in (-8.0, 8.0):
                    arm = (a0 + (a1 - a0) * ease, b0 + (b1 - b0) * ease + overshoot, hand)
                    alpha = renderer.draw(Pose(arm=arm, sing=1.0))[..., 3]
                    assert max(int(edge.max()) for edge in (alpha[:, :2], alpha[:, -2:], alpha[:2, :])) == 0


@pytest.mark.parametrize("key", sorted(SINGERS))
def test_a_singer_dancing_moves_smoothly_and_differently_every_bar(key):
    renderer = SingerRenderer(SINGERS[key], 180, 240, supersample=1)
    mouth = speech_mouth(24, seed=3)
    poses = [renderer.plan("happy", float(m), 0.9, (i / FPS * 2.0) % 1.0) for i, m in enumerate(mouth)]
    arm = np.array([pose.arm[:2] for pose in poses])
    head = np.array([pose.head_angle for pose in poses])
    assert np.abs(np.diff(arm, axis=0)).max() < 12.0  # degrees per frame: fast moves, never a jump
    assert np.abs(np.diff(head)).max() < 1.5
    bar = 2 * FPS  # four beats at 120 bpm
    bars = [arm[i : i + bar] for i in range(bar, len(arm) - bar, bar)]
    assert any(np.abs(a - b).max() > 20 for a, b in zip(bars, bars[1:]))
    gaze = np.array([pose.face[6] for pose in poses])
    assert np.ptp(gaze) > 0.2  # the eyes move


def test_a_singer_nods_on_the_words_when_not_dancing():
    renderer = SingerRenderer(SINGERS["alesia"], 180, 240, supersample=1)
    mouth = speech_mouth(20, seed=5)
    poses = [renderer.plan("happy", float(m)) for m in mouth]
    head_dy = np.array([pose.head_dy for pose in poses])
    angle = np.array([pose.head_angle for pose in poses])
    assert head_dy.max() > 2.0
    assert strongest_rhythm(angle) < 0.15
    assert math.isfinite(float(np.abs(angle).max())) and np.abs(angle).max() < 8.0
