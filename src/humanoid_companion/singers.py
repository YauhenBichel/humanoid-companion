"""Two singers drawn as modern 2D motion-graphics characters, frame by frame, with alpha: Alesia and Maks.

The robot teammates (humanoid_companion.character) show a face on a screen; these singers are young
people, drawn as vector shapes with Pillow (supersampled for smooth edges): gradient shading lit from
above, a coloured rim light on the edges, a dark outline, and a cool light edge around the whole
figure so they read over dark night footage and bright footage alike. Waist up, each holding a
wireless microphone, sized to be used small in a corner of a video (about 300 px wide).

    renderer = SingerRenderer(SINGERS["alesia"], 540, 720, fps=30)
    rgba = renderer.frame("happy", mouth=0.6, energy=0.8, beat_phase=0.25, vowel="o")

What drives what (the same per-frame signals humanoid_companion.perform already makes):

- `mouth` (0..1, the voice's loudness, or the sung-word gate): closed below 0.07, a small opening
  below 0.3, above that the shape of the vowel being sung: `vowel` "a" opens, "e" is wide, "o" is
  round (`vowel_track` finds the vowel from the sung words' timings). Shapes blend, never jump.
- the microphone hand follows the mouth: up beside the chin while the singer sings (a little closer
  when loud), lowered to the chest after a pause, always below and to the side of the mouth.
- `beat_phase` (0..1 inside the beat, None when not dancing): the singer counts beats itself, sways
  once per two beats, dips softly on each beat, and moves the free arm through the dance poses in a
  seeded order (humanoid_companion.motion.Choreography): 2 or 4 beats each, never the same pose twice
  in a row, now and then a rest. The arm goes through springs, the forearm a little behind the upper
  arm, so a move has weight and follow-through. No punches, no shake.
- the head nods on the syllables the voice leans on and lifts at the start of a phrase (brows up a
  touch); the eyes hold a look and glance aside now and then (humanoid_companion.motion.Gaze); a slow
  drift and breathing keep the body alive in every pause.
- `energy` (0..1, loudness of the music): how big the dance is, smoothed over about 0.3 s.
- `expression` (the face's seven expressions): brows, eye opening, squint, smile, gaze, head tilt.
- blinks every 2.5-5.5 s; hair (bun, tendrils, fringe) follows the head through a damped spring.

Design units: every size is in a 540 x 720 reference box (the character's centre line at x = 0, y down),
scaled to the frame.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from humanoid_companion.motion import Choreography, Drift, EmphasisDetector, Gaze, Spring

Colour = tuple[int, int, int]

REFERENCE = (540.0, 720.0)
OUTLINE = (20, 18, 34)  # near-black navy: every shape's outline
EDGE = (214, 240, 255)  # the cool light edge around the whole figure
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
MOUTH_INSIDE, TEETH, TONGUE = (60, 16, 36), (250, 248, 252), (214, 96, 124)
SCLERA = (250, 251, 255)
SILVER = (206, 212, 224)
RIM_SHIFT = (-4.5, 3.0)  # the rim light shows on the edges facing up and to the right


def mix(a: Colour, b: Colour, amount: float) -> Colour:
    return tuple(int(round(a[i] + (b[i] - a[i]) * amount)) for i in range(3))


@dataclass(frozen=True)
class SingerStyle:
    """How one singer looks."""

    key: str
    name: str  # in Belarusian
    hair: str  # "bun" or "fringe"
    skin: Colour
    skin_shadow: Colour
    hair_colour: Colour
    hair_light: Colour  # highlights
    hair_dark: Colour
    brow: Colour
    iris: Colour
    lips: Colour
    jacket: Colour
    inner: Colour  # the top or hoodie under the open jacket
    cuff: Colour
    ornament: Colour  # the small rhombus motif on the cuffs
    accent: Colour  # headphone rings, the microphone's ring
    rim: Colour  # the rim light's colour
    headphones: Colour
    mic_side: int  # the hand holding the microphone: -1 screen left, 1 screen right
    lashes: bool
    face_width: float  # half width at the cheekbones
    jaw: float  # 0 soft, 1 angular
    blush: float  # 0..1
    smirk: float  # the right corner of a smile higher than the left
    brow_attitude: float  # one brow a touch higher
    headphone_size: float  # 1 slim, larger is chunkier


ALESIA = SingerStyle(
    key="alesia",
    name="Алеся",
    hair="bun",
    skin=(248, 216, 198),
    skin_shadow=(222, 174, 156),
    hair_colour=(212, 190, 160),  # ash blonde
    hair_light=(244, 234, 216),
    hair_dark=(146, 120, 96),
    brow=(128, 100, 80),
    iris=(84, 128, 176),  # blue-grey
    lips=(200, 72, 128),  # berry
    jacket=(62, 108, 255),  # electric blue cropped puffer
    inner=(24, 22, 32),  # black top
    cuff=(20, 18, 28),
    ornament=(236, 70, 176),  # magenta
    accent=(236, 70, 176),
    rim=(255, 110, 214),  # magenta rim light
    headphones=(226, 230, 240),  # silver
    mic_side=-1,
    lashes=True,
    face_width=84.0,
    jaw=0.0,
    blush=0.45,
    smirk=0.12,
    brow_attitude=0.0,
    headphone_size=1.0,
)

MAKS = SingerStyle(
    key="maks",
    name="Максім",
    hair="fringe",
    skin=(236, 198, 172),
    skin_shadow=(204, 158, 130),
    hair_colour=(64, 46, 36),  # dark brown
    hair_light=(128, 98, 78),
    hair_dark=(30, 22, 18),
    brow=(46, 34, 28),
    iris=(92, 122, 118),  # grey-green
    lips=(170, 104, 96),
    jacket=(40, 40, 50),  # black leather jacket
    inner=(16, 128, 138),  # deep teal oversized hoodie
    cuff=(16, 128, 138),
    ornament=SILVER,
    accent=(60, 156, 255),  # electric blue
    rim=(80, 222, 255),  # cyan rim light
    headphones=(36, 36, 46),  # matte black, chunky
    mic_side=1,
    lashes=False,
    face_width=82.0,
    jaw=1.0,
    blush=0.0,
    smirk=0.35,
    brow_attitude=3.0,
    headphone_size=1.25,
)

SINGERS = {style.key: style for style in (ALESIA, MAKS)}

# --- mouth -------------------------------------------------------------------------------------------

# half width, opening height, roundness (0 a D shape, 1 an oval), teeth showing (0..1)
MOUTH_SHAPES = {
    "closed": (22.0, 0.0, 0.0, 0.0),
    "small": (17.0, 12.0, 0.45, 0.0),
    "open": (23.0, 31.0, 0.3, 1.0),
    "wide": (30.0, 19.0, 0.0, 1.0),
    "o": (14.0, 26.0, 1.0, 0.0),
}
VOWEL_SHAPES = {"a": "open", "e": "wide", "o": "o"}
VOWELS = {
    **dict.fromkeys("ая", "a"), **dict.fromkeys("оёую", "o"), **dict.fromkeys("эеіыи", "e"),
    **dict.fromkeys("a", "a"), **dict.fromkeys("ou", "o"), **dict.fromkeys("eiy", "e"),
}  # fmt: skip


def word_vowels(word: str) -> list[str]:
    """The vowel classes of a word in order ("ў" is not a vowel); ["a"] when it has none."""
    return [VOWELS[ch] for ch in word.lower() if ch in VOWELS] or ["a"]


def vowel_track(lines: list[dict], times: np.ndarray) -> list[str | None]:
    """The vowel being sung at each time, from the sung words: each word's time is split evenly
    between its vowels. None between words, and for lines without word timings."""
    track: list[str | None] = [None] * len(times)
    for line in lines:
        for word in line.get("words") or []:
            start, end = float(word["start"]), float(word["end"])
            vowels = word_vowels(str(word.get("word", "")))
            step = max(1e-6, (end - start) / len(vowels))
            for i in np.nonzero((times >= start - 0.04) & (times <= end + 0.04))[0]:
                track[i] = vowels[min(len(vowels) - 1, max(0, int((times[i] - start) / step)))]
    return track


def mouth_target(mouth: float, vowel: str | None, rest: str = "closed") -> tuple[float, ...]:
    """The mouth shape parameters for a loudness (0..1) and the vowel sung."""
    if mouth < 0.07:
        return MOUTH_SHAPES[rest]
    if mouth < 0.3:
        return MOUTH_SHAPES["small"]
    w, h, r, teeth = MOUTH_SHAPES[VOWEL_SHAPES.get(vowel or "a", "open")]
    return (w, h * (0.6 + 0.4 * min(1.0, mouth)), r, teeth)


# --- expressions ----------------------------------------------------------------------------------

# brow raise, brow inner lift, right brow extra raise, eye opening, squint, smile, gaze x, gaze y,
# head tilt (degrees); then the resting mouth
FACES = {
    "neutral": ((0, 0, 0, 1.0, 0.0, 0.3, 0, 0, 0), "closed"),
    "happy": ((3, -1, 0, 1.0, 0.4, 0.9, 0, 0, 3), "closed"),
    "thinking": ((2, 0, 8, 0.9, 0.0, -0.05, -0.6, -0.7, -5), "closed"),
    "surprised": ((10, 3, 0, 1.22, 0.0, 0.0, 0, 0, 0), "o"),
    "sad": ((-1, 8, 0, 0.85, 0.0, -0.6, 0, 0.5, 4), "closed"),
    "listening": ((3, 1, 0, 1.05, 0.1, 0.4, 0.25, 0, 7), "closed"),
    "sleeping": ((-2, 0, 0, 0.0, 0.0, 0.1, 0, 0.3, 9), "closed"),
}

# --- dance -----------------------------------------------------------------------------------------

# The free arm (the one without the microphone): (upper arm angle, forearm angle, hand), degrees from
# straight down, positive away from the body. Forearm angles are all negative (-90 points across the
# body, -180 straight up), so moving between poses sweeps across the body, never out of the frame.
# Hands are only loose fists or clearly open palms: no single raised finger, no raised fist.
REST_ARM = (8, -22, "fist")
DANCE_POSES = (
    ("groove", (26, -95, "fist")),
    ("reach", (76, -166, "open")),
    ("chest", (12, -118, "fist")),
    ("palm up", (40, -150, "open")),
)
UPPER_ARM, FOREARM = 100.0, 92.0
SHOULDER = (116.0, 426.0)  # the right shoulder joint; mirrored for the left
HEAD_SCALE = 1.2  # a large head, so the face reads at 300 px
FIGURE_DROP = 26.0  # the whole figure sits this much lower in the box, so the bun and the raised hand fit
MOUTH_Y = 68.0  # in head coordinates
MIC_GRILLE = 58.0  # from the grip to the grille's centre


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def arm_at(beats: float, amount: float, choreography: Choreography | None = None) -> tuple[float, float, str]:
    """The free arm at a beat count, eased into each pose over one beat, then held with a gentle pump;
    `amount` (0..1) blends from the resting arm. Without a choreography the poses go in their fixed
    order, one per two beats (design sheets); with one, in its seeded order and lengths."""
    if choreography is None:
        segment = beats / 2.0
        index = int(math.floor(segment)) % len(DANCE_POSES)
        before, now, since = index - 1, index, (segment - math.floor(segment)) * 2.0
    else:
        before, now, since = choreography.pose_at(beats)
    ease = smoothstep(since)
    (a0, b0, h0), (a1, b1, h1) = (REST_ARM if p == -1 else DANCE_POSES[p % len(DANCE_POSES)][1] for p in (before, now))
    ar, br, hr = REST_ARM
    pump = 6.0 * math.sin(2 * math.pi * beats)  # the forearm moves with each beat, softly
    a, b = a0 + (a1 - a0) * ease, b0 + (b1 - b0) * ease
    a, b = ar + (a - ar) * amount, br + (b + pump - br) * amount
    hand = (h1 if ease > 0.5 else h0) if amount > 0.3 else hr
    return (a, b, hand)


def two_bone(shoulder: np.ndarray, target: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """(elbow, wrist) reaching `target`, the elbow bent outwards and down."""
    d = target - shoulder
    dist = float(np.clip(np.linalg.norm(d), abs(UPPER_ARM - FOREARM) + 1e-3, UPPER_ARM + FOREARM - 1e-3))
    base = math.atan2(d[1], d[0])
    bend = math.acos((UPPER_ARM**2 + dist**2 - FOREARM**2) / (2 * UPPER_ARM * dist))
    options = [shoulder + UPPER_ARM * np.array([math.cos(base + s * bend), math.sin(base + s * bend)]) for s in (-1, 1)]
    elbow = max(options, key=lambda e: e[1] + 0.3 * side * e[0])
    wrist = elbow + FOREARM * (target - elbow) / max(1e-6, float(np.linalg.norm(target - elbow)))
    return elbow, wrist


@dataclass(frozen=True)
class Pose:
    """Everything one frame shows, already decided; SingerRenderer.draw turns it into pixels."""

    face: tuple[float, ...] = FACES["neutral"][0]
    mouth: tuple[float, float, float, float] = MOUTH_SHAPES["closed"]
    blink: float = 0.0  # 0 open .. 1 shut
    head_angle: float = 0.0  # degrees, positive tips the head to the screen right
    head_dy: float = 0.0  # a nod: the head dips this far (design units)
    body_angle: float = 0.0
    body_dx: float = 0.0
    body_dy: float = 0.0
    arm: tuple = REST_ARM  # the free arm
    sing: float = 1.0  # 1 the microphone is up at the mouth, 0 lowered
    push: float = 0.0  # 0..1 the microphone a touch closer when loud
    hair_swing: float = 0.0  # degrees the hair lags behind the head


# --- vector drawing --------------------------------------------------------------------------------


def bezier(p0, p1, p2, p3, n: int = 16) -> list[tuple[float, float]]:
    s = np.linspace(0.0, 1.0, n)[:, None]
    p = (1 - s) ** 3 * np.array(p0) + 3 * (1 - s) ** 2 * s * np.array(p1) + 3 * (1 - s) * s**2 * np.array(p2) + s**3 * np.array(p3)
    return [tuple(q) for q in p]


def path(start, *segments, n: int = 16) -> list[tuple[float, float]]:
    """An outline from cubic segments (c1, c2, end) and straight points (end,)."""
    points, current = [start], start
    for segment in segments:
        if len(segment) == 3:
            points += bezier(current, *segment, n=n)[1:]
            current = segment[2]
        else:
            points.append(segment[0])
            current = segment[0]
    return points


def mirrored(points) -> list[tuple[float, float]]:
    return [(-x, y) for x, y in points]


def ellipse_points(cx, cy, rx, ry, n: int = 40, start: float = 0.0, end: float = 2 * math.pi):
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in np.linspace(start, end, n)]


def tapered(centre, start_width: float, end_width: float) -> list[tuple[float, float]]:
    """A stroke as a polygon whose width runs from `start_width` to `end_width` (brows, locks, wings)."""
    c = np.asarray(centre, dtype=float)
    tangent = np.gradient(c, axis=0)
    tangent /= np.maximum(1e-6, np.linalg.norm(tangent, axis=1))[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    widths = np.linspace(start_width, end_width, len(c))[:, None] / 2
    return [tuple(p) for p in c + normal * widths] + [tuple(p) for p in (c - normal * widths)[::-1]]


def rotation(angle_degrees: float, about=(0.0, 0.0)) -> np.ndarray:
    a = math.radians(angle_degrees)
    c, s = math.cos(a), math.sin(a)
    x, y = about
    return np.array([[c, -s, x - c * x + s * y], [s, c, y - s * x - c * y], [0, 0, 1]])


def scaling(factor: float, about=(0.0, 0.0)) -> np.ndarray:
    x, y = about
    return np.array([[factor, 0, x - factor * x], [0, factor, y - factor * y], [0, 0, 1]])


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)


class Pen:
    """Draws polygons, gradients and strokes in design units through a transform stack onto an RGBA image."""

    def __init__(self, image: Image.Image, matrix: np.ndarray, scale: float):
        self.image, self.draw = image, ImageDraw.Draw(image)
        self.stack, self.scale = [matrix], scale

    @property
    def matrix(self) -> np.ndarray:
        return self.stack[-1]

    @contextmanager
    def transformed(self, matrix: np.ndarray):
        self.stack.append(self.matrix @ matrix)
        try:
            yield self
        finally:
            self.stack.pop()

    def xy(self, points) -> list[tuple[float, float]]:
        p = np.asarray(points, dtype=float)
        q = p @ self.matrix[:2, :2].T + self.matrix[:2, 2]
        return [tuple(v) for v in q]

    def point(self, x: float, y: float) -> np.ndarray:
        return self.matrix[:2, :2] @ np.array([x, y]) + self.matrix[:2, 2]

    def shape(self, points, fill: Colour, outline: Colour | None = OUTLINE, width: float = 3.4) -> None:
        xy = self.xy(points)
        self.draw.polygon(xy, fill=(*fill, 255))
        if outline is not None:
            self._line(xy + xy[:2], outline, width)

    def fill(self, points, top: Colour, bottom: Colour, rim: Colour | None = None,
             outline: Colour | None = OUTLINE, width: float = 3.4) -> None:  # fmt: skip
        """A polygon shaded from `top` to `bottom` (screen vertical), with a rim light on its upper right edges."""
        xy = self.xy(points)
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        left, upper = max(0, int(min(xs)) - 1), max(0, int(min(ys)) - 1)
        right, lower = min(self.image.width, int(max(xs)) + 2), min(self.image.height, int(max(ys)) + 2)
        if right - left < 2 or lower - upper < 2:
            return
        size = (right - left, lower - upper)
        local = [(x - left, y - upper) for x, y in xy]
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).polygon(local, fill=255)
        rows = (np.arange(size[1]) - (min(ys) - upper)) / max(1.0, max(ys) - min(ys))
        t = np.clip(rows, 0, 1)[:, None, None]
        rgb = np.broadcast_to(np.array(top) * (1 - t) + np.array(bottom) * t, (size[1], size[0], 3)).copy()
        if rim is not None:
            dx, dy = RIM_SHIFT[0] * self.scale, RIM_SHIFT[1] * self.scale
            shifted = Image.new("L", size, 0)
            ImageDraw.Draw(shifted).polygon([(x + dx, y + dy) for x, y in local], fill=255)
            edge = (np.asarray(mask) > 0) & (np.asarray(shifted) == 0)
            rgb[edge] = rim
        self.image.paste(Image.fromarray(rgb.astype(np.uint8), "RGB"), (left, upper), mask)
        if outline is not None:
            self._line(xy + xy[:2], outline, width)

    def stroke(self, points, colour: Colour, width: float, caps: bool = True) -> None:
        xy = self.xy(points)
        self._line(xy, colour, width)
        if caps:
            r = width * self.scale / 2
            for x, y in (xy[0], xy[-1]):
                self.draw.ellipse([x - r, y - r, x + r, y + r], fill=(*colour, 255))

    def dot(self, cx, cy, r, fill: Colour, outline: Colour | None = None, width: float = 2.4) -> None:
        self.shape(ellipse_points(cx, cy, r, r, n=28), fill, outline, width)

    def _line(self, xy, colour: Colour, width: float) -> None:
        self.draw.line(xy, fill=(*colour, 255), width=max(1, int(round(width * self.scale))), joint="curve")

    @contextmanager
    def clipped(self, mask_points):
        """Drawing inside the block lands only inside the polygon `mask_points`."""
        xy = self.xy(mask_points)
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        left, top = max(0, int(min(xs)) - 2), max(0, int(min(ys)) - 2)
        right, bottom = min(self.image.width, int(max(xs)) + 3), min(self.image.height, int(max(ys)) + 3)
        if right <= left or bottom <= top:
            yield self
            return
        layer = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
        mask = Image.new("L", layer.size, 0)
        ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in xy], fill=255)
        inner = Pen(layer, self.matrix, self.scale)
        inner.stack = [translation(-left, -top) @ m for m in self.stack]
        yield inner
        mask = Image.fromarray(np.minimum(np.asarray(mask), np.asarray(layer)[..., 3]))
        self.image.paste(layer, (left, top), mask)


# --- the renderer ----------------------------------------------------------------------------------


class SingerRenderer:
    """Stateful: call frame() once per video frame, in order. Frames are RGBA uint8 arrays."""

    NECK = (0.0, 352.0)  # where the head turns, in body coordinates
    CHIN = (0.0, 346.0)  # the head is drawn larger about this point
    HEAD = (0.0, 238.0)  # the head's centre, in body coordinates
    HIP = (0.0, 820.0)  # where the body leans from

    def __init__(self, style: SingerStyle, width: int = 540, height: int = 720, fps: float = 30.0,
                 seed: int = 0, supersample: int = 2):  # fmt: skip
        self.style, self.width, self.height, self.dt = style, width, height, 1.0 / fps
        self.supersample = supersample
        self.scale = min(width / REFERENCE[0], height / REFERENCE[1])
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.face = list(FACES["neutral"][0])
        self.mouth = list(MOUTH_SHAPES["closed"])
        self.energy = 0.0
        self.arm_amount = 0.0
        self.beats, self.last_phase = 0.0, None
        self.next_blink, self.blink_at = 1.5 + self.rng.random() * 2.0, -1.0
        self.hair, self.hair_velocity, self.last_head = 0.0, 0.0, 0.0
        self.sway = 0.0
        self.sing, self.push, self.last_sung = 0.0, 0.0, -10.0
        self.choreography = Choreography(len(DANCE_POSES), seed=seed)
        self.gaze = Gaze(seed=seed + 5)
        self.emphasis = EmphasisDetector(fps=fps)
        self.nod_spring = Spring(frequency=2.2, damping=0.5)  # dips on a stressed syllable, bounces back
        self.lift = Spring(frequency=1.2, damping=1.0)  # the head and brows lift as a phrase starts
        self.lift_until = -1.0
        self.head_drift = Drift(seed=seed + 7, amplitude=1.4)
        self.body_drift = Drift(seed=seed + 8, amplitude=3.0, speed=0.8)
        self.turn = Spring(frequency=1.4, damping=1.0)  # the head follows the eyes, slower than they move
        self.upper = Spring(frequency=3.0, damping=0.9, value=REST_ARM[0])  # the arm leads ...
        self.fore = Spring(frequency=2.3, damping=0.72, value=REST_ARM[1])  # ... the forearm follows through

    # planning ---------------------------------------------------------------------------------------

    def frame(self, expression: str, mouth: float = 0.0, energy: float = 0.0, beat_phase: float | None = None,
              vowel: str | None = None) -> np.ndarray:  # fmt: skip
        return self.draw(self.plan(expression, mouth, energy, beat_phase, vowel))

    def plan(self, expression: str, mouth: float = 0.0, energy: float = 0.0, beat_phase: float | None = None,
             vowel: str | None = None) -> Pose:  # fmt: skip
        dt = self.dt
        values, rest = FACES.get(expression, FACES["neutral"])
        k = 1 - 0.85 ** (60 * dt)  # like the face page: 0.15 per 60 Hz frame
        self.face = [c + (target - c) * k for c, target in zip(self.face, values)]
        target = mouth_target(mouth, vowel, rest)
        km = 1 - 0.45 ** (30 * dt)  # fast enough to follow syllables, never a one-frame flicker
        self.mouth = [c + (t - c) * km for c, t in zip(self.mouth, target)]
        self.energy += (energy - self.energy) * (1 - 0.9 ** (30 * dt))

        # the microphone: up while singing, lowered 1.2 s after the last sung sound, slowly
        if mouth >= 0.07:
            self.last_sung = self.t
        singing = 1.0 if self.t - self.last_sung < 1.2 else 0.0
        rate = 0.8 if singing > self.sing else 0.95
        self.sing += (singing - self.sing) * (1 - rate ** (30 * dt))
        self.push += (min(1.0, mouth) - self.push) * (1 - 0.88 ** (30 * dt))

        if self.t >= self.next_blink and expression != "sleeping":
            self.blink_at, self.next_blink = self.t, self.t + 2.5 + self.rng.random() * 3.0
        since = self.t - self.blink_at
        blink = math.sin(math.pi * since / 0.2) if 0 <= since < 0.2 else 0.0

        dancing = beat_phase is not None
        if dancing:
            if self.last_phase is not None:
                step = beat_phase - self.last_phase
                self.beats += step + 1.0 if step < -0.5 else step
            else:
                self.beats = beat_phase
            self.last_phase = beat_phase
        groove = 0.35 + 0.65 * min(1.0, self.energy * 1.3)
        target_amount = smoothstep((self.energy - 0.12) / 0.45) if dancing else 0.0
        self.arm_amount += (target_amount - self.arm_amount) * (1 - 0.93 ** (30 * dt))
        self.sway += ((groove if dancing else 0.0) - self.sway) * (1 - 0.93 ** (30 * dt))

        pose = self.pose_at(self.beats, self.sway, self.arm_amount, self.t, self.choreography)
        upper, fore, hand = pose.arm
        pose = replace(pose, arm=(self.upper.step(upper, dt), self.fore.step(fore, dt), hand))

        # the voice: a nod on stressed syllables, a lift of head and brows as a phrase starts
        strength = self.emphasis.step(mouth)
        if strength:
            self.nod_spring.velocity += strength * (4.0 if dancing else 7.0)
            if self.emphasis.phrase_start:
                self.lift_until = self.t + 0.7
        lift = self.lift.step(1.0 if self.t < self.lift_until else 0.0, dt)
        nod = self.nod_spring.step(0.0, dt)

        # the eyes: hold, glance aside, come back; the head turns a little with them
        gx, gy = self.gaze.step(dt, speaking=mouth >= 0.07)
        face = list(self.face)  # offsets on top of the eased expression, never stored into it
        face[0] += 3.0 * lift  # brows up a touch while the phrase starts
        face[6] += 0.8 * gx
        face[7] += 0.6 * gy
        turn = self.turn.step(2.0 * gx, dt)
        pose = replace(pose, head_angle=pose.head_angle + turn + self.head_drift.at(self.t) + 1.5 * nod,
                       head_dy=12.0 * nod - 5.0 * lift,
                       body_dx=pose.body_dx + self.body_drift.at(self.t))  # fmt: skip
        head = pose.head_angle + pose.body_angle + self.face[8]
        self._spring(head, dt)
        self.t += dt
        return replace(pose, face=tuple(face), mouth=tuple(self.mouth), blink=blink, hair_swing=self.hair,
                       sing=self.sing, push=self.push)  # fmt: skip

    def _spring(self, head_angle: float, dt: float) -> None:
        """The hair's own angle chases the head's through a damped spring (about 1.2 Hz, no ringing)."""
        steps = 4
        for _ in range(steps):
            h = dt / steps
            acceleration = -60.0 * self.hair - 9.0 * self.hair_velocity
            acceleration -= (head_angle - self.last_head) / dt * 9.0  # the head turning drags the hair back
            self.hair_velocity += acceleration * h
            self.hair += self.hair_velocity * h
        self.last_head = head_angle

    @staticmethod
    def pose_at(beats: float, sway: float, arm_amount: float, t: float = 0.0,
                choreography: Choreography | None = None) -> Pose:  # fmt: skip
        """The body at a beat count: a sway per two beats, a soft dip on every beat, the free arm's pose."""
        swing = math.sin(math.pi * beats)
        breathe = 1.5 * math.sin(2 * math.pi * t / 4.0)
        return Pose(
            head_angle=5.0 * sway * math.sin(math.pi * beats - 0.5),
            body_angle=2.0 * sway * swing,
            body_dx=4.0 * sway * swing,
            body_dy=5.0 * sway * (1 - math.cos(2 * math.pi * beats)) / 2 + breathe,
            arm=arm_at(beats, arm_amount, choreography) if arm_amount > 0 else REST_ARM,
        )

    def still(self, expression: str = "happy", mouth_shape: str | None = None, blink: float = 0.0,
              beats: float | None = None, energy: float = 1.0, sing: float = 1.0) -> np.ndarray:  # fmt: skip
        """One posed frame for design sheets: a named mouth shape, a blink, or a dance pose at a beat."""
        values, rest = FACES[expression]
        pose = Pose(face=values, mouth=MOUTH_SHAPES[mouth_shape or rest], blink=blink, sing=sing)
        if beats is not None:
            body = self.pose_at(beats, energy, energy)
            head = body.head_angle + body.body_angle
            pose = replace(body, face=values, mouth=pose.mouth, blink=blink, sing=sing, hair_swing=-0.8 * head)
        return self.draw(pose)

    def head_box(self, margin: float = 150.0) -> tuple[int, int, int, int]:
        """The frame's pixels around the head at rest (for close-ups), as a square box."""
        scale = self.scale
        cx = self.width / 2
        cy = self.height - (REFERENCE[1] - FIGURE_DROP - (self.CHIN[1] - (self.CHIN[1] - self.HEAD[1]) * HEAD_SCALE)) * scale
        half = margin * HEAD_SCALE * scale
        return (int(cx - half), int(cy - half + 10 * scale), int(cx + half), int(cy + half + 10 * scale))

    # drawing ----------------------------------------------------------------------------------------

    def draw(self, pose: Pose) -> np.ndarray:
        ss, s = self.supersample, self.style
        size = (self.width * ss, self.height * ss)
        image = Image.new("RGBA", size, (0, 0, 0, 0))
        scale = self.scale * ss
        # centre line in the middle of the frame, the reference box's bottom on the frame's bottom
        base = translation(size[0] / 2, size[1] - (REFERENCE[1] - FIGURE_DROP) * scale) @ np.diag([scale, scale, 1.0])
        pen = Pen(image, base, scale)
        body = translation(pose.body_dx, pose.body_dy) @ rotation(pose.body_angle, self.HIP)
        head = (translation(0.0, pose.head_dy) @ rotation(pose.head_angle + pose.face[8], self.NECK)
                @ scaling(HEAD_SCALE, self.CHIN))  # fmt: skip
        with pen.transformed(body):
            with pen.transformed(head):
                self._hair_back(pen, pose)
            self._torso(pen)
            with pen.transformed(head):
                self._head(pen, pose)
            free = -s.mic_side
            self._dance_arm(pen, free, *pose.arm)
            self._mic_arm(pen, pose, head)
        small = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.asarray(self._edge(small))

    def _edge(self, image: Image.Image) -> Image.Image:
        """A cool light edge outside the silhouette, so the outline never meets dark footage directly."""
        alpha = image.getchannel("A")
        spread = alpha.filter(ImageFilter.GaussianBlur(max(1.0, 2.0 * self.scale)))
        edge_alpha = Image.fromarray(np.clip(np.asarray(spread, dtype=np.float32) * 4.0, 0, 215).astype(np.uint8))
        edge = Image.new("RGBA", image.size, (*EDGE, 0))
        edge.putalpha(edge_alpha)
        edge.alpha_composite(image)
        return edge

    def _shaded(self, pen: Pen, points, colour: Colour, rim: bool = True, light: float = 0.16, dark: float = 0.32,
                width: float = 3.4, outline: Colour | None = OUTLINE) -> None:  # fmt: skip
        """`colour` lit from above: lighter at the top, darker at the bottom, with the singer's rim light."""
        top, bottom = mix(colour, WHITE, light), mix(colour, BLACK, dark)
        rim_colour = mix(mix(colour, WHITE, 0.45), self.style.rim, 0.6) if rim else None
        pen.fill(points, top, bottom, rim_colour, outline, width)

    # body

    def _torso(self, pen: Pen) -> None:
        s = self.style
        neck = [(-24, 296), (24, 296), (27, 380), (52, 424), (-52, 424), (-27, 380)]
        pen.fill(neck, mix(s.skin_shadow, BLACK, 0.12), s.skin, None)
        if s.hair == "bun":
            self._puffer(pen)
        else:
            self._leather(pen)

    def _puffer(self, pen: Pen) -> None:
        s = self.style
        top = path((-46, 392), ((-34, 420), (34, 420), (46, 392)), ((132, 760),), ((-132, 760),))
        self._shaded(pen, top, s.inner, rim=False, light=0.12, dark=0.4)
        # a thin silver chain with a small rhombus pendant
        chain = path((-32, 396), ((-20, 420), (20, 420), (32, 396)))
        pen.stroke(chain, OUTLINE, 3.6)
        pen.stroke(chain, SILVER, 1.8)
        pen.fill([(0, 414), (5.5, 422), (0, 430), (-5.5, 422)], WHITE, SILVER, None, OUTLINE, 2.0)
        for side in (-1, 1):
            panel = path((side * 22, 386), ((side * 40, 368), (side * 70, 372), (side * 104, 388)),
                         ((side * 140, 402), (side * 152, 436), (side * 156, 490)),
                         ((side * 162, 560), (side * 166, 610), (side * 156, 648)),
                         ((side * 140, 668), (side * 80, 672), (side * 40, 660)),
                         ((side * 34, 580), (side * 30, 480), (side * 22, 386)))  # fmt: skip
            self._shaded(pen, panel, s.jacket, width=3.8)
            for y in (470, 536, 600):  # quilted channels, each with a highlight above it
                pen.stroke(path((side * 34, y), ((side * 70, y + 12), (side * 120, y + 12), (side * 158, y))),
                           mix(s.jacket, BLACK, 0.45), 2.6)  # fmt: skip
                pen.stroke(path((side * 60, y - 14), ((side * 80, y - 8), (side * 110, y - 8), (side * 128, y - 14))),
                           mix(s.jacket, WHITE, 0.4), 2.2)  # fmt: skip
            collar = path((side * 26, 394), ((side * 22, 360), (side * 32, 338), (side * 54, 338)),
                          ((side * 70, 340), (side * 74, 372), (side * 62, 396)))  # fmt: skip
            self._shaded(pen, collar, mix(s.jacket, WHITE, 0.08), width=3.4)

    def _leather(self, pen: Pen) -> None:
        s = self.style
        hood = path((-98, 406), ((-106, 348), (-58, 326), (0, 330)), ((58, 326), (106, 348), (98, 406)),
                    ((40, 394),), ((-40, 394),))  # fmt: skip
        self._shaded(pen, hood, mix(s.inner, BLACK, 0.15))
        front = path((-40, 388), ((-26, 412), (26, 412), (40, 388)), ((124, 760),), ((-124, 760),))
        self._shaded(pen, front, s.inner, light=0.2, dark=0.35)
        pen.stroke(path((-40, 390), ((-24, 416), (24, 416), (40, 390))), mix(s.inner, BLACK, 0.45), 4.0)
        for x in (-15, 15):  # drawstrings with metal tips
            pen.stroke([(x, 412), (x * 1.15, 488)], OUTLINE, 5.4)
            pen.stroke([(x, 412), (x * 1.15, 488)], (236, 240, 246), 3.2)
            pen.fill([(x * 1.15 - 3.4, 486), (x * 1.15 + 3.4, 486), (x * 1.15 + 3.4, 500), (x * 1.15 - 3.4, 500)],
                     WHITE, SILVER, None, OUTLINE, 2.0)  # fmt: skip
        # a silver chain over the hoodie
        for t in np.linspace(0.06, 0.94, 11):
            x = -40 + 80 * t
            y = 404 + 46 * math.sin(math.pi * t)
            pen.shape(ellipse_points(x, y, 4.6, 3.2, n=16), SILVER, OUTLINE, 1.8)
        for side in (-1, 1):
            panel = path((side * 34, 392), ((side * 44, 378), (side * 62, 372), (side * 100, 386)),
                         ((side * 142, 400), (side * 152, 436), (side * 154, 480)),
                         ((side * 164, 560), (side * 168, 660), (side * 170, 760)), ((side * 62, 760),),
                         ((side * 62, 640), (side * 60, 520), (side * 52, 462)), ((side * 36, 400),))  # fmt: skip
            self._shaded(pen, panel, s.jacket, light=0.2, dark=0.4, width=3.8)
            lapel = [(side * 36, 398), (side * 82, 446), (side * 60, 476), (side * 50, 440)]
            self._shaded(pen, lapel, mix(s.jacket, WHITE, 0.1), rim=False, width=2.8)
            pen.stroke(path((side * 96, 404), ((side * 112, 406), (side * 128, 414), (side * 138, 430))),
                       mix(s.jacket, WHITE, 0.35), 3.0)  # a shine on the shoulder  # fmt: skip
            pen.stroke([(side * 150, 520), (side * 148, 700)], mix(s.jacket, WHITE, 0.18), 2.6)
        # the asymmetric zip on one panel and a zipped pocket on the other
        pen.stroke([(-60, 472), (-98, 760)], OUTLINE, 5.0)
        pen.stroke([(-60, 472), (-98, 760)], SILVER, 2.6)
        pen.fill([(-72, 540), (-62, 541), (-64, 566), (-74, 565)], WHITE, SILVER, None, OUTLINE, 2.0)
        pen.stroke([(84, 616), (128, 574)], OUTLINE, 4.2)
        pen.stroke([(84, 616), (128, 574)], SILVER, 2.0)

    # arms and hands

    def _limb(self, pen: Pen, a, b, width_a: float, width_b: float, colour: Colour, rounded: bool = True,
              rim: bool = True) -> None:  # fmt: skip
        d = (b - a) / max(1e-6, float(np.linalg.norm(b - a)))
        n = np.array([-d[1], d[0]])
        angle = math.atan2(d[1], d[0])
        if rounded:
            points = ellipse_points(a[0], a[1], width_a, width_a, 14, angle + math.pi / 2, angle + 3 * math.pi / 2)
            points += ellipse_points(b[0], b[1], width_b, width_b, 14, angle - math.pi / 2, angle + math.pi / 2)
        else:
            points = [tuple(a + n * width_a), tuple(a - n * width_a), tuple(b - n * width_b), tuple(b + n * width_b)]
        self._shaded(pen, points, colour, rim=rim, width=3.4)

    def _sleeve(self, pen: Pen, shoulder, elbow, wrist) -> None:
        s = self.style
        puffy = s.hair == "bun"
        w0, w1, w2 = (33.0, 29.0, 25.0) if puffy else (29.0, 25.0, 21.0)
        self._limb(pen, shoulder, elbow, w0, w1, s.jacket)
        self._limb(pen, elbow, wrist, w1, w2, s.jacket)
        for a, b, w in ((shoulder, elbow, w0), (elbow, wrist, w1)):
            d = (b - a) / max(1e-6, float(np.linalg.norm(b - a)))
            n = np.array([-d[1], d[0]])
            mid = a + (b - a) * 0.55
            if puffy:  # a quilted channel across the sleeve
                pen.stroke([tuple(mid + n * (w - 4)), tuple(mid - n * (w - 4))], mix(s.jacket, BLACK, 0.45), 2.4)
            else:  # a leather crease
                pen.stroke([tuple(mid + n * (w * 0.5) - d * 6), tuple(mid - n * (w * 0.1))], mix(s.jacket, WHITE, 0.3), 2.2)
        d = (wrist - elbow) / FOREARM
        cuff_start = wrist - d * 16
        self._limb(pen, cuff_start, wrist + d * 3, w2 - 3, w2 - 4, s.cuff, rounded=False, rim=False)
        normal = np.array([-d[1], d[0]])
        centre = cuff_start + d * 9
        for offset in (-9.0, 0.0, 9.0):  # the small rhombus motif
            x, y = centre + normal * offset
            with pen.transformed(rotation(math.degrees(math.atan2(d[1], d[0])) - 90, (x, y))):
                pen.shape([(x, y - 4.4), (x + 3.4, y), (x, y + 4.4), (x - 3.4, y)], s.ornament, None)

    def _dance_arm(self, pen: Pen, side: int, upper_deg: float, fore_deg: float, hand: str) -> None:
        shoulder = np.array([side * SHOULDER[0], SHOULDER[1]])

        def direction(deg):
            a = math.radians(deg)
            return np.array([side * math.sin(a), math.cos(a)])

        elbow = shoulder + UPPER_ARM * direction(upper_deg)
        wrist = elbow + FOREARM * direction(fore_deg)
        self._sleeve(pen, shoulder, elbow, wrist)
        d = direction(fore_deg)
        self._hand(pen, wrist + d * 12, d, side, hand)

    def _hand(self, pen: Pen, centre, d, side: int, hand: str) -> None:
        s = self.style
        angle = math.degrees(math.atan2(d[1], d[0])) - 90  # the hand's local y runs along the forearm
        x, y = centre
        top, bottom = mix(s.skin, WHITE, 0.1), mix(s.skin, s.skin_shadow, 0.8)
        with pen.transformed(translation(x, y) @ rotation(angle)):
            if hand == "open":
                # a clearly open palm: long fingers fanned apart and the thumb out
                for spread, length in ((-30, 40), (-10, 46), (10, 46), (30, 40)):
                    with pen.transformed(rotation(spread, (0, 10))):
                        finger = path((-5, 10), ((-5, 10 + length),), ((-5, 16 + length), (5, 16 + length), (5, 10 + length)), ((5, 10),))
                        pen.fill(finger, top, bottom, None, OUTLINE, 3.0)
                pen.fill(ellipse_points(0, 8, 19, 17), top, bottom, None, OUTLINE, 3.4)
                with pen.transformed(rotation(-side * 60, (side * 14, 4))):
                    pen.fill(ellipse_points(side * 14, 20, 6, 15), top, bottom, None, OUTLINE, 3.0)
            else:  # a loose fist
                pen.fill(ellipse_points(0, 8, 19, 17), top, bottom, None, OUTLINE, 3.4)
                for fx in (-8, 0, 8):
                    pen.stroke([(fx, 15), (fx, 22)], s.skin_shadow, 2.6)

    def _mic_arm(self, pen: Pen, pose: Pose, head: np.ndarray) -> None:
        """The microphone below and to the side of the mouth, following it; lowered after a pause."""
        s = self.style
        side = s.mic_side
        mouth = (head @ np.array([0.0, self.HEAD[1] + MOUTH_Y, 1.0]))[:2]
        lowered = 1.0 - smoothstep(pose.sing)
        grille = mouth + np.array([side * 62.0 + side * lowered * 10, 50.0 + lowered * 78 - pose.push * 6])
        towards = (mouth - grille) / max(1e-6, float(np.linalg.norm(mouth - grille)))
        axis = 0.4 * towards + 0.6 * np.array([0.0, -1.0])
        axis /= np.linalg.norm(axis)
        grip = grille - axis * MIC_GRILLE
        shoulder = np.array([side * SHOULDER[0], SHOULDER[1]])
        elbow, wrist = two_bone(shoulder, grip, side)
        for _ in range(2):  # the fist, not the wrist, sits on the grip
            d = (wrist - elbow) / max(1e-6, float(np.linalg.norm(wrist - elbow)))
            elbow, wrist = two_bone(shoulder, grip - d * 14, side)
        self._sleeve(pen, shoulder, elbow, wrist)
        self._microphone(pen, grip, axis, side)

    def _microphone(self, pen: Pen, grip, axis, side: int) -> None:
        s = self.style
        angle = math.degrees(math.atan2(axis[0], -axis[1]))  # local -y runs along the microphone
        body = (34, 34, 42)
        with pen.transformed(translation(*grip) @ rotation(angle)):
            handle = path((-6, 34), ((-6, 42), (6, 42), (6, 34)), ((9.5, -34),), ((-9.5, -34),))
            pen.fill(handle, mix(body, WHITE, 0.25), mix(body, BLACK, 0.5), mix(s.rim, WHITE, 0.2), OUTLINE, 3.0)
            pen.fill([(-11, -33), (11, -33), (11, -42), (-11, -42)], mix(s.accent, WHITE, 0.2), mix(s.accent, BLACK, 0.2), None, OUTLINE, 2.4)
            grille = ellipse_points(0, -MIC_GRILLE, 17, 18.5, n=36)
            pen.fill(grille, (196, 202, 214), (84, 90, 104), mix(s.rim, WHITE, 0.3), OUTLINE, 3.2)
            with pen.clipped(grille) as mesh:
                for k in range(-4, 5):
                    mesh.stroke([(k * 6 - 20, -MIC_GRILLE - 20), (k * 6 + 20, -MIC_GRILLE + 20)], (70, 76, 90), 1.2, caps=False)
            pen.stroke(ellipse_points(0, -MIC_GRILLE, 11, 12, n=12, start=math.radians(200), end=math.radians(250)), WHITE, 2.6)
            # the fist around the handle
            top, bottom = mix(s.skin, WHITE, 0.1), mix(s.skin, s.skin_shadow, 0.8)
            pen.fill(ellipse_points(0, 2, 17, 19), top, bottom, None, OUTLINE, 3.4)
            for fy in (-6, 2, 10):
                pen.stroke([(-side * 15, fy), (-side * 4, fy)], s.skin_shadow, 2.4)
            pen.fill(ellipse_points(side * 7, -13, 9, 6), top, bottom, None, OUTLINE, 2.6)

    # head

    def _face_outline(self) -> list[tuple[float, float]]:
        w, jaw = self.style.face_width, self.style.jaw
        if jaw > 0.5:  # angular: a defined jaw corner and a squarer chin
            right = path((0, -112), ((w * 0.62, -112), (w, -78), (w, -14)), ((w, 22), (w * 0.94, 50), (w * 0.8, 72)),
                         ((w * 0.66, 90), (w * 0.46, 108), (w * 0.3, 113)), ((w * 0.14, 116), (0, 116), (0, 116)))  # fmt: skip
        else:  # soft: high cheekbones narrowing to a small chin
            right = path((0, -112), ((w * 0.62, -112), (w, -78), (w, -16)), ((w, 32), (w * 0.8, 74), (w * 0.44, 100)),
                         ((w * 0.26, 112), (w * 0.08, 114), (0, 114)))  # fmt: skip
        return mirrored(right[::-1])[:-1] + right[1:]

    def _head(self, pen: Pen, pose: Pose) -> None:
        s = self.style
        face = pose.face
        with pen.transformed(translation(0, self.HEAD[1])):
            outline = self._face_outline()
            pen.fill(outline, mix(s.skin, WHITE, 0.14), mix(s.skin, s.skin_shadow, 0.35), mix(mix(s.skin, WHITE, 0.4), s.rim, 0.35), OUTLINE, 3.8)
            with pen.clipped(outline) as inner:
                if s.jaw > 0.5:  # a light stubble shadow along the jaw
                    band = path((-84, 40), ((-80, 80), (-40, 118), (0, 118)), ((40, 118), (80, 80), (84, 40)),
                                ((70, 70), (40, 98), (0, 98)), ((-40, 98), (-70, 70), (-84, 40)))  # fmt: skip
                    inner.shape(band, mix(s.skin, s.hair_dark, 0.12), None)
                else:  # contour under the cheekbones
                    for side in (-1, 1):
                        inner.shape(ellipse_points(side * 80, 58, 18, 36, n=24), mix(s.skin, s.skin_shadow, 0.35), None)
            if s.blush > 0:
                for side in (-1, 1):
                    pen.shape(ellipse_points(side * 54, 44, 15, 8, n=24), mix(s.skin, (236, 140, 170), 0.3 * s.blush + 0.1), None)
            self._nose(pen)
            self._eyes(pen, face, pose.blink)
            self._brows(pen, face)
            self._mouth(pen, face, pose.mouth)
            self._hair_front(pen, pose)
            self._headphones(pen)

    def _nose(self, pen: Pen) -> None:
        s = self.style
        if s.jaw > 0.5:
            pen.stroke(path((-8, -8), ((-4, 12), (-2, 26), (4, 36))), mix(s.skin, s.skin_shadow, 0.8), 3.2)
            pen.stroke(path((-8, 40), ((-2, 46), (6, 46), (12, 40))), s.skin_shadow, 3.6)
        else:
            pen.stroke(path((4, 26), ((8, 34), (8, 40), (-2, 43))), s.skin_shadow, 3.6)
            pen.dot(-1, 30, 2.4, mix(s.skin, WHITE, 0.6))

    def _eyes(self, pen: Pen, face, blink: float) -> None:
        s = self.style
        raise_, inner, asym, opening, squint, smile, gx, gy, tilt = face
        o = opening * (1 - blink)
        ew, top, bottom, lift_amount = (22.0, 16.0, 11.0, 4.0) if s.lashes else (20.0, 13.5, 10.0, 1.5)
        liner = 6.0 if s.lashes else 5.4
        for side in (-1, 1):
            cx, cy = side * 39.0, 8.0
            xs = np.linspace(-ew, ew, 22)
            profile = np.sin(np.pi * (xs + ew) / (2 * ew))
            lift = side * xs / ew * lift_amount  # outer corners lifted: a confident almond shape
            if s.lashes:  # soft eyeshadow above the lid
                lid = [(cx + x, cy - top * p**0.75 - li - 12 * p) for x, p, li in zip(xs, profile, lift)]
                base = [(cx + x, cy - top * p**0.75 - li) for x, p, li in zip(xs, profile, lift)][::-1]
                pen.shape(lid + base, mix(s.skin, (180, 110, 170), 0.22), None)
            else:  # a crease above the lid
                crease = [(cx + x, cy - top * p**0.75 - li - 7) for x, p, li in zip(xs[4:-3], profile[4:-3], lift[4:-3])]
                pen.stroke(crease, s.skin_shadow, 2.4)
            if o < 0.14:  # shut: a lash line curving down
                line = path((cx - ew, cy - side * 0), ((cx - ew * 0.5, cy + 7), (cx + ew * 0.5, cy + 7), (cx + ew, cy)))
                pen.stroke(line, OUTLINE, liner)
                if s.lashes:
                    end = (cx + side * ew, cy)
                    pen.stroke([end, (end[0] + side * 7, end[1] + 4)], OUTLINE, 3.6)
                continue
            upper = [(cx + x, cy - top * o * p**0.75 - li) for x, p, li in zip(xs, profile, lift)]
            lower = [(cx + x, cy + bottom * min(o, 1.0) * p**0.9 - li) for x, p, li in zip(xs, profile, lift)][::-1]
            eye = upper + lower
            with pen.clipped(eye) as ep:
                ep.fill(eye, SCLERA, (218, 222, 236), None, None)
                ix, iy = cx + gx * 7, cy + gy * 5 + 1
                ep.fill(ellipse_points(ix, iy, 13.0, 13.0, n=28), mix(s.iris, BLACK, 0.45), mix(s.iris, WHITE, 0.25), None, mix(s.iris, BLACK, 0.6), 1.8)
                ep.dot(ix, iy, 6.2, (16, 14, 22))
                ep.dot(ix - 4.6, iy - 5.0, 3.8, WHITE)
                ep.dot(ix + 4.4, iy + 4.4, 1.7, WHITE)
                if squint > 0.02:  # the cheeks push up: smiling eyes
                    edge = [(cx + x, cy + bottom - squint * (top + bottom) * 0.8 * p) for x, p in zip(xs, profile)]
                    ep.shape(edge + [(cx + ew, cy + 30), (cx - ew, cy + 30)], s.skin, None)
            pen.stroke(lower[3:-3], s.skin_shadow, 2.4)
            pen.stroke(upper, OUTLINE, liner)
            if s.lashes:  # a winged liner and three lashes at the outer corner
                end = upper[-1] if side > 0 else upper[0]
                wing = [end, (end[0] + side * 11, end[1] - 8), (end[0] - side * 6, end[1] + 1.5)]
                pen.shape(wing, OUTLINE, None)
                for k, i in enumerate((14, 17, 20) if side > 0 else (7, 4, 1)):
                    px, py = upper[i]
                    pen.stroke([(px, py), (px + side * (3 + k * 1.5), py - 6 - k)], OUTLINE, 2.4)

    def _brows(self, pen: Pen, face) -> None:
        s = self.style
        raise_, inner, asym, opening, squint, smile, gx, gy, tilt = face
        for side in (-1, 1):
            lift = raise_ + (asym + s.brow_attitude if side > 0 else 0)
            y = -24 - lift
            if s.lashes:  # arched and tapered
                centre = path((side * 20, y - inner + 3), ((side * 34, y - 8 - inner * 0.4), (side * 50, y - 11), (side * 62, y + 1)), n=14)
                pen.fill(tapered(centre, 8.0, 2.6), mix(s.brow, WHITE, 0.1), s.brow, None, None)
            else:  # straight, thick, a slight lift at the tail
                centre = path((side * 18, y - inner + 2), ((side * 32, y - 4 - inner * 0.4), (side * 48, y - 7), (side * 62, y - 2)), n=14)
                pen.fill(tapered(centre, 11.0, 5.0), mix(s.brow, WHITE, 0.1), s.brow, None, None)

    def _mouth(self, pen: Pen, face, mouth) -> None:
        s = self.style
        smile = face[5]
        w, h, roundness, teeth = mouth
        cy = MOUTH_Y
        if h < 2.0:
            self._closed_mouth(pen, w * 0.95, smile, cy)
            return
        corner = -smile * 6.0
        n = 24
        sx = np.linspace(0, 1, n)
        ell = np.sqrt(np.clip(1 - (2 * sx - 1) ** 2, 0, 1))
        soft = np.sin(np.pi * sx)
        lift = corner * (1 - soft) * (1 - roundness) * (1 + s.smirk * (2 * sx - 1))
        xs = -w + 2 * w * sx
        top = cy - (roundness * h * 0.5 * ell + (1 - roundness) * h * 0.1 * soft) + lift
        bottom = cy + (roundness * h * 0.5 * ell + (1 - roundness) * h * 0.9 * soft) + lift
        shape = list(zip(xs, top)) + list(zip(xs[::-1], bottom[::-1]))
        with pen.clipped(shape) as inner:
            inner.shape(shape, MOUTH_INSIDE, None)
            if teeth > 0.05:
                band = min(h * 0.3, 7.5) * teeth
                inner.shape([(x, y + band) for x, y in zip(xs, top)][::-1] + [(-w - 2, cy - h), (w + 2, cy - h)][::-1], TEETH, None)
            inner.shape(ellipse_points(0, float(bottom.max()) + 1, w * 0.62, max(4.0, h * 0.34), n=24), TONGUE, None)
        lip_width = 6.4 if s.lashes else 4.2
        pen.stroke(shape + shape[:2], s.lips, lip_width, caps=False)
        pen.stroke(shape + shape[:2], mix(s.lips, OUTLINE, 0.6), 1.8, caps=False)
        if s.lashes:  # a gloss highlight on the lower lip
            low = float(bottom.max())
            pen.stroke(path((-7, low + 3.4), ((-3, low + 4.6), (3, low + 4.6), (6, low + 3.4))), mix(s.lips, WHITE, 0.55), 2.0)

    def _closed_mouth(self, pen: Pen, w: float, smile: float, cy: float) -> None:
        s = self.style
        xs = np.linspace(-w, w, 25)
        u = xs / w
        corner = -smile * 6.0 * (1 + s.smirk * np.sign(u))
        line = cy + smile * 4.0 * (1 - u**2) + corner * u**2
        if s.lashes:  # full lips: an upper lip with a soft bow, a fuller lower lip, a gloss highlight
            upper = line - 5.4 * (1 - u**2) ** 0.6 * (1 - 0.35 * np.exp(-((xs / 4.5) ** 2)))
            lower = line + 7.6 * (1 - u**2) ** 0.7
            lips = list(zip(xs, upper)) + list(zip(xs[::-1], lower[::-1]))
            pen.fill(lips, mix(s.lips, WHITE, 0.12), mix(s.lips, BLACK, 0.15), None, mix(s.lips, OUTLINE, 0.5), 1.8)
            pen.stroke(list(zip(xs, line)), mix(s.lips, OUTLINE, 0.6), 2.6)
            pen.stroke(path((-6, float(line[12]) + 4), ((-2, float(line[12]) + 5.4), (2, float(line[12]) + 5.4), (5, float(line[12]) + 4))),
                       mix(s.lips, WHITE, 0.55), 2.0)  # fmt: skip
        else:
            pen.stroke(list(zip(xs, line)), mix(s.lips, OUTLINE, 0.55), 4.4)
            pen.stroke(path((-9, float(line[12]) + 9), ((-3, float(line[12]) + 11), (3, float(line[12]) + 11), (9, float(line[12]) + 9))),
                       s.skin_shadow, 2.6)  # fmt: skip

    # hair and headphones

    def _hair_back(self, pen: Pen, pose: Pose) -> None:
        s = self.style
        with pen.transformed(translation(0, self.HEAD[1])):
            back = path((-94, 0), ((-104, -86), (-60, -130), (0, -130)), ((60, -130), (104, -86), (94, 0)), ((0, -30),))
            self._hair_fill(pen, back, dark=True)
            if s.hair == "bun":  # a sleek high bun, swaying a little with the spring
                with pen.transformed(rotation(pose.hair_swing * 0.5, (4, -126))):
                    bun = ellipse_points(4, -154, 40, 34, n=40)
                    self._hair_fill(pen, bun)
                    pen.stroke(ellipse_points(4, -154, 26, 20, n=24, start=math.radians(160), end=math.radians(400)), s.hair_dark, 2.8)
                    pen.stroke(ellipse_points(4, -154, 34, 28, n=18, start=math.radians(200), end=math.radians(290)), s.hair_light, 3.2)
                    pen.stroke(ellipse_points(4, -154, 14, 10, n=14, start=math.radians(220), end=math.radians(420)), s.hair_light, 2.4)

    def _hair_fill(self, pen: Pen, points, dark: bool = False, width: float = 3.6) -> None:
        s = self.style
        base = s.hair_dark if dark else s.hair_colour
        top = mix(base, s.hair_light, 0.15 if dark else 0.35)
        bottom = mix(base, s.hair_dark, 0.6)
        rim = mix(mix(s.hair_colour, WHITE, 0.4), s.rim, 0.55)
        pen.fill(points, top, bottom, rim, OUTLINE, width)

    def _hair_front(self, pen: Pen, pose: Pose) -> None:
        s = self.style
        swing = pose.hair_swing
        if s.hair == "bun":
            # hair pulled up sleekly, with shine lines running to the bun
            cap = path((-92, -8), ((-100, -88), (-58, -128), (0, -128)), ((58, -128), (100, -88), (92, -8)),
                       ((82, -44), (58, -84), (0, -92)), ((-58, -84), (-82, -44), (-92, -8)))  # fmt: skip
            self._hair_fill(pen, cap)
            for side in (-1, 1):
                pen.stroke(path((side * 30, -112), ((side * 22, -118), (side * 14, -124), (side * 8, -128))), s.hair_light, 3.0)
                pen.stroke(path((side * 70, -76), ((side * 64, -100), (side * 44, -118), (side * 20, -124))), s.hair_dark, 2.4)
                # curtain bangs from the centre part to the cheekbones, layered with a highlight
                bang = path((side * 2, -98), ((side * 30, -96), (side * 66, -70), (side * 80, -18)),
                            ((side * 70, -40), (side * 48, -68), (side * 8, -80)))  # fmt: skip
                self._hair_fill(pen, bang, width=3.2)
                pen.stroke(path((side * 18, -90), ((side * 40, -86), (side * 60, -66), (side * 70, -38))), s.hair_light, 2.6)
                # a wavy tendril framing the face, moving with the spring
                sw = swing * 0.7
                centre = path((side * 74, -36), ((side * 90, -6), (side * 66 + sw * 0.4, 26), (side * 80 + sw, 70)), n=20)
                self._hair_fill(pen, tapered(centre, 15.0, 3.0), width=2.8)
        else:
            # a mid fade: the sides shaded from the hair colour into the skin
            for side in (-1, 1):
                fade = [(side * 90, -74), (side * 70, -74), (side * 80, -4), (side * 92, -4)]
                pen.fill(fade, mix(s.hair_colour, s.hair_dark, 0.3), mix(s.hair_colour, s.skin, 0.7), None, None)
            # a textured fringe: volume on top, layered locks falling forward and to one side
            lift = swing * 0.4
            top = path((-94, -58), ((-102, -112), (-64, -152), (-4, -154)), ((58, -156), (102, -126), (96, -58)),
                       ((90, -70), (82, -80), (70, -82)), ((58, -72), (46, -58), (34, -50 + lift)),
                       ((40, -66), (30, -80), (16, -86)), ((2, -74), (-10, -62), (-22, -54 + lift)),
                       ((-16, -72), (-28, -86), (-42, -88)), ((-54, -78), (-64, -68), (-72, -60 + lift)),
                       ((-80, -70), (-88, -66), (-94, -58)))  # fmt: skip
            self._hair_fill(pen, top, width=3.8)
            for x0, tip in ((-44, -62), (0, -56), (40, -52)):  # soft layered locks swept forward, one shine each
                lock = path((x0 + 16, -140), ((x0 + 0, -134), (x0 - 16, -110), (x0 - 20 + lift * 0.5, tip)),
                            ((x0 - 4, -98), (x0 + 8, -122), (x0 + 16, -140)))  # fmt: skip
                pen.fill(lock, mix(s.hair_colour, s.hair_light, 0.3), s.hair_colour, None, s.hair_dark, 2.2)
                pen.stroke(path((x0 + 8, -130), ((x0 + 0, -124), (x0 - 8, -112), (x0 - 12, -98))), mix(s.hair_light, WHITE, 0.1), 2.4)

    def _headphones(self, pen: Pen) -> None:
        s = self.style
        k = s.headphone_size
        colour = s.headphones
        crown = -138 - 6 * k
        band = path((-(s.face_width + 6), -18), ((-(s.face_width + 14), -104), (-54, crown), (0, crown)),
                    ((54, crown), (s.face_width + 14, -104), (s.face_width + 6, -18)))  # fmt: skip
        pen.stroke(band, OUTLINE, 10.0 * k + 4)
        pen.stroke(band, mix(colour, WHITE, 0.15), 10.0 * k)
        pen.stroke(band[3:-3], mix(colour, BLACK, 0.3), 3.0 * k, caps=False)
        for side in (-1, 1):
            cx, cy = side * (s.face_width + 10 * k), 10.0
            cup_w, cup_h = 15 * k, 26 * k
            cup = path((cx - cup_w, cy - cup_h + 10), ((cx - cup_w, cy - cup_h - 2), (cx + cup_w, cy - cup_h - 2), (cx + cup_w, cy - cup_h + 10)),
                       ((cx + cup_w, cy + cup_h - 10),), ((cx + cup_w, cy + cup_h + 2), (cx - cup_w, cy + cup_h + 2), (cx - cup_w, cy + cup_h - 10)))  # fmt: skip
            self._shaded(pen, cup, colour, light=0.2, dark=0.35, width=3.6)
            ring = ellipse_points(cx + side * 3, cy, cup_w * 0.62, cup_h * 0.62, n=28)
            pen.stroke(ring + ring[:2], s.accent, 3.2, caps=False)
            # the rhombus motif on the cup
            r = 5.0 * k
            pen.shape([(cx + side * 3, cy - r), (cx + side * 3 + r * 0.75, cy), (cx + side * 3, cy + r), (cx + side * 3 - r * 0.75, cy)],
                      s.accent if s.hair == "fringe" else s.ornament, None)  # fmt: skip


# --- design sheet ----------------------------------------------------------------------------------

SHEET_BACKGROUNDS = (("dark", (14, 15, 24)), ("light", (238, 240, 244)))


def design_sheet(style: SingerStyle) -> Image.Image:
    """Front pose, the same pose at 300 px, four dance poses, five mouth shapes, a blink and three
    expressions, on a dark and on a light background."""
    from humanoid_companion.face.render import caption_font

    big = SingerRenderer(style, 540, 720, supersample=3)
    small = SingerRenderer(style, 300, 400, supersample=3)
    dance = SingerRenderer(style, 270, 360, supersample=3)
    head_box = big.head_box()
    mouths = ("open", "wide", "o", "open")
    tiles_top = [("front", big.still("happy"), 540), ("at 300 px", small.still("happy"), 300)]
    tiles_top += [(f"dance: {name}", dance.still("happy", mouth_shape=mouths[i], beats=2 * i + 1.5), 270)
                  for i, (name, _) in enumerate(DANCE_POSES)]  # fmt: skip
    heads = [(f"mouth: {shape}", big.still("neutral", mouth_shape=shape)) for shape in MOUTH_SHAPES]
    heads.append(("blink", big.still("neutral", blink=1.0)))
    heads += [(expression, big.still(expression)) for expression in ("happy", "surprised", "sad")]
    font, title_font = caption_font(22), caption_font(34)
    width, head_size, label = 1920, 1920 // 9, 36
    section = 720 + label + head_size + label
    sheet = Image.new("RGB", (width, 80 + section * 2), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 22), f"{style.name} / {style.key}: design sheet", fill=(20, 20, 30), font=title_font)
    for row, (name, colour) in enumerate(SHEET_BACKGROUNDS):
        top = 80 + row * section
        text = (230, 230, 240) if name == "dark" else (40, 40, 52)
        draw.rectangle([0, top, width, top + section], fill=colour)
        x = 0
        for caption, rgba, tile_width in tiles_top:
            tile = Image.fromarray(rgba)
            sheet.paste(tile, (x, top + 720 - tile.height), tile)
            draw.text((x + 12, top + 724), caption, fill=text, font=font)
            x += tile_width
        y = top + 720 + label
        for i, (caption, rgba) in enumerate(heads):
            tile = Image.fromarray(rgba).crop(head_box).resize((head_size, head_size), Image.Resampling.LANCZOS)
            sheet.paste(tile, (i * head_size, y), tile)
            draw.text((i * head_size + 12, y + head_size + 4), caption, fill=text, font=font)
    return sheet


def showcase(fps: int = 10, bpm: float = 120.0, size: tuple[int, int] = (240, 300)) -> list[Image.Image]:
    """Both singers side by side on a night-dark background, singing and dancing through one full cycle
    of the dance poses (eight beats), as frames for a looping GIF. The singing is a made-up pattern of
    vowels, not a song."""
    width, height = size
    beat = 60.0 / bpm
    loop = int(round(2 * len(DANCE_POSES) * beat * fps))
    renderers = [SingerRenderer(style, width, height, fps=fps, seed=3) for style in SINGERS.values()]
    vowels = ("a", "o", "e", "a", "o", "e", "a", None)

    def signals(t: float, offset: float) -> tuple[float, str | None]:
        step = int(((t + offset) / (beat / 2)) % len(vowels))
        vowel = vowels[step]
        inside = ((t + offset) / (beat / 2)) % 1.0 < 0.75
        return (0.85 if vowel and inside else 0.0), vowel

    frames = []
    for i in range(loop * 2):  # one cycle to settle the smoothing, then the cycle that is kept
        t = i / fps
        tiles = []
        for n, renderer in enumerate(renderers):
            mouth, vowel = signals(t, n * beat / 2)
            tiles.append(renderer.frame("happy", mouth, 0.9, ((t / beat) % 1.0), vowel))
        if i >= loop:
            canvas = Image.new("RGBA", (width * 2, height), (16, 16, 24, 255))
            for n, tile in enumerate(tiles):
                canvas.alpha_composite(Image.fromarray(tile), (n * width, 0))
            frames.append(canvas.convert("RGB"))
    return frames


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Write a design sheet PNG per singer, and singers.gif of both.")
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for key, style in SINGERS.items():
        path = args.out / f"{key}-design.png"
        design_sheet(style).save(path, optimize=True)
        print(f"wrote {path}")
    frames = showcase()
    palette = frames[len(frames) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)  # one palette: no flicker
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    gif = args.out / "singers.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
