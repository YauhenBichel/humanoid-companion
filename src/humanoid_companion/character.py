"""A teammate drawn as a character, frame by frame, on a transparent background for video overlays.

The character is a robot bust: shoulders, a neck and a head whose screen shows the companion's face
(humanoid_companion.face.render) in the teammate's colours, with its accessory: an antenna for Byte,
headphones for Tempo. It moves with what it hears:

- speaking: the head nods a little with the voice's loudness;
- singing: the head sways and the whole bust bounces on the beat, stronger when the music is loud.

    renderer = CharacterRenderer(TEAMMATES["tempo"], 540, 720, fps=30)
    rgba = renderer.frame("happy", mouth=0.4, energy=0.8, beat_phase=0.25)

`beat_phase` is the position inside the current beat (0..1); without it the character does not dance.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from humanoid_companion.face.render import FaceRenderer
from humanoid_companion.teammates import Teammate

SWAY_DEGREES = 7.0  # head sway at full energy, singing
NOD_DEGREES = 3.0  # head nod at full loudness, speaking
BOUNCE = 0.025  # bust bounce on the beat, as a fraction of the frame height
FACE_SCALE = 1.35  # eyes and mouth larger than on the full-screen face, so they read in a small overlay


def shade(colour: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """`colour` lightened (amount > 0) towards white or darkened (amount < 0) towards black."""
    if amount >= 0:
        return tuple(int(c + (255 - c) * amount) for c in colour)
    return tuple(int(c * (1 + amount)) for c in colour)


class CharacterRenderer:
    """Stateful like FaceRenderer: call frame() once per video frame, in order. Frames are RGBA."""

    def __init__(self, teammate: Teammate, width: int = 540, height: int = 720, fps: float = 30.0, seed: int = 0):
        self.teammate, self.width, self.height, self.dt = teammate, width, height, 1.0 / fps
        self.t = 0.0
        unit = min(width, height * 0.75)
        self.head_width, self.head_height = int(unit * 0.72), int(unit * 0.56)
        self.bezel = max(4, int(unit * 0.035))
        screen = (self.head_width - 2 * self.bezel, self.head_height - 2 * self.bezel)
        self.face = FaceRenderer(*screen, fps=fps, seed=seed, look=teammate.look, feature_scale=FACE_SCALE)
        self.head_centre = (width / 2, height * 0.40)
        self.nod = 0.0

    def frame(
        self, expression: str, mouth: float = 0.0, energy: float = 0.0, beat_phase: float | None = None
    ) -> np.ndarray:
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        dancing = beat_phase is not None
        bounce = self.height * BOUNCE * energy * abs(math.sin(math.pi * beat_phase)) if dancing else 0.0
        self._draw_shoulders(canvas, lift=bounce)
        if dancing:
            angle = SWAY_DEGREES * energy * math.sin(2 * math.pi * beat_phase / 2)  # one sway per two beats
        else:
            self.nod += (mouth - self.nod) * 0.3  # smoothed, so the head does not jitter with every syllable
            angle = NOD_DEGREES * self.nod * math.sin(2 * math.pi * 1.7 * self.t)
        head = self._head(expression, mouth)
        rotated = head.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        cx, cy = self.head_centre
        canvas.alpha_composite(rotated, (int(cx - rotated.width / 2), int(cy - bounce - rotated.height / 2)))
        self.t += self.dt
        return np.asarray(canvas)

    def _head(self, expression: str, mouth: float) -> Image.Image:
        """The head on its own layer, with room around it for the accessory, centred on the screen."""
        margin = int(self.head_width * 0.22)
        size = (self.head_width + 2 * margin, self.head_height + 2 * margin)
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        left, top = margin, margin
        right, bottom = left + self.head_width, top + self.head_height
        trim = self.teammate.trim
        radius = int(self.head_height * 0.22)
        if self.teammate.accessory == "antenna":
            self._draw_antenna(draw, (left + right) / 2, top)
        draw.rounded_rectangle([left, top, right, bottom], radius=radius, fill=(*trim, 255))
        screen = Image.fromarray(self.face.frame(expression, mouth)).convert("RGBA")
        mask = Image.new("L", screen.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, screen.width - 1, screen.height - 1], radius=max(1, radius - self.bezel), fill=255
        )
        layer.paste(screen, (left + self.bezel, top + self.bezel), mask)
        if self.teammate.accessory == "headphones":
            self._draw_headphones(draw, left, top, right, bottom)
        return layer

    def _draw_antenna(self, draw: ImageDraw.ImageDraw, x: float, top: float) -> None:
        length, ball = self.head_height * 0.2, self.head_height * 0.07
        draw.line([(x, top), (x, top - length)], fill=(*shade(self.teammate.trim, 0.2), 255), width=self.bezel)
        glow = self.teammate.look.glow
        pulse = 0.6 + 0.4 * math.sin(2 * math.pi * 0.8 * self.t)  # a slow blink, like a status light
        draw.ellipse([x - ball, top - length - ball, x + ball, top - length + ball], fill=(*glow, int(255 * pulse)))

    def _draw_headphones(self, draw: ImageDraw.ImageDraw, left: float, top: float, right: float, bottom: float) -> None:
        trim, glow = self.teammate.trim, self.teammate.look.glow
        band = int(self.bezel * 1.6)
        overhang = self.head_width * 0.12
        draw.arc(
            [
                left - overhang * 0.4,
                top - self.head_height * 0.28,
                right + overhang * 0.4,
                bottom - self.head_height * 0.2,
            ],
            start=190,
            end=350,
            fill=(*shade(trim, 0.15), 255),
            width=band,
        )
        cup_width, cup_height = overhang, self.head_height * 0.42
        middle = (top + bottom) / 2
        for side_x in (left - cup_width * 0.55, right + cup_width * 0.55):
            draw.rounded_rectangle(
                [side_x - cup_width / 2, middle - cup_height / 2, side_x + cup_width / 2, middle + cup_height / 2],
                radius=int(cup_width * 0.45),
                fill=(*shade(trim, 0.1), 255),
                outline=(*glow, 255),
                width=max(2, self.bezel // 2),
            )

    def _draw_shoulders(self, canvas: Image.Image, lift: float) -> None:
        draw = ImageDraw.Draw(canvas)
        trim = self.teammate.trim
        cx = self.width / 2
        neck_top = self.head_centre[1] + self.head_height / 2 - self.bezel - lift
        shoulders_top = neck_top + self.head_height * 0.16
        neck_half = self.head_width * 0.09
        draw.rectangle([cx - neck_half, neck_top, cx + neck_half, shoulders_top + 4], fill=(*shade(trim, -0.3), 255))
        half = self.head_width * 0.62
        draw.rounded_rectangle(
            [cx - half, shoulders_top, cx + half, self.height + half],
            radius=int(half * 0.5),
            fill=(*trim, 255),
        )
        light = self.teammate.look.glow
        badge = self.head_width * 0.05
        badge_y = shoulders_top + self.head_height * 0.22
        draw.ellipse([cx - badge, badge_y - badge, cx + badge, badge_y + badge], fill=(*light, 230))
