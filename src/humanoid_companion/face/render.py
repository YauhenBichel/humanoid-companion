"""The face drawn frame by frame in Python, for recorded videos (the live face is face.html).

Same expression table (expressions.json) and the same geometry as the page: eyes, mouth, brows,
thinking dots, listening ring, sleep z's, soft glow, caption. The mouth opens with the speech's
loudness per frame (`mouth_track`), like the page's Web Audio analyser. One difference: eye tilt
is not drawn (it only applies to "sad", whose eyes are nearly round; the brows carry it).

Colours come from a Look (face/look.py). With transparent=True the frames are RGBA with nothing but
the face, its glow and the caption, for laying a character over another video.
"""

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from humanoid_companion.face.look import DEFAULT_LOOK, Look
from humanoid_companion.face.server import TABLE

BG, FG, CAPTION = DEFAULT_LOOK.background, DEFAULT_LOOK.glow, DEFAULT_LOOK.caption
KEYS = ("open", "w", "curve", "mw")
# Captions can be Cyrillic (the Belarusian farewell); Pillow's default font has no Cyrillic glyphs.
FONTS = ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf",
         "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def caption_font(size: int):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def mouth_track(samples: np.ndarray, rate: int, fps: float, frames: int) -> np.ndarray:
    """Mouth opening 0..1 per video frame: RMS loudness x 6, clipped, smoothed like the page."""
    out, level = np.zeros(frames), 0.0
    hop = rate / fps
    for i in range(frames):
        chunk = samples[int(i * hop): int((i + 1) * hop)]
        target = min(1.0, float(np.sqrt(np.mean(chunk ** 2))) * 6) if len(chunk) else 0.0
        level += (target - level) * 0.5
        out[i] = level
    return out


class FaceRenderer:
    """Stateful: call frame() once per video frame, in order."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 25.0, seed: int = 0,
                 look: Look = DEFAULT_LOOK, transparent: bool = False, feature_scale: float = 1.0):
        self.W, self.H, self.dt = width, height, 1.0 / fps
        self.look, self.transparent = look, transparent
        self.u = min(width, height) / 10 * feature_scale  # the unit every eye, mouth and brow size is drawn in
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.name = "neutral"
        self.cur = {k: TABLE["neutral"][k] for k in KEYS}
        self.gaze, self.gaze_target = [0.0, 0.0], [0.0, 0.0]
        self.next_blink, self.blink_at = 2.5, -1.0
        self.next_gaze = 0.0
        size = int(max(14, min(28, 0.032 * width)))
        self.font = caption_font(size)
        self.zfont = ImageFont.load_default(size=int(self.u * 0.7))

    def _advance(self, expression: str) -> dict:
        target = TABLE.get(expression, TABLE["neutral"])
        self.name = expression if expression in TABLE else "neutral"
        k = 1 - 0.85 ** (60 * self.dt)          # the page lerps 0.15 per 60 Hz frame
        for key in KEYS:
            self.cur[key] += (target[key] - self.cur[key]) * k
        if self.t >= self.next_blink:
            self.blink_at, self.next_blink = self.t, self.t + 2.5 + self.rng.random() * 3.5
        if self.t >= self.next_gaze:
            self.gaze_target = [(self.rng.random() - 0.5) * 0.4, (self.rng.random() - 0.5) * 0.25]
            self.next_gaze = self.t + 1.5 + self.rng.random() * 2.5
        g = 1 - 0.92 ** (60 * self.dt)
        self.gaze = [self.gaze[i] + (target["look"][i] + self.gaze_target[i] - self.gaze[i]) * g for i in (0, 1)]
        self.t += self.dt
        return target

    def frame(self, expression: str, mouth: float = 0.0, caption: str = "") -> np.ndarray:
        """One frame: RGB, or RGBA when the renderer is transparent."""
        target = self._advance(expression)
        W, H, u, c = self.W, self.H, self.u, self.cur
        fg = self.look.glow
        layer = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        blink = max(0.0, 1 - (self.t - self.blink_at) / 0.15) if self.name != "sleeping" else 0.0
        eye_w, eye_h = u * 1.2 * c["w"], u * 1.6 * c["open"] * (1 - blink)
        for side in (-1, 1):
            x, y = W / 2 + side * u * 2.2 + self.gaze[0] * u, H * 0.42 + self.gaze[1] * u
            if target.get("arc"):
                r = eye_w * 0.55
                d.arc([x - r, y + u * 0.3 - r, x + r, y + u * 0.3 + r], 207, 333, fill=fg, width=int(u * 0.35))
            else:
                h = max(u * 0.08, eye_h)
                d.rounded_rectangle([x - eye_w / 2, y - h / 2, x + eye_w / 2, y + h / 2], radius=min(eye_w, h) / 2, fill=fg)
            if target.get("brows"):
                d.line([(x - side * eye_w * 0.6, y - u * 1.45), (x + side * eye_w * 0.6, y - u * 1.05)], fill=fg,
                       width=int(u * 0.22), joint="curve")
        mx, my, mw = W / 2, H * 0.7, u * 2.2 * c["mw"]
        if mouth > 0.04 or target.get("o"):
            rx, ry = (u * 0.55, u * 0.45) if target.get("o") else (mw / 2, u * (0.15 + 1.3 * mouth) / 2)
            d.ellipse([mx - rx, my - ry, mx + rx, my + ry], fill=fg)
        else:
            pts = [((1 - s) ** 2 * (mx - mw / 2) + 2 * (1 - s) * s * mx + s ** 2 * (mx + mw / 2),
                    (1 - s) ** 2 * my + 2 * (1 - s) * s * (my + c["curve"] * u * 1.2) + s ** 2 * my)
                   for s in np.linspace(0, 1, 24)]
            d.line(pts, fill=fg, width=int(u * 0.28), joint="curve")
            for p in (pts[0], pts[-1]):   # round caps
                r = u * 0.14
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fg)
        glow = layer.filter(ImageFilter.GaussianBlur(u * 0.3))
        img = self._transparent_face(layer, glow) if self.transparent else self._opaque_face(layer, glow)

        d = ImageDraw.Draw(img)
        if target.get("dots"):
            for i in range(3):
                a = 1.0 if int(self.t / 0.3) % 3 == i else 0.3
                cx, cy, r = W / 2 + u * 3.6 + i * u * 0.5, H * 0.2, u * 0.13
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=self._faded(a))
        if target.get("ring"):
            a, r = 0.35 + 0.25 * math.sin(self.t / 0.25), u * 3.7
            d.ellipse([W / 2 - r, H * 0.52 - r, W / 2 + r, H * 0.52 + r], outline=self._faded(a), width=max(1, int(u * 0.08)))
        if target.get("zzz"):
            a = 0.5 + 0.5 * math.sin(self.t / 0.6)
            d.text((W / 2 + u * 3.5, H * 0.25), "z", fill=self._faded(a), font=self.zfont, anchor="ls")
            d.text((W / 2 + u * 4.1, H * 0.17), "z", fill=self._faded(a), font=self.zfont, anchor="ls")
        if caption:
            lines = self._wrap(caption, W * 0.88)
            lh = self.font.size * 1.25
            y0 = H * 0.94 - lh * len(lines)
            # Over another video the caption needs an outline to stay readable on any picture.
            outline = {"stroke_width": max(1, self.font.size // 12), "stroke_fill": (*self.look.background, 255)} if self.transparent else {}
            fill = (*self.look.caption, 255) if self.transparent else self.look.caption
            for i, line in enumerate(lines):
                d.text((W / 2, y0 + i * lh), line, fill=fill, font=self.font, anchor="ma", **outline)
        return np.asarray(img)

    def _opaque_face(self, layer: Image.Image, glow: Image.Image) -> Image.Image:
        background = Image.new("RGB", (self.W, self.H), self.look.background)
        return ImageChops.lighter(ImageChops.add(background, glow), layer)

    def _transparent_face(self, layer: Image.Image, glow: Image.Image) -> Image.Image:
        """The glow colour everywhere, with the drawing and its glow as the alpha channel."""
        ink = np.asarray(ImageChops.lighter(glow, layer)).max(axis=2).astype(np.float32)
        alpha = np.clip(ink * (255.0 / max(self.look.glow)), 0, 255).astype(np.uint8)
        rgba = np.empty((self.H, self.W, 4), np.uint8)
        rgba[..., :3] = self.look.glow
        rgba[..., 3] = alpha
        return Image.fromarray(rgba, "RGBA")

    def _faded(self, amount: float) -> tuple[int, ...]:
        """The glow colour at `amount` (0..1) of full strength: blended into the background, or as alpha."""
        glow, background = self.look.glow, self.look.background
        if self.transparent:
            return (*glow, int(255 * amount))
        return tuple(int(background[i] + (glow[i] - background[i]) * amount) for i in range(3))

    def _wrap(self, text: str, width: float) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if self.font.getlength(trial) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        return lines + ([line] if line else [])
