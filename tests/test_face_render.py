import numpy as np

from humanoid_companion.face.render import FaceRenderer, mouth_track
from humanoid_companion.face.server import EXPRESSIONS


def settled(expression, mouth=0.0, caption="", frames=40):
    r = FaceRenderer(320, 240, seed=1)
    img = None
    for _ in range(frames):
        img = r.frame(expression, mouth, caption)
    return img


def bright(img, box):
    x0, y0, x1, y1 = box
    return int((img[y0:y1, x0:x1].mean(axis=2) > 120).sum())


def test_every_expression_renders_a_distinct_face():
    faces = {e: settled(e) for e in EXPRESSIONS}
    assert all(f.shape == (240, 320, 3) and f.dtype == np.uint8 for f in faces.values())
    names = list(faces)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.abs(faces[a].astype(int) - faces[b].astype(int)).mean() > 0.5, (a, b)


def test_the_mouth_opens_with_loudness():
    mouth_box = (100, 150, 220, 195)   # around (W/2, 0.7 H) at 320x240
    assert bright(settled("neutral", mouth=0.9), mouth_box) > 2 * bright(settled("neutral", mouth=0.0), mouth_box)


def test_captions_are_drawn_at_the_bottom():
    bottom = (0, 200, 320, 240)
    assert bright(settled("neutral", caption="Hello there, I am your humanoid."), bottom) > bright(settled("neutral"), bottom)


def test_cyrillic_captions_get_real_glyphs_not_boxes():
    from humanoid_companion.face.render import caption_font

    f = caption_font(20)
    widths = {ch: f.getlength(ch) for ch in "ДзякуйЎ"}
    assert len(set(widths.values())) > 2, widths   # placeholder boxes all have one width


def test_mouth_track_follows_the_audio():
    rate, fps = 24000, 25
    silence, tone = np.zeros(rate), 0.3 * np.sin(np.linspace(0, 2 * np.pi * 200, rate))
    track = mouth_track(np.concatenate([silence, tone]).astype(np.float32), rate, fps, 2 * fps)
    assert track.shape == (50,) and track[:25].max() == 0.0 and track[30:].min() > 0.8
