"""Colours of a face: one Look is shared by the live page (face.html) and the video renderer (render.py).

The default Look is the original companion face: light blue on near-black. A teammate
(humanoid_companion.teammates) brings its own.
"""

from dataclasses import dataclass

Colour = tuple[int, int, int]


def hex_colour(colour: Colour) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


@dataclass(frozen=True)
class Look:
    glow: Colour = (127, 216, 255)  # eyes, mouth, brows and their glow
    background: Colour = (5, 7, 10)  # the screen behind the face
    caption: Colour = (207, 233, 255)

    def css(self) -> dict[str, str]:
        """The same colours as CSS hex strings, for the page."""
        return {
            "glow": hex_colour(self.glow),
            "background": hex_colour(self.background),
            "caption": hex_colour(self.caption),
        }


DEFAULT_LOOK = Look()
