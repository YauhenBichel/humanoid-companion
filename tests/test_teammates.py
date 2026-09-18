import numpy as np

from humanoid_companion.character import CharacterRenderer
from humanoid_companion.conversation import persona
from humanoid_companion.teammates import BYTE, TEAMMATES, TEMPO


def test_the_two_teammates_look_different():
    assert set(TEAMMATES) == {"byte", "tempo", "alesia", "maks"}
    assert BYTE.look != TEMPO.look and BYTE.trim != TEMPO.trim and BYTE.accessory != TEMPO.accessory


def test_a_teammate_persona_names_it_and_adds_its_role_to_the_companion():
    byte = BYTE.persona("Alex")
    assert byte.startswith("You are Byte, a humanoid teammate:") and "computer science" in byte
    assert "call Alex by name" in byte and "owner" not in byte.replace("never say 'my owner'", "")
    tempo = TEMPO.persona("")
    assert "singer" in tempo and "copyrighted" in tempo and "Answer only with the JSON object" in tempo
    assert persona("Alex").startswith("You are the humanoid:")  # the plain companion is unchanged


def rendered(teammate, frames=30, **frame_args):
    renderer = CharacterRenderer(teammate, 270, 360, fps=30, seed=2)
    image = None
    for _ in range(frames):
        image = renderer.frame(**frame_args)
    return image


def test_a_character_is_a_bust_on_a_transparent_background():
    image = rendered(BYTE, expression="neutral")
    assert image.shape == (360, 270, 4) and image.dtype == np.uint8
    assert image[0, 0, 3] == 0 and image[0, -1, 3] == 0  # corners stay clear for the video underneath
    assert image[150, 135, 3] == 255  # the head's screen is solid
    assert image[-1, 135, 3] == 255  # the shoulders reach the bottom edge


def test_each_teammate_has_its_own_accessory():
    byte, tempo = rendered(BYTE, expression="neutral"), rendered(TEMPO, expression="neutral")
    side = (slice(110, 180), slice(0, 30))  # beside the head: Tempo's headphone cup, nothing for Byte
    assert tempo[side][..., 3].max() == 255 and byte[side][..., 3].max() == 0
    above = (slice(0, 40), slice(125, 145))  # above the head: Byte's antenna
    assert byte[above][..., 3].max() > 0


def top_of_bust(image):
    """The height of the badge on the chest, the only part drawn at alpha 230."""
    return float(np.nonzero(image[..., 3] == 230)[0].mean())


def test_the_singer_bounces_on_the_beat():
    on_beat = rendered(TEMPO, frames=1, expression="happy", energy=1.0, beat_phase=0.0)
    between = rendered(TEMPO, frames=1, expression="happy", energy=1.0, beat_phase=0.5)
    assert top_of_bust(between) < top_of_bust(on_beat)  # lifted halfway through the beat


def test_without_a_beat_the_character_does_not_dance():
    quiet = rendered(TEMPO, frames=1, expression="happy", energy=1.0)
    still = rendered(TEMPO, frames=1, expression="happy", energy=0.0)
    assert top_of_bust(quiet) == top_of_bust(still)
