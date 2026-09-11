import numpy as np

from humanoid_companion.contact_sheet import sheet


def test_sheet_is_a_grid_of_evenly_spaced_downscaled_frames():
    frames = np.stack([np.full((8, 10, 3), i, np.uint8) for i in range(15)])  # frame i has value i
    s = sheet(frames, n=8, cols=4, scale=2)
    assert s.shape == (2 * 4, 4 * 5, 3)
    firsts = [s[r * 4, c * 5, 0] for r in range(2) for c in range(4)]
    assert firsts == [0, 2, 4, 6, 8, 10, 12, 14]
