"""A still-frame sheet of a video: evenly spaced frames in a grid, for READMEs and quick review.

    python -m humanoid_companion.contact_sheet demo/video.mp4      # -> frames.png next to it
"""

import argparse
from pathlib import Path

import numpy as np


def sheet(frames: np.ndarray, n: int = 8, cols: int = 4, scale: int = 2) -> np.ndarray:
    """n frames spread over the video, downscaled by `scale`, `cols` per row."""
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    tiles = [frames[i][::scale, ::scale] for i in idx]
    rows = [np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, n, cols)]
    return np.concatenate(rows, axis=0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, help="default: frames.png next to the video")
    args = p.parse_args(argv)

    import mediapy

    out = args.out or args.video.with_name("frames.png")
    mediapy.write_image(out, sheet(mediapy.read_video(args.video)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
