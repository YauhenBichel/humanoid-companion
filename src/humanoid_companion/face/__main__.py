"""Run the face on its own, to look at it or try expressions and the voice.

    python -m humanoid_companion.face                  # then open http://127.0.0.1:8765 and click once
    python -m humanoid_companion.face --tour           # cycle every expression and say a line (needs a speech server)
    python -m humanoid_companion.face --screenshots face-shots   # one PNG per expression (headless Chrome) + a sheet
"""

import argparse
import time
from pathlib import Path

from humanoid_companion.face.server import EXPRESSIONS, FaceServer
from humanoid_companion.face.snapshot import snapshot


def screenshots(face: FaceServer, out: Path) -> list[Path]:
    """Each expression rendered by headless Chrome from the served page, plus a sheet of all."""
    import mediapy
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    shots = [snapshot(face.url, e, e, out / f"{e}.png") for e in EXPRESSIONS]
    imgs = [mediapy.read_image(p)[..., :3] for p in shots]
    imgs += [np.zeros_like(imgs[0])] * (-len(imgs) % 4)
    rows = [np.concatenate(imgs[i:i + 4], axis=1) for i in range(0, len(imgs), 4)]
    mediapy.write_image(out / "expressions.png", np.concatenate(rows, axis=0))
    return shots


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tour", action="store_true")
    p.add_argument("--screenshots", type=Path, metavar="DIR")
    args = p.parse_args(argv)
    if args.screenshots:
        face = FaceServer(port=0).start()
        shots = screenshots(face, args.screenshots)
        face.stop()
        print(f"wrote {len(shots)} screenshots and {args.screenshots / 'expressions.png'}")
        return
    face = FaceServer(port=args.port).start()
    print(f"face at {face.url}  (open it in a browser and click once so it may play sound; Ctrl-C to stop)")
    try:
        if args.tour:
            while face.viewers == 0:
                time.sleep(0.5)
            time.sleep(3)  # time to click "wake"
            from humanoid_companion.voice import speak

            for e in EXPRESSIONS:
                face.set_expression(e, caption=e)
                time.sleep(2.5)
            uid = face.say(speak("Hello! I am your humanoid. This is my face."), "Hello! I am your humanoid.", "happy")
            face.wait_played(uid, 15)
            face.set_expression("neutral", caption="")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        face.stop()


if __name__ == "__main__":
    main()
