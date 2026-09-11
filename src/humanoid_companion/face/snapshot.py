"""A PNG of the face in a given state, rendered by headless Chrome from the served page."""

import os
import subprocess
import urllib.parse
from pathlib import Path

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def available() -> bool:
    return Path(CHROME).exists()


def snapshot(base_url: str, expression: str, caption: str, path: Path, size: str = "640,400") -> Path:
    """static=1: no event stream, which would keep the page loading and stall the screenshot."""
    query = urllib.parse.urlencode({"kiosk": 1, "static": 1, "expression": expression, "caption": caption})
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={size}",
                    "--virtual-time-budget=1500", f"--screenshot={path}", f"{base_url}/?{query}"],
                   check=True, capture_output=True, timeout=60)
    return path
