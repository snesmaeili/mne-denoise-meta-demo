"""Generate the QR code for the closing slide.

    python make_qr.py

Writes ``qr_colab.png`` and ``qr_colab.svg`` pointing at the Colab launcher.
Regenerate if the repository ever moves.
"""

from __future__ import annotations

from pathlib import Path

import segno

HERE = Path(__file__).resolve().parent
REPO = "snesmaeili/mne-denoise-meta-demo"
NOTEBOOK = "meta_mne_denoise_demo.ipynb"

COLAB_URL = f"https://colab.research.google.com/github/{REPO}/blob/main/{NOTEBOOK}"


def main() -> int:
    qr = segno.make(COLAB_URL, error="h")  # high correction: survives a projector
    png = HERE / "qr_colab.png"
    svg = HERE / "qr_colab.svg"
    # scale 12 gives ~500 px, large enough to scan from the back of a room
    qr.save(png, scale=12, border=3, dark="#16202b", light="#ffffff")
    qr.save(svg, scale=12, border=3, dark="#16202b", light="#ffffff")
    print(f"  {COLAB_URL}")
    print(f"  {png.name}  {png.stat().st_size / 1024:.0f} KB")
    print(f"  {svg.name}  {svg.stat().st_size / 1024:.0f} KB")
    print("\n  Test it with a phone before the talk. The link only resolves once")
    print("  the repository is public.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
