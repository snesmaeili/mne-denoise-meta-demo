"""Download and verify the prepared demo assets.

The talk runs from a small cache of derived artefacts. On a machine that has
already run ``prepare_meta_demo.py`` this is a no-op; on a fresh machine, or in
Colab, it pulls the published archive and checks its SHA-256 before unpacking.

    python fetch_demo_data.py            # fetch if needed
    python fetch_demo_data.py --force    # re-download and overwrite
    python fetch_demo_data.py --check    # report only

Nothing here reaches OpenNeuro. Only the release archive is fetched, and only
when the cache is incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_utils as du  # noqa: E402

DATA_VERSION = "v2"
REPO = "snesmaeili/mne-denoise-meta-demo"
ARCHIVE = f"demo-data-{DATA_VERSION}.zip"
RELEASE_URL = f"https://github.com/{REPO}/releases/download/demo-data-{DATA_VERSION}/{ARCHIVE}"

#: SHA-256 of the published archive, from package_demo_data.py. Pinned so the
#: notebook six months from now runs on exactly the data shown at the talk.
EXPECTED_SHA256: str | None = (
    "63a427c75a828cb16566f78638e7b01eb41a93ee304efc1d03c496b9457c6e85"
)

RETRIES = 3


def _missing() -> list[str]:
    report = du.check_demo_assets(verbose=False)
    out: list[str] = []
    for state in report["acts"].values():
        out.extend(state["missing"])
    return out


def _download(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            print(f"  attempt {attempt}/{RETRIES} failed: {exc}")
    raise SystemExit(
        f"Could not download {url}\n  last error: {last}\n"
        "  If you have the datasets locally, run "
        "`python prepare_meta_demo.py --all` instead."
    )


def fetch(force: bool = False) -> int:
    cache = du.cache_dir()
    missing = _missing()
    if not missing and not force:
        print(f"demo assets already present in {cache}")
        return 0

    if missing:
        print(f"{len(missing)} asset(s) missing from {cache}")
    print(f"downloading {RELEASE_URL}")
    blob = _download(RELEASE_URL)

    actual = hashlib.sha256(blob).hexdigest()
    if EXPECTED_SHA256 is None:
        print(f"  sha256 {actual}  (not pinned in this checkout)")
    elif actual != EXPECTED_SHA256:
        raise SystemExit(
            f"Checksum mismatch for {ARCHIVE}\n"
            f"  expected {EXPECTED_SHA256}\n  actual   {actual}\n"
            "  Refusing to unpack."
        )
    else:
        print(f"  sha256 {actual}  verified")

    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            # Refuse absolute paths and traversal before writing anything.
            target = (cache / name).resolve()
            if not str(target).startswith(str(cache.resolve())):
                raise SystemExit(f"unsafe path in archive: {name}")
        zf.extractall(cache)
        print(f"  unpacked {len(zf.namelist())} files into {cache}")

    report = du.check_demo_assets(verbose=True)
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("--check", action="store_true", help="report only")
    args = parser.parse_args(argv)
    if args.check:
        report = du.check_demo_assets(verbose=True)
        return 0 if report["ok"] else 1
    return fetch(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
