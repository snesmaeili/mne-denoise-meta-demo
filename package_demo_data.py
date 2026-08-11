"""Bundle the prepared demo assets into a versioned, checksummed release.

The notebook reads a small set of derived artefacts, not the source recordings.
This packs exactly those into one archive so a fresh machine -- or Colab -- can
reproduce the talk without touching OpenNeuro.

    python package_demo_data.py            # build dist/demo-data-<version>.zip
    python package_demo_data.py --check    # verify an existing archive

Two of the bundled files are excerpts of real recordings released under
share-alike terms. ``NOTICE.md`` travels inside the archive so the attribution
obligation follows the data rather than living only in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_utils as du  # noqa: E402

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"

#: Bump when the contents change in a way that would alter the talk.
DATA_VERSION = "v2"

#: Files that ship inside the archive but are not demo assets.
EXTRA_FILES = ("NOTICE.md",)


def bundled_files() -> list[str]:
    """Every cached artefact the notebook needs, from the preflight registry."""
    names: list[str] = []
    for asset in du.ASSETS:
        for name in asset.files:
            if name not in names:
                names.append(name)
    return sorted(names)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(version: str = DATA_VERSION) -> Path:
    """Create ``dist/demo-data-<version>.zip`` plus a checksum manifest."""
    cache = du.cache_dir()
    names = bundled_files()

    missing = [n for n in names if not (cache / n).exists()]
    if missing:
        raise SystemExit(
            "Cannot package: these assets are missing from "
            f"{cache}\n  " + "\n  ".join(missing)
            + "\n\nRun `python prepare_meta_demo.py --all` first."
        )

    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / f"demo-data-{version}.zip"

    entries: dict[str, dict[str, object]] = {}
    total = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            src = cache / name
            zf.write(src, arcname=name)
            size = src.stat().st_size
            total += size
            entries[name] = {"sha256": sha256(src), "bytes": size}
            print(f"  + {name:<30s} {size / 1048576:7.2f} MB")
        for extra in EXTRA_FILES:
            src = HERE / extra
            if src.exists():
                zf.write(src, arcname=extra)
                entries[extra] = {"sha256": sha256(src), "bytes": src.stat().st_size}
                print(f"  + {extra:<30s} (attribution)")

    manifest = {
        "version": version,
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "uncompressed_bytes": total,
        "environment": du.environment_info(),
        "files": entries,
    }
    du.save_json(DIST / f"demo-data-{version}.json", manifest)

    print(f"\n  {archive}")
    print(f"  {archive.stat().st_size / 1048576:.2f} MB compressed "
          f"({total / 1048576:.2f} MB raw), {len(entries)} files")
    print(f"  sha256 {manifest['archive_sha256']}")
    return archive


def check(version: str = DATA_VERSION) -> int:
    """Verify a built archive against its manifest."""
    archive = DIST / f"demo-data-{version}.zip"
    manifest_path = DIST / f"demo-data-{version}.json"
    if not archive.exists() or not manifest_path.exists():
        print(f"nothing to check: {archive.name} or its manifest is absent")
        return 1
    manifest = du.load_json(manifest_path)
    actual = sha256(archive)
    ok = actual == manifest["archive_sha256"]
    print(f"  archive  {archive.name}")
    print(f"  expected {manifest['archive_sha256']}")
    print(f"  actual   {actual}")
    print("  MATCH" if ok else "  MISMATCH")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    missing = sorted(set(manifest["files"]) - names)
    if missing:
        print("  missing from archive: " + ", ".join(missing))
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=DATA_VERSION)
    parser.add_argument("--check", action="store_true", help="verify, do not build")
    args = parser.parse_args(argv)
    if args.check:
        return check(args.version)
    build(args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
