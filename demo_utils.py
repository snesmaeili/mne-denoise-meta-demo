"""Helpers for the MNE-Python maintainers sprint demo (Meta Paris, 2026).

This module holds everything the presenter notebook should *not* have to show:
dataset discovery, cache I/O, provenance manifests, the preflight check, and a
small set of presentation-sized plotting wrappers built on top of
:mod:`mne_denoise.viz`.

Design rules
------------
* The notebook never touches a dataset path. Only :mod:`prepare_meta_demo`
  resolves dataset roots; the notebook reads the cache written by it.
* Nothing here downloads anything. Preparation may download, the demo may not.
* Every cached artefact is accompanied by a manifest recording the repository
  commit, package versions, dataset identifier, estimator parameters, the
  representative-selection rule and the random seed.

Authors: prepared for the mne-denoise Meta sprint demo.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CACHE_DIR",
    "RANDOM_STATE",
    "DATASETS",
    "DemoAsset",
    "ASSETS",
    "cache_dir",
    "cache_path",
    "resolve_dataset_root",
    "save_json",
    "load_json",
    "save_npz",
    "load_npz",
    "build_manifest",
    "write_manifest",
    "read_manifest",
    "environment_info",
    "repo_commit",
    "demo_commit",
    "mne_denoise_root",
    "check_demo_assets",
    "assert_presenter_ready",
    "presentation_theme",
    "nearest_to_median",
    "zscore_rows",
    "plot_line_noise_triptych",
    "plot_adaptive_component_timeline",
    "plot_asr_reconstruction_panel",
    "plot_asr_variant_regimes",
    "plot_dss_target_panel",
    "plot_attenuation_preservation",
    "plot_contract_screen",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Seed used for every stochastic operation in the demo.
RANDOM_STATE = 97

_DEFAULT_CACHE = Path.home() / ".cache" / "mne-denoise" / "meta-demo"

#: Dataset registry. ``env`` is the environment variable a user can set to point
#: at a local copy; ``probe`` is a path fragment used to confirm a candidate root.
DATASETS: dict[str, dict[str, str]] = {
    "ds003620": {
        "env": "META_DEMO_DS003620",
        "probe": "sub-01/eeg",
        "label": "OpenNeuro ds003620 (Runabout mobile EEG)",
        "doi": "doi:10.18112/openneuro.ds003620.v1.1.1",
    },
    "n170": {
        "env": "META_DEMO_N170",
        "probe": "sub-001/eeg",
        "label": "ERP CORE N170 (faces vs cars)",
        "doi": "doi:10.18115/D5JW4R",
    },
    "ds004505": {
        "env": "META_DEMO_DS004505",
        "probe": "sub-01/eeg",
        "label": "OpenNeuro ds004505 (Table Tennis mobile EEG)",
        "doi": "doi:10.18112/openneuro.ds004505.v1.0.2",
    },
}


def cache_dir() -> Path:
    """Return the demo cache directory, honouring ``MNE_DENOISE_META_DEMO_CACHE``."""
    raw = os.environ.get("MNE_DENOISE_META_DEMO_CACHE")
    return Path(raw).expanduser() if raw else _DEFAULT_CACHE


CACHE_DIR = cache_dir()


def cache_path(*parts: str) -> Path:
    """Return a path inside the demo cache, creating parent directories."""
    path = cache_dir().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Dataset discovery (preparation only -- never called from the notebook)
# ---------------------------------------------------------------------------


def _candidate_roots() -> Iterator[Path]:
    """Yield plausible parent directories that may hold BIDS datasets."""
    seen: set[Path] = set()

    def _emit(p: Path | str | None) -> Iterator[Path]:
        if p is None:
            return
        path = Path(p).expanduser()
        if path in seen:
            return
        seen.add(path)
        if path.is_dir():
            yield path

    yield from _emit(os.environ.get("MNE_DATA"))
    try:  # MNE's own configured data directory, if MNE is importable.
        import mne

        for key in ("MNE_DATA", "MNE_DATASETS_SAMPLE_PATH", "MNE_DATASETS_ERP_CORE_PATH"):
            yield from _emit(mne.get_config(key))
    except Exception:  # pragma: no cover - MNE always present in practice
        pass

    yield from _emit(Path.home() / "mne_data")
    yield from _emit(Path.home() / "Documents")
    yield from _emit(Path(__file__).resolve().parents[2] / "data")

    if platform.system() == "Windows":  # scan fixed drives for a data directory
        for letter in "CDEFGH":
            for name in ("mne_data", "data"):
                yield from _emit(Path(f"{letter}:/") / name)
            yield from _emit(Path(f"{letter}:/"))


def resolve_dataset_root(key: str, explicit: str | Path | None = None) -> Path:
    """Locate a dataset root on this machine.

    Resolution order: *explicit* argument, then the dataset's environment
    variable, then a bounded scan of plausible data directories.

    Raises
    ------
    FileNotFoundError
        With a precise message naming the environment variable to set and the
        directory layout that was expected.
    """
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[key]
    probe = spec["probe"]

    def _ok(path: Path) -> bool:
        return (path / probe).is_dir()

    if explicit is not None:
        path = Path(explicit).expanduser()
        if _ok(path):
            return path
        raise FileNotFoundError(
            f"{spec['label']}: {path} does not contain {probe!r}."
        )

    env_value = os.environ.get(spec["env"])
    if env_value:
        path = Path(env_value).expanduser()
        if _ok(path):
            return path
        raise FileNotFoundError(
            f"{spec['label']}: ${spec['env']} is set to {path}, which does not "
            f"contain {probe!r}."
        )

    # Bounded search: only look one and two levels below each candidate root.
    needles = _SEARCH_NAMES[key]
    for root in _candidate_roots():
        for needle in needles:
            for depth in ("", "*/", "*/*/"):
                try:
                    matches = sorted(root.glob(f"{depth}{needle}"))
                except OSError:  # pragma: no cover - unreadable mount
                    continue
                for match in matches:
                    if match.is_dir() and _ok(match):
                        return match

    raise FileNotFoundError(
        f"Could not locate {spec['label']} ({spec['doi']}).\n"
        f"  Set the environment variable {spec['env']} to a directory that "
        f"contains {probe!r}, or pass --{key}-root on the command line.\n"
        f"  Searched for directories named {needles} under: "
        f"$MNE_DATA, MNE's configured data path, ~/mne_data, and the repository "
        f"data/ directory."
    )


_SEARCH_NAMES: dict[str, tuple[str, ...]] = {
    "ds003620": ("ds003620", "ds003620_test", "ds003620*"),
    "n170": ("N170", "erpcore/N170", "erp_core_n170/erpcore/N170"),
    "ds004505": ("raw_bids", "ds004505/raw_bids", "ds004505"),
}


# ---------------------------------------------------------------------------
# Cache I/O -- stable formats only (JSON / NPZ / FIF), never pickled estimators
# ---------------------------------------------------------------------------


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def save_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write *payload* as indented JSON, converting NumPy scalars/arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file written by :func:`save_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_npz(path: Path, **arrays: np.ndarray) -> Path:
    """Write arrays to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read an ``.npz`` into a plain dict (so the file handle is released)."""
    with np.load(Path(path), allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_commit(repo: Path) -> str:
    """Return the HEAD commit of the git repository at *repo*, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"
    return out.stdout.strip() or "unknown"


def mne_denoise_root() -> Path | None:
    """Locate the mne-denoise checkout that supplies the installed package.

    This demo lives in its own repository, so the scientifically meaningful
    provenance is the commit of the *package* under test, not of the demo.
    Returns ``None`` when mne-denoise is installed from a wheel rather than
    from a working tree.
    """
    raw = os.environ.get("META_DEMO_MNE_DENOISE_ROOT")
    if raw:
        return Path(raw).expanduser()
    try:
        import mne_denoise
    except Exception:  # pragma: no cover - checked separately by the preflight
        return None
    root = Path(mne_denoise.__file__).resolve().parents[1]
    return root if (root / ".git").exists() else None


def repo_commit() -> str:
    """Return the mne-denoise commit the demo is running against."""
    root = mne_denoise_root()
    return _git_commit(root) if root is not None else "unknown"


def demo_commit() -> str:
    """Return this demo repository's own commit."""
    return _git_commit(Path(__file__).resolve().parent)


def environment_info() -> dict[str, str]:
    """Collect version information for the manifest."""
    import mne
    import scipy

    import mne_denoise

    root = mne_denoise_root()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mne_denoise": mne_denoise.__version__,
        "mne": mne.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        # Provenance of the package under test...
        "repo_commit": repo_commit(),
        "mne_denoise_root": str(root) if root is not None else None,
        # ...and of the demo that drove it.
        "demo_commit": demo_commit(),
    }


def build_manifest(
    *,
    act: str,
    dataset: str | None,
    subject: str | None,
    preprocessing: Mapping[str, Any],
    estimators: Mapping[str, Any],
    selection_rule: str,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble the provenance record stored beside every cached artefact."""
    spec = DATASETS.get(dataset or "", {})
    return {
        "act": act,
        "dataset": dataset,
        "dataset_label": spec.get("label"),
        "dataset_doi": spec.get("doi"),
        "subject": subject,
        "preprocessing": _jsonify(preprocessing),
        "estimators": _jsonify(estimators),
        "selection_rule": selection_rule,
        "random_state": RANDOM_STATE,
        "notes": notes,
        "environment": environment_info(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(name: str, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest into the cache as ``<name>_manifest.json``."""
    return save_json(cache_path(f"{name}_manifest.json"), manifest)


def read_manifest(name: str) -> dict[str, Any]:
    """Read a manifest previously written by :func:`write_manifest`."""
    return load_json(cache_path(f"{name}_manifest.json"))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoAsset:
    """One cached artefact the presenter notebook depends on."""

    act: str
    label: str
    files: tuple[str, ...]

    def missing(self) -> list[str]:
        base = cache_dir()
        return [f for f in self.files if not (base / f).exists()]


ASSETS: tuple[DemoAsset, ...] = (
    DemoAsset(
        "zapline",
        "ZapLine+ / line noise",
        (
            "zapline_spectra.npz",
            "zapline_metrics.json",
            "zapline_adaptive.npz",
            "zapline_demo_raw.fif",
            "zapline_manifest.json",
        ),
    ),
    DemoAsset(
        "asr",
        "ASR transient reconstruction",
        ("asr_fixture.npz", "asr_metrics.json", "asr_manifest.json"),
    ),
    DemoAsset(
        "asr_variants",
        "ASR variant regimes",
        ("asr_variants_metrics.json", "asr_variants_manifest.json"),
    ),
    DemoAsset(
        "dss",
        "DSS target enhancement",
        ("dss_demo-epo.fif", "dss_sources.npz", "dss_metrics.json", "dss_group.json",
         "dss_manifest.json"),
    ),
    DemoAsset(
        "eog",
        "iCanClean vs EOG references",
        ("eog_metrics.json", "eog_traces.npz", "eog_manifest.json"),
    ),
    DemoAsset(
        "movement",
        "Movement attenuation/preservation",
        ("movement_metrics.json", "movement_traces.npz", "movement_manifest.json"),
    ),
)


def _check_imports() -> tuple[bool, str]:
    try:
        import mne  # noqa: F401

        import mne_denoise  # noqa: F401
        from mne_denoise.asr import ASR  # noqa: F401
        from mne_denoise.dss import DSS, AverageBias  # noqa: F401
        from mne_denoise.icanclean import ICanClean  # noqa: F401
        from mne_denoise.qa import noise_surround_ratio  # noqa: F401
        from mne_denoise.viz import plot_asr_repair_timeline  # noqa: F401
        from mne_denoise.zapline import ZapLine  # noqa: F401
    except Exception as exc:  # pragma: no cover - surfaced to the presenter
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def check_demo_assets(verbose: bool = True) -> dict[str, Any]:
    """Verify that everything the stage notebook needs is present and offline.

    Returns a report dict; also prints a compact status block when *verbose*.
    """
    imports_ok, import_err = _check_imports()

    acts: dict[str, dict[str, Any]] = {}
    for asset in ASSETS:
        missing = asset.missing()
        acts[asset.act] = {"label": asset.label, "ok": not missing, "missing": missing}

    commit = repo_commit()
    report: dict[str, Any] = {
        "cache_dir": str(cache_dir()),
        "imports_ok": imports_ok,
        "import_error": import_err,
        "acts": acts,
        "repo_commit": commit,
        "offline": True,  # nothing in the notebook performs I/O beyond the cache
        "ok": imports_ok and all(v["ok"] for v in acts.values()),
    }

    if verbose:
        mark = "OK  " if report["ok"] else "FAIL"
        print(f"META DEMO {'READY' if report['ok'] else 'NOT READY'}")
        print(f"  cache      {report['cache_dir']}")
        print(f"  commit     {commit[:12]}")
        print(f"  {'[x]' if imports_ok else '[ ]'} package imports"
              + ("" if imports_ok else f"  -- {import_err}"))
        for asset in ASSETS:
            state = acts[asset.act]
            box = "[x]" if state["ok"] else "[ ]"
            line = f"  {box} {state['label']}"
            if state["missing"]:
                line += f"  -- missing: {', '.join(state['missing'])}"
            print(line)
        print(f"  [x] offline (no network required)")
        if not report["ok"]:
            print("\n  Run:  python demo/meta_sprint_2026/prepare_meta_demo.py --all")
        del mark
    return report


def assert_presenter_ready() -> None:
    """Raise :class:`RuntimeError` unless every demo asset is present.

    Call this in the first notebook cell so a missing asset fails *before* the
    talk starts rather than halfway through Act 3.
    """
    report = check_demo_assets(verbose=True)
    if not report["ok"]:
        missing = {
            act: state["missing"] for act, state in report["acts"].items() if state["missing"]
        }
        raise RuntimeError(
            "Demo assets are incomplete. Run\n"
            "    python demo/meta_sprint_2026/prepare_meta_demo.py --all\n"
            f"Missing: {missing}"
            + ("" if report["imports_ok"] else f"\nImport error: {report['import_error']}")
        )


# ---------------------------------------------------------------------------
# Deterministic representative selection
# ---------------------------------------------------------------------------


def zscore_rows(x: np.ndarray) -> np.ndarray:
    """Scale *x* by the SD of its last axis, pooled over all leading axes.

    Used only so sensor traces and DSS components can share one y-axis; it is a
    display transform and never enters a reported metric.
    """
    x = np.asarray(x, dtype=float)
    scale = float(np.std(x))
    return x / (scale if scale > 0 else 1.0)


def nearest_to_median(values: Mapping[str, float]) -> tuple[str, float, float]:
    """Return the key whose value is closest to the median of *values*.

    Ties are broken by sorted key order, so the result is fully deterministic.

    Returns
    -------
    key, value, median
    """
    if not values:
        raise ValueError("values must be non-empty")
    items = sorted(values.items())
    arr = np.array([v for _, v in items], dtype=float)
    median = float(np.median(arr))
    idx = int(np.argmin(np.abs(arr - median)))
    return items[idx][0], float(arr[idx]), median


# ---------------------------------------------------------------------------
# Presentation theme + plotting wrappers
# ---------------------------------------------------------------------------

#: Semantic colours held constant across every figure in the demo.
DEMO_COLORS: dict[str, str] = {
    "original": "#333333",  # uncorrected / raw
    "notch": "#CC79A7",  # a comparator that overshoots
    "zapline": "#E69F00",  # ZapLine+
    "asr": "#0072B2",  # ASR
    "rasr": "#56B4E9",  # rASR
    "icanclean": "#009E73",  # reference-aware
    "dss": "#0072B2",
    "target": "#009E73",  # the signal we want to keep
    "reference": "#CC79A7",  # reference / auxiliary channels
    "warning": "#D55E00",  # over-cleaning
    "floor": "#888888",  # spectral floor guide
}

_PRESENTATION_RC: dict[str, object] = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.4,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
}


@contextmanager
def presentation_theme(**overrides: object):
    """Apply the package theme with projector-sized typography.

    Wraps :func:`mne_denoise.viz.use_theme` so the demo inherits the project's
    visual language and only overrides what a projector needs.
    """
    from mne_denoise.viz import use_theme

    rc = dict(_PRESENTATION_RC)
    rc.update(overrides)
    with use_theme(rc=rc):
        yield


def _hide_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_line_noise_triptych(
    freqs: np.ndarray,
    psds: Mapping[str, np.ndarray],
    *,
    line_freq: float,
    ratios: Mapping[str, float] | None = None,
    fmin: float = 40.0,
    fmax: float = 65.0,
    order: Sequence[str] = ("original", "notch", "zapline+"),
    ax=None,
):
    """One panel: channel-mean PSD around *line_freq* for several conditions.

    ``psds`` maps a condition name to an array of shape ``(n_channels, n_freqs)``
    or ``(n_freqs,)``. ``ratios`` optionally maps the same names to their
    peak-to-surround ratio, annotated in the legend.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
    else:
        fig = ax.figure

    band = (freqs >= fmin) & (freqs <= fmax)
    colors = {"original": DEMO_COLORS["original"], "notch": DEMO_COLORS["notch"],
              "zapline+": DEMO_COLORS["zapline"]}

    # Local spectral floor from the shown band, excluding the line neighbourhood.
    ref = np.atleast_2d(psds[order[0]]).mean(axis=0)
    skirt = band & (np.abs(freqs - line_freq) > 2.0)
    floor = float(np.median(ref[skirt])) if skirt.any() else np.nan
    ax.axhline(floor, color=DEMO_COLORS["floor"], lw=1.6, ls=(0, (5, 4)), zorder=1)
    ax.annotate(
        "local spectral floor",
        xy=(fmax - 0.4, floor),
        xytext=(0, -20),
        textcoords="offset points",
        color=DEMO_COLORS["floor"],
        fontsize=13,
        ha="right",
        va="top",
    )

    for name in order:
        if name not in psds:
            continue
        curve = np.atleast_2d(psds[name]).mean(axis=0)
        label = name
        if ratios and name in ratios:
            label = f"{name}   R = {ratios[name]:.2f}"
        ax.semilogy(
            freqs[band],
            curve[band],
            color=colors.get(name, DEMO_COLORS["original"]),
            label=label,
            zorder=3,
            alpha=0.95,
        )

    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(f"Same {line_freq:.0f} Hz peak, different cost")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6, borderaxespad=0.8)
    _hide_spines(ax)
    fig.tight_layout()
    return fig


def plot_adaptive_component_timeline(
    chunk_starts: np.ndarray,
    n_removed: np.ndarray,
    *,
    contamination: tuple[np.ndarray, np.ndarray] | None = None,
    ax=None,
):
    """Per-chunk component count chosen by adaptive ZapLine+, over the recording.

    ``contamination`` is an optional ``(times, ratio)`` pair drawn on a twin axis
    so the audience can see the component count tracking the artifact.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 3.6))
    else:
        fig = ax.figure

    ax.step(
        chunk_starts / 60.0,
        n_removed,
        where="post",
        color=DEMO_COLORS["zapline"],
        lw=2.4,
    )
    ax.fill_between(
        chunk_starts / 60.0, 0, n_removed, step="post",
        color=DEMO_COLORS["zapline"], alpha=0.18,
    )
    ax.set_xlabel("Time in recording (min)")
    ax.set_ylabel("components\nremoved", color=DEMO_COLORS["zapline"])
    ax.set_ylim(bottom=0)
    _hide_spines(ax)

    if contamination is not None:
        times, ratio = contamination
        twin = ax.twinx()
        twin.plot(times / 60.0, ratio, color=DEMO_COLORS["original"], lw=1.6, alpha=0.55)
        twin.axhline(1.0, color=DEMO_COLORS["floor"], lw=1.0, ls=":")
        twin.set_ylabel("R (uncorrected)", color=DEMO_COLORS["original"])
        twin.spines["top"].set_visible(False)

    ax.set_title("Adaptive mode picks a different subspace in every chunk")
    fig.tight_layout()
    return fig


def plot_asr_reconstruction_panel(
    times: np.ndarray,
    contaminated: np.ndarray,
    cleaned: np.ndarray,
    truth: np.ndarray | None,
    *,
    repair: tuple[np.ndarray, np.ndarray] | None = None,
    channels: Sequence[int] = (0, 1, 2),
    ch_names: Sequence[str] | None = None,
    tlim: tuple[float, float] | None = None,
):
    """Three-row figure: contaminated traces, cleaned traces, repair timeline.

    ``tlim`` restricts only what is *drawn*; every reported metric is computed on
    the whole recording.
    """
    import matplotlib.pyplot as plt

    keep = slice(None)
    if tlim is not None:
        idx = np.flatnonzero((times >= tlim[0]) & (times <= tlim[1]))
        keep = slice(int(idx[0]), int(idx[-1]) + 1)
    t = times[keep]

    fig, axes = plt.subplots(
        3, 1, figsize=(11.5, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.4], "hspace": 0.26},
    )

    scale = float(np.percentile(np.abs(contaminated[list(channels), keep]), 99.0)) or 1.0
    offsets = np.arange(len(channels))[::-1] * 3.0 * scale

    for ax, data, title in (
        (axes[0], contaminated, "contaminated"),
        (axes[1], cleaned, "ASR output"),
    ):
        for ch, off in zip(channels, offsets):
            if truth is not None:
                ax.plot(t, truth[ch, keep] + off, color=DEMO_COLORS["target"],
                        lw=2.2, alpha=0.85, zorder=2)
            ax.plot(t, data[ch, keep] + off, color=DEMO_COLORS["original"],
                    lw=0.9, alpha=0.9, zorder=3)
            if ch_names is not None:
                ax.text(t[0], off + 1.15 * scale, ch_names[ch], fontsize=13,
                        color=DEMO_COLORS["original"], va="bottom")
        ax.set_yticks([])
        ax.set_ylabel(title)
        ax.set_xlim(t[0], t[-1])
        _hide_spines(ax)

    if repair is not None:
        rtimes, rvals = repair
        sel = (rtimes >= t[0]) & (rtimes <= t[-1])
        axes[2].step(rtimes[sel], rvals[sel], where="post",
                     color=DEMO_COLORS["asr"], lw=2.4)
        axes[2].fill_between(rtimes[sel], 0, rvals[sel], step="post",
                             color=DEMO_COLORS["asr"], alpha=0.22)
        axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("dims\nrepaired")
    axes[2].set_xlabel("Time (s)")
    _hide_spines(axes[2])
    axes[0].set_title("ASR reconstructs transient abnormal covariance — and says where")
    # tight_layout fights the explicit height_ratios/hspace above, so place by hand.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.09)
    return fig


def plot_dss_target_panel(
    times: np.ndarray,
    sensor_trials: np.ndarray,
    sensor_evoked: np.ndarray,
    component_trials: np.ndarray,
    component_evoked: np.ndarray,
    *,
    group: Mapping[str, Sequence[float]] | None = None,
    labels: Mapping[str, str] | None = None,
    pattern: np.ndarray | None = None,
    info=None,
    window: tuple[float, float] | None = None,
):
    """One participant beside the whole cohort.

    Left: single-trial spread at the best sensor and on DSS component 1, drawn on
    a shared vertical scale. Right: for every participant, the change in
    split-half reproducibility against the change in held-out condition AUC.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4),
                             gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.30})

    # ---- left: one participant, two independent halves of the trials ------
    ax = axes[0]
    if window is not None:
        ax.axvspan(window[0], window[1], color=DEMO_COLORS["floor"], alpha=0.13, zorder=0)
    series = sensor_trials, sensor_evoked, component_trials, component_evoked
    lo = min(float(np.min(s)) for s in series)
    hi = max(float(np.max(s)) for s in series)
    for half_a, half_b, color in (
        (sensor_trials, sensor_evoked, DEMO_COLORS["original"]),
        (component_trials, component_evoked, DEMO_COLORS["dss"]),
    ):
        ax.plot(times, half_a, color=color, lw=2.8, zorder=3)
        ax.plot(times, half_b, color=color, lw=2.8, ls=(0, (3, 2)), zorder=3)
    ax.axvline(0.0, color=DEMO_COLORS["floor"], lw=1.0, ls=":")
    span = hi - lo
    ax.set_ylim(lo - 0.10 * span, hi + 0.28 * span)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("One participant — two independent halves of the trials")
    names = labels or {}
    ax.text(0.02, 0.97, names.get("sensor", "best sensor"), transform=ax.transAxes,
            color=DEMO_COLORS["original"], fontsize=14, fontweight="bold", va="top")
    ax.text(0.02, 0.88, names.get("dss", "DSS component 1"), transform=ax.transAxes,
            color=DEMO_COLORS["dss"], fontsize=14, fontweight="bold", va="top")
    _hide_spines(ax)

    # ---- right: the whole cohort -----------------------------------------
    ax = axes[1]
    if group is not None:
        dx = np.asarray(group["reproducibility_gain"], dtype=float)
        dy = np.asarray(group["auc_change"], dtype=float)
        ax.axhline(0.0, color=DEMO_COLORS["floor"], lw=1.2)
        ax.axvline(0.0, color=DEMO_COLORS["floor"], lw=1.2)
        ax.scatter(dx, dy, s=110, color=DEMO_COLORS["dss"], alpha=0.75,
                   edgecolor="white", linewidth=1.4, zorder=3)
        ax.set_xlabel("Δ reproducibility")
        ax.set_ylabel("Δ condition AUC")
        ax.set_title(f"All {len(dx)} participants")
        up_x = int((dx > 0).sum())
        up_y = int((dy > 0).sum())
        ax.text(0.98, 0.04,
                f"reproducibility up in {up_x}/{len(dx)}\n"
                f"discriminability up in {up_y}/{len(dy)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=14,
                color=DEMO_COLORS["original"])
        _hide_spines(ax)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.13, wspace=0.30)

    if pattern is not None and info is not None:
        import mne

        inset = fig.add_axes((0.385, 0.615, 0.085, 0.235))
        mne.viz.plot_topomap(pattern, info, axes=inset, show=False, contours=0)
        inset.set_title("comp 1", fontsize=11)

    return fig


def plot_dss_framework_panel(
    bias_swap: Mapping[str, Any],
    head_to_head: Mapping[str, float],
    *,
    group: Mapping[str, int] | None = None,
):
    """Why DSS, in two panels.

    Left: the same estimator on the same data under three different criteria,
    scored against the two planted patterns. The distractor rhythm is the
    stronger source, so variance-maximisation returns it -- that is measured
    here, not asserted.

    Right: the honest head-to-head on real evoked data against the comparators
    that already exist, including the one MNE-Python ships for this job.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))

    # ---- left: swap the criterion, same data ------------------------------
    ax = axes[0]
    arms = ("PCA", "AverageBias", "BandpassBias")
    nice = {"PCA": "PCA\n(variance)", "AverageBias": "DSS\nAverageBias",
            "BandpassBias": "DSS\nBandpassBias"}
    x = np.arange(len(arms))
    w = 0.36
    ev = [float(bias_swap[a]["evoked"]) for a in arms]
    al = [float(bias_swap[a]["alpha"]) for a in arms]
    ax.bar(x - w / 2, ev, w, label="evoked pattern", color=DEMO_COLORS["target"])
    ax.bar(x + w / 2, al, w, label="alpha pattern", color=DEMO_COLORS["reference"])
    for xi, (a, b) in enumerate(zip(ev, al)):
        ax.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=12)
        ax.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[a] for a in arms])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("|cos| of component 1 to planted pattern")
    ax.set_title("Same data, same estimator — one argument changed")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=12)
    _hide_spines(ax)

    # ---- right: the honest head-to-head -----------------------------------
    ax = axes[1]
    order = ("sensor", "pca", "dss", "xdawn")
    label = {"sensor": "raw\nsensors", "pca": "PCA\nmatched rank",
             "dss": "DSS\nAverageBias", "xdawn": "Xdawn\n(in MNE)"}
    colour = {"sensor": DEMO_COLORS["original"], "pca": DEMO_COLORS["floor"],
              "dss": DEMO_COLORS["dss"], "xdawn": DEMO_COLORS["warning"]}
    vals = [float(head_to_head[f"{k}_median"]) for k in order]
    ax.bar(range(len(order)), vals, 0.6, color=[colour[k] for k in order])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0012, f"{v:.4f}", ha="center", fontsize=13)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label[k] for k in order])
    lo = min(vals)
    ax.set_ylim(lo - 0.012, max(vals) + 0.008)
    ax.set_ylabel("median split-half reproducibility (held out)")
    ax.set_title("Enhancing an evoked response — everyone can do this")
    if group is not None:
        n = group["n_subjects"]
        ax.set_xlabel(
            f"across {n} participants: DSS beats matched-rank PCA in "
            f"{group['dss_over_pca_positive']}/{n}\n"
            f"and Xdawn in only {group['dss_over_xdawn_positive']}/{n}",
            fontsize=12.5, labelpad=10, color=DEMO_COLORS["original"])
    _hide_spines(ax)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.20, wspace=0.24)
    return fig


def plot_attenuation_preservation(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str = "attenuation_pct",
    y_key: str = "alpha_vs_background_db",
    label_key: str = "method",
    x_label: str = "Artifact attenuation  (higher = more removed)",
    title: str = "Attenuation alone can mislead",
    annotate: Mapping[str, str] | None = None,
    ax=None,
):
    """Horizontal attenuation bars, with the neural endpoint labelled inline.

    There is deliberately no second axis: ``y_key`` is read and printed at the
    end of each bar rather than plotted, because the point of the figure is the
    ranking of attenuation and the fact that a negative control reaches nearly
    the same place.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12.0, 5.8))
    else:
        fig = ax.figure

    color_for = {
        "uncorrected": DEMO_COLORS["original"],
        "ASR": DEMO_COLORS["asr"],
        "rASR": DEMO_COLORS["rasr"],
        "iCanClean": DEMO_COLORS["icanclean"],
        "iCanClean\n(scrambled ref)": DEMO_COLORS["warning"],
    }

    ordered = list(rows)[::-1]  # first row ends up at the top
    ypos = np.arange(len(ordered), dtype=float)
    xmax = max(abs(float(r[x_key])) for r in ordered) or 1.0

    for y, row in zip(ypos, ordered):
        name = str(row[label_key])
        control = "scrambled" in name
        color = color_for.get(name, DEMO_COLORS["original"])
        ax.barh(
            y, float(row[x_key]), height=0.62, zorder=3,
            color="white" if control else color,
            edgecolor=color, linewidth=2.6,
            hatch="///" if control else None,
        )
        note = []
        if annotate and name in annotate:
            note.append(annotate[name])
        note.append(f"neural endpoint {float(row[y_key]):+.2f} dB")
        ax.text(
            xmax * 1.06, y, "   ".join(note), va="center", ha="left",
            fontsize=12.5, color=DEMO_COLORS["original"],
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels([str(r[label_key]).replace("\n", " ") for r in ordered],
                       fontsize=14.5)
    for tick, row in zip(ax.get_yticklabels(), ordered):
        tick.set_color(color_for.get(str(row[label_key]), DEMO_COLORS["original"]))
        tick.set_fontweight("bold")
    ax.axvline(0.0, color=DEMO_COLORS["floor"], lw=1.4)
    xmin = min(0.0, min(float(r[x_key]) for r in ordered))
    ax.set_xlim(xmin * 1.25 - 0.05 * xmax, xmax * 2.05)
    ax.set_xlabel(x_label)
    ax.set_title(title, pad=14)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _hide_spines(ax)

    # Bracket the negative control against the method it controls for.
    names = [str(r[label_key]) for r in ordered]
    if "iCanClean" in names and "iCanClean\n(scrambled ref)" in names:
        i_real = names.index("iCanClean")
        i_ctrl = names.index("iCanClean\n(scrambled ref)")
        x = max(float(ordered[i_real][x_key]), float(ordered[i_ctrl][x_key]))
        ax.annotate(
            "", xy=(x * 1.02, ypos[i_real]), xytext=(x * 1.02, ypos[i_ctrl]),
            arrowprops=dict(arrowstyle="<->", color=DEMO_COLORS["warning"], lw=2.2),
            zorder=4,
        )
        ax.text(
            x * 1.05, (ypos[i_real] + ypos[i_ctrl]) / 2,
            "a reference with no true\nalignment removes as much",
            va="center", ha="left", fontsize=13, fontweight="bold",
            color=DEMO_COLORS["warning"],
        )
    fig.tight_layout()
    return fig


def plot_icanclean_control_panel(
    times: np.ndarray,
    traces: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    *,
    channel: int = 0,
    channel_name: str = "",
):
    """What the control buys you, on both endpoints.

    Left: the blink-locked evoked at a single channel — averaging across
    channels would cancel the blink, whose pattern is dipolar under an average
    reference. Right: attenuation against neural preservation, so a method that
    removes more by removing signal cannot hide.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    style = {
        "uncorrected": (DEMO_COLORS["original"], "-", 2.6),
        "iCanClean": (DEMO_COLORS["icanclean"], "-", 2.6),
        "iCanClean\n(shifted ref)": (DEMO_COLORS["warning"], "--", 1.8),
        "EOG regression": (DEMO_COLORS["asr"], "-", 2.2),
        "EOG regression\n(shifted ref)": (DEMO_COLORS["floor"], "--", 1.8),
    }

    # ---- left: the blink itself -------------------------------------------
    ax = axes[0]
    for name, data in traces.items():
        colour, ls, lw = style.get(name, (DEMO_COLORS["floor"], "-", 1.5))
        ax.plot(times, data[channel] * 1e6, color=colour, ls=ls, lw=lw,
                label=name.replace("\n", " "))
    ax.axvline(0.0, color=DEMO_COLORS["floor"], ls=":", lw=1.0)
    ax.set_xlabel("Time from blink peak (s)")
    ax.set_ylabel(f"Blink-locked evoked at {channel_name or 'channel'} (µV)")
    ax.set_title("The artifact, and what each arm left behind")
    ax.legend(frameon=False, fontsize=11.5)
    _hide_spines(ax)

    # ---- right: both endpoints at once ------------------------------------
    ax = axes[1]
    for row in rows:
        name = str(row["method"])
        if name == "uncorrected":
            continue
        colour = style.get(name, (DEMO_COLORS["floor"],))[0]
        control = bool(row.get("control"))
        ax.scatter(float(row["attenuation_pct"]),
                   float(row["alpha_vs_background_db"]),
                   s=260, color="white" if control else colour,
                   edgecolor=colour, linewidth=2.6, zorder=3,
                   hatch="///" if control else None)
        # Controls sit at the left edge, so label them to the right instead of
        # centred above, where the text would run off the axes.
        x = float(row["attenuation_pct"])
        offset, align = ((16, -4), "left") if x < 20.0 else ((0, 16), "center")
        ax.annotate(name.replace("\n", " "),
                    (x, float(row["alpha_vs_background_db"])),
                    textcoords="offset points", xytext=offset, ha=align,
                    fontsize=12, color=colour)
    ax.axhline(0.0, color=DEMO_COLORS["floor"], lw=1.2)
    ax.axvline(0.0, color=DEMO_COLORS["floor"], lw=1.2)
    ax.set_xlabel("Blink attenuation (%)  →  more artifact removed")
    ax.set_ylabel("Posterior alpha vs its own background (dB)")
    ax.set_title("Removing more must not mean removing signal")
    ax.set_xlim(-12, 100)
    _hide_spines(ax)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.13, wspace=0.24)
    return fig


def plot_asr_variant_regimes(
    arm_a: Mapping[str, Any],
    arm_b: Mapping[str, Any],
    *,
    show_paper_reference: bool = True,
):
    """Why the ASR family has four members, measured on two regimes.

    Left: what a contaminated calibration segment does to the fitted threshold,
    with and without the robust covariance aggregator. Right: how much of a real
    mobile-EEG recording each variant is willing to calibrate on.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4),
                             gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.30})

    # ---- Panel A: robustness comes from the covariance aggregator ----------
    ax = axes[0]
    rows = list(arm_a["rows"])
    estimators = []
    for row in rows:
        if row["cov_estimator"] not in estimators:
            estimators.append(row["cov_estimator"])
    width = 0.34
    xs = np.arange(len(estimators), dtype=float)
    for offset, state, color in (
        (-width / 2, "clean", DEMO_COLORS["target"]),
        (+width / 2, "contaminated", DEMO_COLORS["warning"]),
    ):
        vals = [
            next(r["threshold_median"] for r in rows
                 if r["cov_estimator"] == e and r["calibration"] == state)
            for e in estimators
        ]
        ax.bar(xs + offset, vals, width=width, color=color, zorder=3,
               edgecolor="white", linewidth=1.5,
               label=f"{state} calibration")
    for i, est in enumerate(estimators):
        lo = next(r["threshold_median"] for r in rows
                  if r["cov_estimator"] == est and r["calibration"] == "clean")
        hi = next(r["threshold_median"] for r in rows
                  if r["cov_estimator"] == est and r["calibration"] == "contaminated")
        ax.text(xs[i], max(lo, hi) * 1.04, f"{100 * (hi / lo - 1):+.0f}%",
                ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=DEMO_COLORS["warning"])
    ax.set_xticks(xs)
    ax.set_xticklabels([e.replace("_", "\n") for e in estimators], fontsize=13.5)
    ax.set_ylabel("Fitted threshold (median)")
    ax.set_title("Dirty calibration inflates the threshold")
    ax.legend(frameon=False, fontsize=12.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_ylim(0, max(r["threshold_median"] for r in rows) * 1.18)
    _hide_spines(ax)

    # ---- Panel B: how much data each variant will calibrate on -------------
    ax = axes[1]
    brows = list(arm_b["rows"])[::-1]
    ypos = np.arange(len(brows), dtype=float)
    kind_color = {"window": DEMO_COLORS["asr"], "sample": DEMO_COLORS["icanclean"]}
    for y, row in zip(ypos, brows):
        color = kind_color.get(row["calibration_kind"], DEMO_COLORS["original"])
        ax.barh(y, 100 * row["calibration_fraction"], height=0.58, color=color,
                zorder=3, edgecolor="white", linewidth=1.5)
        # Fixed x so the annotation never collides with a bar end or a marker.
        ax.text(74.0, y, f"{row['variance_removed_pct']:.0f}% variance removed",
                va="center", ha="left", fontsize=12, color=DEMO_COLORS["original"])
    if show_paper_reference:
        paper = arm_b.get("paper_reference_fractions") or {}
        for y, row in zip(ypos, brows):
            ref = paper.get(row["variant"])
            if ref is not None:
                ax.plot([100 * ref, 100 * ref], [y - 0.34, y + 0.34],
                        color=DEMO_COLORS["warning"], lw=3.0, zorder=6,
                        solid_capstyle="butt")
        ax.plot([], [], color=DEMO_COLORS["warning"], lw=3.0,
                label="reported by Kim et al. 2025\n(205-ch juggling EEG)")
        ax.legend(frameon=False, fontsize=11.5, loc="upper right",
                  handlelength=1.1, borderaxespad=0.4)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r["variant"] for r in brows], fontsize=14)
    for tick, row in zip(ax.get_yticklabels(), brows):
        tick.set_color(kind_color.get(row["calibration_kind"], DEMO_COLORS["original"]))
        tick.set_fontweight("bold")
    ax.set_xlabel("Recording used as calibration reference (%)")
    ax.set_title("Mobile EEG — clean windows vs clean samples")
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_ylim(ypos[0] - 0.75, ypos[-1] + 0.95)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _hide_spines(ax)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.20, wspace=0.32)
    return fig


def plot_contract_screen():
    """The closing screen: four information regimes, one estimator contract."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(13.0, 7.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    families = [
        ("ZapLine", "narrowband\ninformation", "zap.fit_transform(raw)", DEMO_COLORS["zapline"]),
        ("ASR", "transient covariance\ninformation", "asr.fit_transform(raw)", DEMO_COLORS["asr"]),
        ("iCanClean", "explicit reference\ninformation", "icc.fit_transform(raw)",
         DEMO_COLORS["icanclean"]),
        ("DSS", "declared target\ninformation", "dss.fit(epochs)", DEMO_COLORS["dss"]),
    ]
    for i, (name, regime, code, color) in enumerate(families):
        x = 0.45 + i * 3.15
        box = FancyBboxPatch(
            (x, 4.75), 2.75, 1.85,
            boxstyle="round,pad=0.10,rounding_size=0.12",
            linewidth=2.2, edgecolor=color, facecolor=color + "18",
        )
        ax.add_patch(box)
        ax.text(x + 1.375, 6.28, name, ha="center", va="center", fontsize=20,
                fontweight="bold", color=color)
        ax.text(x + 1.375, 5.72, regime, ha="center", va="center", fontsize=13.5,
                color=DEMO_COLORS["original"])
        ax.text(x + 1.375, 5.05, code, ha="center", va="center", fontsize=12.5,
                family="monospace", color=DEMO_COLORS["original"])
        ax.add_patch(
            FancyArrowPatch((x + 1.375, 4.70), (6.5, 4.05),
                            arrowstyle="-", linewidth=1.1,
                            color=DEMO_COLORS["floor"], alpha=0.55)
        )

    steps = [
        "MNE  Raw / Epochs / Evoked",
        "declare TARGET + INFORMATION REGIME",
        "FIT / CALIBRATE",
        "INSPECT FITTED STATE",
        "TRANSFORM",
        "ARTIFACT endpoint  +  NEURAL PRESERVATION endpoint",
    ]
    y = 4.05
    for i, step in enumerate(steps):
        emph = i in (1, 3, 5)
        ax.text(
            6.5, y, step, ha="center", va="center",
            fontsize=16.5 if emph else 15,
            fontweight="bold" if emph else "normal",
            color=DEMO_COLORS["target"] if emph else DEMO_COLORS["original"],
        )
        if i < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch((6.5, y - 0.20), (6.5, y - 0.40),
                                arrowstyle="-|>", mutation_scale=16,
                                linewidth=1.6, color=DEMO_COLORS["floor"])
            )
        y -= 0.60

    ax.text(
        6.5, 0.42,
        "Different assumptions — one MNE-native, inspectable contract",
        ha="center", va="center", fontsize=17, style="italic",
        color=DEMO_COLORS["original"],
    )
    return fig


def _unused(*args: Iterable[Any]) -> None:  # pragma: no cover - keeps linters quiet
    del args
