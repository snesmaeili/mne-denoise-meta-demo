"""A known-clean EEG fixture with transient artifacts, for the ASR act.

The point of this module is that the *clean* signal is known exactly, so the
demo can score two disjoint things: how much artifact was removed inside the
contaminated intervals, and how much untouched data survived outside them.

Everything here is pre-registered: the artifact typology, the amplitudes (as
multiples of the background SD, from textbook EEG amplitude ratios), the event
times, and the guard band. None of it was chosen after looking at a result.

The structural idiom ``amplitude * median_channel_SD * outer(spatial, temporal)``
is the same one used by the repository's existing burst injectors
(``scripts/run_asr_paper_validation.py::_inject_bursts``), so amplitudes are
comparable with the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FixtureSpec", "build_fixture", "build_masks", "rrmse", "mean_channel_corr"]

CH_NAMES: tuple[str, ...] = (
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FCz", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "POz", "O1", "Oz", "O2",
)

BACKGROUND_RMS_UV = 10.0  # median per-channel SD of the clean background

# Artifact amplitudes as multiples of the background SD, from textbook EEG
# amplitude ratios. Declared before any endpoint was computed.
AMP = {"blink": 10.0, "muscle": 6.0, "pop": 20.0, "covshift": 4.0}
POP_CHANNEL = "CP6"

#: Guard band around each event, excluded from BOTH endpoints. Equal to
#: ``window_length`` (0.5 s) + default ``lookahead`` (window_length / 2), i.e.
#: exactly how far a repaired ASR window may legitimately bleed past an edge.
GUARD_S = 0.75


@dataclass(frozen=True)
class FixtureSpec:
    """Declarative description of one fixture realisation."""

    duration_s: float = 60.0
    sfreq: float = 250.0
    random_state: int = 97
    #: (label, onset_s, offset_s), non-overlapping, repeated every ``period_s``.
    events: tuple[tuple[str, float, float], ...] = (
        ("blink", 3.00, 3.40),
        ("muscle", 7.00, 7.60),
        ("pop", 11.00, 11.30),
        ("covshift", 15.00, 17.00),
    )
    period_s: float = 20.0
    meta: dict = field(default_factory=dict)

    @property
    def n_times(self) -> int:
        return int(round(self.sfreq * self.duration_s))

    def expanded_events(self) -> list[tuple[str, float, float]]:
        """Tile the event block across the requested duration."""
        out: list[tuple[str, float, float]] = []
        n_blocks = int(np.floor(self.duration_s / self.period_s))
        for block in range(max(1, n_blocks)):
            shift = block * self.period_s
            for label, on, off in self.events:
                if off + shift <= self.duration_s:
                    out.append((f"{label}{block:d}" if n_blocks > 1 else label,
                                on + shift, off + shift))
        return out


def _pink_sources(rng, n_src, n_times, sfreq, exponent=1.0):
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    scale = np.zeros_like(freqs)
    scale[1:] = freqs[1:] ** (-exponent / 2.0)
    spec = (
        rng.standard_normal((n_src, freqs.size))
        + 1j * rng.standard_normal((n_src, freqs.size))
    ) * scale
    x = np.fft.irfft(spec, n=n_times, axis=1)
    x /= x.std(axis=1, keepdims=True)
    return x


def _dipolar_pattern(pos, c1, c2, sigma):
    d1 = np.linalg.norm(pos - c1, axis=1)
    d2 = np.linalg.norm(pos - c2, axis=1)
    p = np.exp(-(d1**2) / (2 * sigma**2)) - np.exp(-(d2**2) / (2 * sigma**2))
    return p / np.linalg.norm(p)


def _gaussian_pattern(pos, c, sigma):
    d = np.linalg.norm(pos - c, axis=1)
    p = np.exp(-(d**2) / (2 * sigma**2))
    return p / np.linalg.norm(p)


def _bandpass(x, sfreq, lo, hi):
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    spec = np.fft.rfft(x, axis=-1)
    spec[..., (freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n=n, axis=-1)


def _tukey(n, alpha=0.4):
    w = np.ones(n)
    edge = int(alpha * n / 2)
    if edge > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
        w[:edge] = ramp
        w[-edge:] = ramp[::-1]
    return w


def build_fixture(spec: FixtureSpec | None = None):
    """Build the fixture.

    Returns
    -------
    clean : ndarray, shape (n_channels, n_times), microvolts
        The ground truth. Contains no artifact.
    contaminated : ndarray, same shape
        ``clean`` plus the transient events.
    events : dict[str, tuple[int, int]]
        Event label -> (onset_sample, offset_sample).
    pos : ndarray, shape (n_channels, 3)
        Standard 10-20 electrode positions, metres.
    meta : dict
        Measured properties (background SD, covariance-direction leakage, ...).
    """
    import mne

    spec = spec or FixtureSpec()
    rng = np.random.default_rng(spec.random_state)
    n_times, sfreq = spec.n_times, spec.sfreq
    n_channels = len(CH_NAMES)

    montage = mne.channels.make_standard_montage("standard_1020")
    all_pos = montage.get_positions()["ch_pos"]
    pos = np.asarray([all_pos[name] for name in CH_NAMES])

    # --- clean background: spatially-correlated 1/f + posterior alpha --------
    n_src = 24
    src = _pink_sources(rng, n_src, n_times, sfreq, exponent=1.0)
    pairs = np.stack([rng.choice(n_channels, size=2, replace=False) for _ in range(n_src)])
    mixing = np.stack(
        [_dipolar_pattern(pos, pos[pairs[k, 0]], pos[pairs[k, 1]], 0.05) for k in range(n_src)],
        axis=1,
    )
    background = mixing @ src

    t = np.arange(n_times) / sfreq
    occipital = np.asarray([all_pos[n] for n in ("O1", "Oz", "O2")])
    alpha = np.zeros_like(background)
    for k in range(3):
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi)) ** 2
        wave = env * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
        alpha += np.outer(_gaussian_pattern(pos, occipital[k], 0.06), wave)

    background /= np.median(background.std(axis=1))
    alpha /= np.median(alpha.std(axis=1))
    # Independent per-electrode noise. Without it the 27 modelled sources span
    # fewer than 32 dimensions and the calibration covariance is singular.
    sensor = rng.standard_normal((n_channels, n_times))
    sensor /= np.median(sensor.std(axis=1))
    clean = BACKGROUND_RMS_UV * (background + 0.6 * alpha + 0.10 * sensor)
    clean *= BACKGROUND_RMS_UV / np.median(clean.std(axis=1))
    bg_sd = float(np.median(clean.std(axis=1)))

    # --- transient artifacts -------------------------------------------------
    contaminated = clean.copy()
    events: dict[str, tuple[int, int]] = {}
    for label, on_s, off_s in spec.expanded_events():
        events[label] = (int(round(on_s * sfreq)), int(round(off_s * sfreq)))

    covariance = np.cov(clean)
    _, evecs = np.linalg.eigh(covariance)
    top = evecs[:, -8:]
    leakage = []

    for label, (a, b) in events.items():
        kind = "".join(ch for ch in label if not ch.isdigit())
        n = b - a
        if kind == "blink":
            tt = np.linspace(-1.5, 2.5, n)
            shape = np.exp(-(tt**2) / (2 * 0.55**2))
            shape /= np.abs(shape).max()
            pattern = _gaussian_pattern(pos, all_pos["Fpz"], 0.055)
            pattern /= np.abs(pattern).max()
            contaminated[:, a:b] += AMP[kind] * bg_sd * np.outer(pattern, shape)
        elif kind == "muscle":
            emg = _bandpass(rng.standard_normal(n), sfreq, 20.0, 70.0) * _tukey(n, 0.3)
            emg /= emg.std()
            pattern = _gaussian_pattern(pos, all_pos["T7"], 0.05)
            pattern /= np.linalg.norm(pattern)
            contaminated[:, a:b] += AMP[kind] * bg_sd * np.outer(pattern, emg)
        elif kind == "pop":
            idx = CH_NAMES.index(POP_CHANNEL)
            decay = np.exp(-np.arange(n) / (0.10 * sfreq))
            contaminated[idx, a:b] += AMP[kind] * bg_sd * decay
        elif kind == "covshift":
            # A direction orthogonal to the top-8 principal directions of the
            # clean background: genuinely outside the calibration subspace, i.e.
            # the textbook ASR target. Flagged as ASR-favourable by construction.
            vec = rng.standard_normal(n_channels)
            vec -= top @ (top.T @ vec)
            vec /= np.linalg.norm(vec)
            leakage.append(float(np.linalg.norm(top.T @ vec)))
            wave = _bandpass(rng.standard_normal(n), sfreq, 1.0, 40.0)
            wave /= wave.std()
            contaminated[:, a:b] += AMP[kind] * bg_sd * np.outer(vec, wave)

    meta = {
        "background_sd_uv": bg_sd,
        "cov_direction_leakage": max(leakage) if leakage else None,
        "pop_channel": POP_CHANNEL,
        "n_events": len(events),
        "duration_s": spec.duration_s,
        "sfreq": sfreq,
        "amplitudes_x_background_sd": dict(AMP),
        "guard_s": GUARD_S,
    }
    return clean, contaminated, events, pos, meta


def build_masks(events, n_times: int, sfreq: float):
    """Return disjoint ``(artifact, guard, clean)`` boolean sample masks."""
    artifact = np.zeros(n_times, dtype=bool)
    for a, b in events.values():
        artifact[a:b] = True
    guard_w = int(round(GUARD_S * sfreq))
    dilated = artifact.copy()
    for a, b in events.values():
        dilated[max(0, a - guard_w): min(n_times, b + guard_w)] = True
    guard = dilated & ~artifact
    clean_mask = ~dilated
    if (artifact & clean_mask).any():  # pragma: no cover - guarded by construction
        raise AssertionError("artifact and clean masks overlap")
    return artifact, guard, clean_mask


def rrmse(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Relative RMSE of *estimate* against *truth*, restricted to *mask*."""
    num = np.linalg.norm(estimate[:, mask] - truth[:, mask])
    den = np.linalg.norm(truth[:, mask])
    return float(num / den)


def mean_channel_corr(estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray):
    """Return ``(mean, worst)`` per-channel Pearson correlation under *mask*."""
    a = estimate[:, mask]
    b = truth[:, mask]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    corr = num / np.maximum(den, np.finfo(float).tiny)
    return float(corr.mean()), float(corr.min())
