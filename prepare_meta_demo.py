"""Build every cached asset the Meta sprint demo notebook needs.

Run this once, the day before the talk::

    python demo/meta_sprint_2026/prepare_meta_demo.py --all

Individual stages::

    python demo/meta_sprint_2026/prepare_meta_demo.py --zapline
    python demo/meta_sprint_2026/prepare_meta_demo.py --asr
    python demo/meta_sprint_2026/prepare_meta_demo.py --dss
    python demo/meta_sprint_2026/prepare_meta_demo.py --movement

Everything is written to ``~/.cache/mne-denoise/meta-demo`` (override with
``MNE_DENOISE_META_DEMO_CACHE``). Only compact derivatives are stored -- no raw
OpenNeuro recordings are copied.

Dataset locations are resolved from environment variables, then from a bounded
scan of the usual data directories. See ``README.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_utils as du  # noqa: E402

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("meta-demo-prep")


# ---------------------------------------------------------------------------
# Shared numerical settings. These are declared ONCE, before any result is seen.
# ---------------------------------------------------------------------------

#: Welch settings used for every spectrum in Act 1.
PSD_KWARGS: dict[str, Any] = dict(method="welch", fmin=1.0, fmax=125.0, n_fft=8192)

#: Peak half-width for the peak-to-surround ratio. The 50 Hz line on ds003620 has
#: a measured FWHM of ~0.24 Hz, so the package default of 2.0 Hz averages the peak
#: together with 16 bins of clean floor and dilutes it. 0.5 Hz is ~2x the observed
#: FWHM. The identical window is applied to every condition, and the ratio is ALSO
#: stored at the package default so the choice is auditable.
PEAK_BW_HZ = 0.5
SURROUND_BW_HZ = 5.0

LINE_FREQ_DS003620 = 50.0  # PowerLineFrequency in the BIDS sidecar (Australia)


def _timed(label: str):
    class _Ctx:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.dt = time.perf_counter() - self.t0
            log.info("    %-46s %6.1f s", label, self.dt)
            return False

    return _Ctx()


# ---------------------------------------------------------------------------
# Act 1 -- adaptive line noise on OpenNeuro ds003620 (Runabout mobile EEG)
# ---------------------------------------------------------------------------


def _ds003620_subjects(root: Path) -> list[tuple[str, Path]]:
    out = []
    for sub in sorted(root.glob("sub-*")):
        vhdr = sub / "eeg" / f"{sub.name}_task-oddball_eeg.vhdr"
        if vhdr.exists():
            out.append((sub.name, vhdr))
    return out


def prepare_zapline(root: Path | None = None) -> None:
    """Act 1: ZapLine+ vs a notch filter on a mobile EEG recording."""
    import mne

    from mne_denoise.qa import below_noise_distortion_db, noise_surround_ratio
    from mne_denoise.zapline import ZapLine

    mne.set_log_level("ERROR")
    log.info("[act 1] line noise -- ds003620 (Runabout mobile EEG)")

    data_root = du.resolve_dataset_root("ds003620", root)
    subjects = _ds003620_subjects(data_root)
    if not subjects:
        raise FileNotFoundError(
            f"No ds003620 recordings with a task-oddball .vhdr under {data_root}."
        )
    log.info("  root      %s", data_root)
    log.info("  subjects  %d available", len(subjects))

    # -- deterministic representative selection, on a BASELINE property only ---
    log.info("  selecting representative recording by BASELINE contamination")
    baseline: dict[str, float] = {}
    for name, vhdr in subjects:
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR").pick("eeg")
        psd = raw.compute_psd(**PSD_KWARGS, verbose="ERROR")
        ratio = noise_surround_ratio(
            psd.freqs, psd.get_data(), LINE_FREQ_DS003620,
            peak_bw=PEAK_BW_HZ, surround_bw=SURROUND_BW_HZ,
        )
        baseline[name] = float(np.median(ratio))
        log.info("    %-8s baseline R(%.0f) = %.3f", name, LINE_FREQ_DS003620, baseline[name])
        del raw, psd

    subject, subject_value, median_value = du.nearest_to_median(baseline)
    selection_rule = (
        "Recording whose whole-recording median peak-to-surround ratio R(50 Hz), "
        "computed on the UNCORRECTED data, is closest to the median of that "
        "statistic across all available recordings. No cleaning result is used "
        "in the selection. Ties break on sorted subject id."
    )
    log.info("  -> representative: %s (R=%.3f, cohort median %.3f)",
             subject, subject_value, median_value)

    vhdr = dict(subjects)[subject]
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR").pick("eeg")
    sfreq = float(raw.info["sfreq"])
    duration = float(raw.times[-1])
    log.info("  %s: %d channels, %.0f s, %.0f Hz", subject, len(raw.ch_names), duration, sfreq)

    # -- the two cleaning paths ------------------------------------------------
    zap_kwargs = dict(sfreq=sfreq, line_freq=LINE_FREQ_DS003620, adaptive=True, n_select="auto")
    with _timed("ZapLine+ (adaptive) fit_transform") as t_zap:
        zap = ZapLine(**zap_kwargs)
        cleaned = zap.fit_transform(raw.copy())
    with _timed("mne notch_filter (comparator)") as t_notch:
        notched = raw.copy().notch_filter(freqs=[LINE_FREQ_DS003620], picks="eeg", verbose="ERROR")

    # -- spectra ---------------------------------------------------------------
    spectra: dict[str, np.ndarray] = {}
    with _timed("power spectra (3 conditions)"):
        for name, obj in (("original", raw), ("notch", notched), ("zapline+", cleaned)):
            psd = obj.compute_psd(**PSD_KWARGS, verbose="ERROR")
            spectra[name] = psd.get_data()
            freqs = psd.freqs

    # -- metrics ---------------------------------------------------------------
    metrics: dict[str, Any] = {
        "subject": subject,
        "line_freq": LINE_FREQ_DS003620,
        "duration_s": duration,
        "sfreq": sfreq,
        "n_channels": len(raw.ch_names),
        "baseline_ratio_by_subject": baseline,
        "cohort_median_baseline_ratio": median_value,
        "peak_bw_hz": PEAK_BW_HZ,
        "surround_bw_hz": SURROUND_BW_HZ,
        "n_removed_total": int(zap.n_removed_),
        "runtime_s": {"zapline": t_zap.dt, "notch": t_notch.dt},
        "ratio": {},
        "ratio_package_default_bw": {},
        "distortion_db": {},
        "peak_power_change_db": {},
    }
    i_line = int(np.argmin(np.abs(freqs - LINE_FREQ_DS003620)))
    p_original = spectra["original"].mean(axis=0)[i_line]
    for name, psd in spectra.items():
        metrics["ratio"][name] = float(np.median(noise_surround_ratio(
            freqs, psd, LINE_FREQ_DS003620, peak_bw=PEAK_BW_HZ, surround_bw=SURROUND_BW_HZ)))
        metrics["ratio_package_default_bw"][name] = float(np.median(noise_surround_ratio(
            freqs, psd, LINE_FREQ_DS003620)))
        metrics["peak_power_change_db"][name] = float(
            10.0 * np.log10(psd.mean(axis=0)[i_line] / p_original))
        if name != "original":
            metrics["distortion_db"][name] = float(np.median(below_noise_distortion_db(
                freqs, spectra["original"], psd, exclude_freq=LINE_FREQ_DS003620,
                exclude_bw=5.0, fmin=1.0, fmax=45.0)))

    log.info("  R(%.0f Hz): %s", LINE_FREQ_DS003620,
             "  ".join(f"{k}={v:.3f}" for k, v in metrics["ratio"].items()))
    log.info("  broadband distortion 1-45 Hz: %s",
             "  ".join(f"{k}={v:.4f} dB" for k, v in metrics["distortion_db"].items()))

    # -- adaptive diagnostics --------------------------------------------------
    chunks = zap.adaptive_results_["chunk_info"]
    chunk_start = np.array([float(c["start"]) / sfreq for c in chunks])
    chunk_end = np.array([float(c["end"]) / sfreq for c in chunks])
    chunk_n = np.array([int(c["n_removed"]) for c in chunks])
    chunk_fine = np.array([float(c["fine_freq"]) for c in chunks])
    chunk_present = np.array([bool(c["artifact_present"]) for c in chunks])
    metrics["n_chunks"] = int(len(chunks))
    metrics["chunk_n_removed_range"] = [int(chunk_n.min()), int(chunk_n.max())]
    metrics["chunks_with_zero_removed"] = int((chunk_n == 0).sum())
    log.info("  adaptive: %d chunks, components removed per chunk %d..%d "
             "(%d chunks removed nothing)",
             len(chunks), chunk_n.min(), chunk_n.max(), (chunk_n == 0).sum())

    # -- time-resolved contamination on the UNCORRECTED recording --------------
    with _timed("time-resolved contamination (60 s windows)"):
        win = 60.0
        times, ratios = [], []
        t = 0.0
        while t + win <= duration:
            seg = raw.copy().crop(t, t + win)
            psd = seg.compute_psd(method="welch", fmin=1.0, fmax=125.0, n_fft=2048,
                                  verbose="ERROR")
            ratios.append(float(np.median(noise_surround_ratio(
                psd.freqs, psd.get_data(), LINE_FREQ_DS003620,
                peak_bw=PEAK_BW_HZ, surround_bw=SURROUND_BW_HZ))))
            times.append(t + win / 2.0)
            t += win
    contamination_t = np.asarray(times)
    contamination_r = np.asarray(ratios)
    metrics["contamination_range"] = [float(contamination_r.min()), float(contamination_r.max())]

    # -- a short crop so the notebook can run the real estimator live ----------
    # Chosen by the same rule as the recording: the 60 s window whose UNCORRECTED
    # contamination is nearest the median across all windows. Never the window
    # where ZapLine+ happens to do best.
    live_idx = int(np.argmin(np.abs(contamination_r - np.median(contamination_r))))
    live_start = float(contamination_t[live_idx] - win / 2.0)
    live = raw.copy().crop(live_start, min(live_start + win, duration))
    live.save(du.cache_path("zapline_demo_raw.fif"), overwrite=True, verbose="ERROR")
    metrics["live_crop"] = {
        "start_s": live_start,
        "duration_s": win,
        "window_ratio": float(contamination_r[live_idx]),
        "median_window_ratio": float(np.median(contamination_r)),
        "rule": "60 s window nearest the median uncorrected R(50 Hz)",
    }
    log.info("  live crop: [%.0f, %.0f] s (window R=%.2f, median %.2f)",
             live_start, live_start + win, contamination_r[live_idx],
             np.median(contamination_r))

    du.save_npz(
        du.cache_path("zapline_spectra.npz"),
        freqs=freqs,
        psd_original=spectra["original"],
        psd_notch=spectra["notch"],
        psd_zapline=spectra["zapline+"],
    )
    du.save_npz(
        du.cache_path("zapline_adaptive.npz"),
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        chunk_n_removed=chunk_n,
        chunk_fine_freq=chunk_fine,
        chunk_artifact_present=chunk_present,
        contamination_times=contamination_t,
        contamination_ratio=contamination_r,
    )
    du.save_json(du.cache_path("zapline_metrics.json"), metrics)
    du.write_manifest("zapline", du.build_manifest(
        act="zapline",
        dataset="ds003620",
        subject=subject,
        preprocessing={
            "picks": "eeg",
            "filtering": "none (raw BrainVision as distributed)",
            "reference": "as recorded (FCz)",
            "psd": {k: v for k, v in PSD_KWARGS.items()},
        },
        estimators={
            "ZapLine": zap_kwargs,
            "notch": {"method": "mne.io.Raw.notch_filter", "freqs": [LINE_FREQ_DS003620],
                      "params": "MNE defaults"},
            "metric": {"peak_bw_hz": PEAK_BW_HZ, "surround_bw_hz": SURROUND_BW_HZ,
                       "rationale": "measured line FWHM ~0.24 Hz; identical window for all "
                                    "conditions; package-default value also stored"},
        },
        selection_rule=selection_rule,
        notes="Line frequency taken from the BIDS sidecar (PowerLineFrequency=50).",
    ))
    log.info("[act 1] done\n")


# ---------------------------------------------------------------------------
# Act 2 -- ASR on a known-clean transient-artifact fixture
# ---------------------------------------------------------------------------

#: Minimum calibration samples per channel dimension. A covariance estimate over
#: ``n`` channels needs on the order of 20 samples per dimension to be usable;
#: the fixture length is chosen to satisfy this, NOT by inspecting any endpoint.
MIN_CALIBRATION_SAMPLES_PER_DIM = 20

#: Fixture lengths. The 60 s fixture is the headline case. The 20 s fixture is
#: kept as a diagnostic: it starves auto-calibration, and ``calibration_info_``
#: shows exactly that -- which is the point of an inspectable fitted state.
FIXTURE_DURATIONS_S = (60.0, 20.0)

#: ASR documents that it expects high-pass filtered data and warns otherwise.
#: 1 Hz is the value used by ``clean_rawdata`` and by ds004505's own pipeline.
#: Declared before any endpoint was computed.
ASR_HIGHPASS_HZ = 1.0


def _score_asr(clean, contaminated, cleaned, masks, events, sfreq):
    from _asr_fixture import mean_channel_corr, rrmse

    artifact, _guard, clean_mask = masks
    mean_corr, worst_corr = mean_channel_corr(cleaned, clean, clean_mask)
    per_event = {}
    for label, (a, b) in events.items():
        m = np.zeros(clean.shape[1], dtype=bool)
        m[a:b] = True
        kind = "".join(ch for ch in label if not ch.isdigit())
        per_event.setdefault(kind, []).append(
            (rrmse(contaminated, clean, m), rrmse(cleaned, clean, m))
        )
    return {
        "artifact_rrmse_before": rrmse(contaminated, clean, artifact),
        "artifact_rrmse_after": rrmse(cleaned, clean, artifact),
        "clean_rrmse_before": rrmse(contaminated, clean, clean_mask),
        "clean_rrmse_after": rrmse(cleaned, clean, clean_mask),
        "clean_corr_mean": mean_corr,
        "clean_corr_worst": worst_corr,
        "clean_variance_retained": float(
            np.var(cleaned[:, clean_mask]) / np.var(clean[:, clean_mask])
        ),
        "per_event_rrmse": {
            k: {"before": float(np.mean([x[0] for x in v])),
                "after": float(np.mean([x[1] for x in v]))}
            for k, v in per_event.items()
        },
    }


def prepare_asr(_root: Path | None = None) -> None:
    """Act 2: ASR against a known-clean ground truth."""
    import mne

    from mne_denoise.asr import ASR

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _asr_fixture import CH_NAMES, FixtureSpec, build_fixture, build_masks

    mne.set_log_level("ERROR")
    log.info("[act 2] ASR -- synthetic fixture with known clean ground truth")

    results: dict[str, Any] = {}
    payload: dict[str, np.ndarray] = {}
    headline_duration = FIXTURE_DURATIONS_S[0]

    for duration in FIXTURE_DURATIONS_S:
        spec = FixtureSpec(duration_s=duration, random_state=du.RANDOM_STATE)
        clean, contaminated, events, _pos, meta = build_fixture(spec)
        masks = build_masks(events, spec.n_times, spec.sfreq)
        artifact, guard, clean_mask = masks
        log.info("  fixture %.0f s: %d events, artifact %.1f%% / guard %.1f%% / clean %.1f%%",
                 duration, len(events),
                 100 * artifact.mean(), 100 * guard.mean(), 100 * clean_mask.mean())

        # ASR documents that it assumes high-pass filtered input, and warns
        # otherwise. Honour that precondition: filter the ground truth and the
        # contaminated data with the SAME filter so the truth stays exact.
        clean = mne.filter.filter_data(
            clean, spec.sfreq, ASR_HIGHPASS_HZ, None, verbose="ERROR")
        contaminated = mne.filter.filter_data(
            contaminated, spec.sfreq, ASR_HIGHPASS_HZ, None, verbose="ERROR")
        meta["highpass_hz"] = ASR_HIGHPASS_HZ

        info = mne.create_info(list(CH_NAMES), spec.sfreq, "eeg")
        info.set_montage("standard_1020", verbose="ERROR")
        with info._unlock():
            info["highpass"] = ASR_HIGHPASS_HZ
        raw = mne.io.RawArray(contaminated * 1e-6, info, verbose="ERROR")

        per_method: dict[str, Any] = {}
        estimators: dict[str, Any] = {}
        for method in ("standard", "riemannian_windowed"):
            asr = ASR(method=method, random_state=du.RANDOM_STATE)
            with _timed(f"ASR({method}) fit_transform [{duration:.0f}s]") as t:
                out = asr.fit_transform(raw.copy())
            cleaned = out.get_data() * 1e6
            scores = _score_asr(clean, contaminated, cleaned, masks, events, spec.sfreq)
            cal = asr.calibration_info_
            n_cal = int(cal["calibration_samples"])
            scores.update(
                runtime_s=t.dt,
                n_calibration_samples=n_cal,
                calibration_samples_per_dim=n_cal / len(CH_NAMES),
                n_clean_windows=int(cal["n_clean_windows"]),
                n_calibration_windows=int(cal["n_calibration_windows"]),
                calibration_fraction=float(cal["n_clean_windows"] / cal["n_calibration_windows"]),
                fraction_samples_repaired=float(asr.sample_mask_.mean()),
                fraction_windows_modified=float(
                    (asr.n_components_reconstructed_ > 0).mean()),
                max_components_reconstructed=int(asr.n_components_reconstructed_.max()),
                sample_mask_recall_on_artifact=float(asr.sample_mask_[artifact].mean()),
                sample_mask_fp_on_clean=float(asr.sample_mask_[clean_mask].mean()),
            )
            per_method[method] = scores
            estimators[method] = asr
            if duration == headline_duration and method == "standard":
                payload.update(
                    times=np.arange(spec.n_times) / spec.sfreq,
                    clean=clean,
                    contaminated=contaminated,
                    cleaned=cleaned,
                    artifact_mask=artifact,
                    clean_mask=clean_mask,
                    n_components_reconstructed=asr.n_components_reconstructed_,
                    sample_mask=asr.sample_mask_,
                )
                window_starts = np.asarray(asr.diagnostics_["window_starts"], dtype=float)
                payload["window_times"] = window_starts / spec.sfreq
                results["headline_event_bounds"] = {
                    k: [int(a), int(b)] for k, (a, b) in events.items()
                }
                results["headline_ch_names"] = list(CH_NAMES)

        # Do the two backends actually differ here?
        std, rie = per_method["standard"], per_method["riemannian_windowed"]
        identical = abs(std["artifact_rrmse_after"] - rie["artifact_rrmse_after"]) < 1e-9
        per_method["backends_numerically_identical"] = bool(identical)

        # Control: the same estimator applied to artifact-free ground truth.
        clean_raw = mne.io.RawArray(clean * 1e-6, info, verbose="ERROR")
        ctrl = ASR(random_state=du.RANDOM_STATE)
        ctrl_out = ctrl.fit_transform(clean_raw)
        ctrl_data = ctrl_out.get_data() * 1e6
        from _asr_fixture import mean_channel_corr as _mcc
        from _asr_fixture import rrmse as _rr
        per_method["control_on_clean_input"] = {
            "clean_rrmse_after": _rr(ctrl_data, clean, clean_mask),
            "clean_corr_mean": _mcc(ctrl_data, clean, clean_mask)[0],
            "fraction_samples_repaired": float(ctrl.sample_mask_.mean()),
            "calibration_samples_per_dim": float(
                ctrl.calibration_info_["calibration_samples"] / len(CH_NAMES)),
        }

        per_method["fixture"] = meta
        results[f"{duration:.0f}s"] = per_method

        cal_per_dim = std["calibration_samples_per_dim"]
        log.info("    calibration %.1f samples/dim (target >= %d) | artifact RRMSE %.4f -> %.4f "
                 "| clean RRMSE %.4f -> %.4f",
                 cal_per_dim, MIN_CALIBRATION_SAMPLES_PER_DIM,
                 std["artifact_rrmse_before"], std["artifact_rrmse_after"],
                 std["clean_rrmse_before"], std["clean_rrmse_after"])
        del estimators

    results["headline_duration_s"] = headline_duration
    results["min_calibration_samples_per_dim"] = MIN_CALIBRATION_SAMPLES_PER_DIM
    results["cutoff"] = float(ASR().cutoff)

    du.save_npz(du.cache_path("asr_fixture.npz"), **payload)
    du.save_json(du.cache_path("asr_metrics.json"), results)
    du.write_manifest("asr", du.build_manifest(
        act="asr",
        dataset=None,
        subject=None,
        preprocessing={
            "source": "synthetic fixture (demo/meta_sprint_2026/_asr_fixture.py)",
            "sfreq": 250.0,
            "n_channels": len(CH_NAMES),
            "durations_s": list(FIXTURE_DURATIONS_S),
        },
        estimators={
            "ASR": {"method": ["standard", "riemannian_windowed"],
                    "cutoff": results["cutoff"],
                    "note": "every kwarg at its package default; no sweep"},
        },
        selection_rule=(
            "No subject selection: the fixture is synthetic with a known clean "
            "ground truth. Fixture length chosen so auto-calibration yields at "
            f"least {MIN_CALIBRATION_SAMPLES_PER_DIM} samples per channel "
            "dimension -- a standard covariance-estimation requirement declared "
            "independently of any endpoint. The 20 s case is retained as a "
            "starved-calibration diagnostic."
        ),
        notes=(
            "The covariance-shift event is constructed orthogonal to the top-8 "
            "principal directions of the clean background, i.e. deliberately "
            "inside ASR's intended regime. The other three events use "
            "anatomically motivated patterns with no such construction."
        ),
    ))
    log.info("[act 2] done\n")


# ---------------------------------------------------------------------------
# Act 2b -- the ASR family: which variant does this recording actually need?
# ---------------------------------------------------------------------------

#: Calibration segment length for the rASR arm. Blum et al. (2019) calibrated on
#: a one-minute resting epoch; 30 s of this fixture carries the same number of
#: artifact events. Declared before any endpoint was computed.
VARIANT_CALIBRATION_S = 30.0

#: Covariance aggregators contrasted in the rASR arm. ``geometric_median`` is the
#: package default for every variant and is the Blum 2019 contribution;
#: ``mean`` is the non-robust comparator that shows what it buys.
VARIANT_COV_ESTIMATORS = ("mean", "geometric_median")

#: The rASR sample recording distributed with the MATLAB reference
#: implementation (Blum et al. 2019): 24-channel Smarting mobile EEG.
SME_RELATIVE = Path("refs/asr/repos/rASRMatlab/sampleData/filtered/sme_1_1.xdf_filt.set")
SME_CROP_S = 120.0
SME_HIGHPASS_HZ = 1.0

#: Reference fractions reported by Kim et al. 2025 on their 205-channel juggling
#: recording. Quoted for context only -- never plotted as if we measured them.
KIM2025_REFERENCE_FRACTIONS = {"standard": 0.09, "juggler-gev": 0.24, "juggler-dbscan": 0.42}


def _calibration_fraction(estimator) -> tuple[float, str]:
    """Fraction of the recording used as a calibration reference, and its kind."""
    info = estimator.calibration_info_
    if "reference_selected_fraction" in info:
        return float(info["reference_selected_fraction"]), "sample"
    n_all = max(int(info["n_calibration_windows"]), 1)
    return float(info["n_clean_windows"]) / n_all, "window"


def prepare_asr_variants(_root: Path | None = None) -> None:
    """Act 2b: what separates the ASR variants, measured rather than asserted."""
    import mne

    from mne_denoise.asr import ASR, JugglerASR
    from mne_denoise.qa import variance_removed

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _asr_fixture import FixtureSpec, build_fixture, build_masks, rrmse

    mne.set_log_level("ERROR")
    log.info("[act 2b] ASR variants -- which one does this recording need?")

    results: dict[str, Any] = {}

    # -- Arm A: rASR's regime -- a contaminated calibration segment ------------
    # One declared change: the estimator calibrates on a stretch that still
    # contains the artifacts, instead of on the ground-truth signal.
    spec = FixtureSpec(duration_s=60.0, random_state=du.RANDOM_STATE)
    clean, contaminated, events, _pos, _meta = build_fixture(spec)
    artifact, _guard, clean_mask = build_masks(events, spec.n_times, spec.sfreq)
    clean = mne.filter.filter_data(clean, spec.sfreq, ASR_HIGHPASS_HZ, None, verbose="ERROR")
    contaminated = mne.filter.filter_data(
        contaminated, spec.sfreq, ASR_HIGHPASS_HZ, None, verbose="ERROR")

    n_cal = int(VARIANT_CALIBRATION_S * spec.sfreq)
    segments = {"clean": clean[:, :n_cal], "contaminated": contaminated[:, :n_cal]}
    arm_a: dict[str, Any] = {
        "calibration_s": VARIANT_CALIBRATION_S,
        "artifact_fraction_in_segment": float(artifact[:n_cal].mean()),
        "rows": [],
    }
    with _timed("arm A: contaminated-calibration sweep"):
        for cov in VARIANT_COV_ESTIMATORS:
            for label, segment in segments.items():
                est = ASR(sfreq=spec.sfreq, cov_estimator=cov)
                est.fit(segment)
                out = np.asarray(est.transform(contaminated.copy()))
                arm_a["rows"].append({
                    "cov_estimator": cov,
                    "calibration": label,
                    "artifact_rrmse": rrmse(out, clean, artifact),
                    "clean_rrmse": rrmse(out, clean, clean_mask),
                    "threshold_median": float(np.median(est.thresholds_)),
                })
        # Does the method= flag change anything at package defaults?
        method_check = {}
        for method in ("standard", "riemannian_windowed"):
            est = ASR(sfreq=spec.sfreq, method=method)
            est.fit(segments["contaminated"])
            out = np.asarray(est.transform(contaminated.copy()))
            method_check[method] = {
                "artifact_rrmse": rrmse(out, clean, artifact),
                "threshold_median": float(np.median(est.thresholds_)),
            }
    arm_a["method_flag_check"] = method_check
    arm_a["methods_identical"] = bool(abs(
        method_check["standard"]["artifact_rrmse"]
        - method_check["riemannian_windowed"]["artifact_rrmse"]) < 1e-9)

    for cov in VARIANT_COV_ESTIMATORS:
        rows = {r["calibration"]: r for r in arm_a["rows"] if r["cov_estimator"] == cov}
        inflation = rows["contaminated"]["threshold_median"] / rows["clean"]["threshold_median"]
        log.info("    cov_estimator=%-17s threshold inflation %+.1f%%  "
                 "artifact RRMSE %.4f -> %.4f",
                 cov, 100 * (inflation - 1),
                 rows["clean"]["artifact_rrmse"], rows["contaminated"]["artifact_rrmse"])
    log.info("    method='standard' and 'riemannian_windowed' identical: %s",
             arm_a["methods_identical"])

    # -- Arm B: calibration supply on the rASR paper's own recording ----------
    # This file ships with the rASR MATLAB reference vendored inside the
    # mne-denoise working tree, so it is only available when the package is
    # installed from a checkout rather than a wheel.
    explicit = os.environ.get("META_DEMO_SME")
    if explicit:
        sme = Path(explicit).expanduser()
    else:
        root = du.mne_denoise_root()
        if root is None:
            raise FileNotFoundError(
                "Act 2b needs the Blum 2019 SME sample recording, which lives in "
                "the mne-denoise working tree, but mne-denoise does not appear to "
                "be installed from a checkout.\n"
                "  Set META_DEMO_SME to the path of sme_1_1.xdf_filt.set, or "
                "META_DEMO_MNE_DENOISE_ROOT to an mne-denoise checkout."
            )
        sme = root / SME_RELATIVE
    if not sme.exists():
        raise FileNotFoundError(
            f"The Blum 2019 SME sample recording is missing: {sme}\n"
            "  It ships with the vendored rASR MATLAB reference under "
            "refs/asr/repos/rASRMatlab/sampleData/. Set META_DEMO_SME to point at "
            "it, or skip this stage."
        )
    raw = mne.io.read_raw_eeglab(sme, preload=True, verbose="ERROR")
    raw.crop(0, SME_CROP_S).filter(SME_HIGHPASS_HZ, None, verbose="ERROR")
    data = raw.get_data()
    log.info("  SME (Blum 2019 sample): %d ch, %.0f Hz, %.0f s",
             len(raw.ch_names), raw.info["sfreq"], raw.times[-1])

    builders = {
        "standard": lambda: ASR(sfreq=raw.info["sfreq"]),
        "rASR": lambda: ASR(sfreq=raw.info["sfreq"], method="riemannian_windowed"),
        "juggler-gev": lambda: JugglerASR(sfreq=raw.info["sfreq"], strategy="gev"),
        "juggler-dbscan": lambda: JugglerASR(sfreq=raw.info["sfreq"], strategy="dbscan"),
    }
    arm_b: dict[str, Any] = {
        "dataset": "SME sme_1_1 (Blum et al. 2019 rASR sample, Smarting mobile EEG)",
        "n_channels": len(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]),
        "crop_s": SME_CROP_S,
        "highpass_hz": SME_HIGHPASS_HZ,
        "rows": [],
        "paper_reference_fractions": KIM2025_REFERENCE_FRACTIONS,
    }
    for name, make in builders.items():
        est = make()
        with _timed(f"arm B: {name}") as t:
            cleaned = np.asarray(est.fit_transform(data.copy()))
        fraction, kind = _calibration_fraction(est)
        arm_b["rows"].append({
            "variant": name,
            "calibration_fraction": fraction,
            "calibration_kind": kind,
            "variance_removed_pct": float(variance_removed(data, cleaned)),
            "runtime_s": t.dt,
        })
        log.info("    %-15s calibration %5.1f%% (%s)  variance removed %5.1f%%",
                 name, 100 * fraction, kind, arm_b["rows"][-1]["variance_removed_pct"])

    results["arm_a_contaminated_calibration"] = arm_a
    results["arm_b_calibration_supply"] = arm_b

    du.save_json(du.cache_path("asr_variants_metrics.json"), results)
    du.write_manifest("asr_variants", du.build_manifest(
        act="asr_variants",
        dataset=None,
        subject="sme_1_1",
        preprocessing={
            "arm_a": "synthetic fixture, 1 Hz high-pass, 30 s calibration segment",
            "arm_b": f"SME sample cropped to {SME_CROP_S:.0f} s, "
                     f"{SME_HIGHPASS_HZ:.0f} Hz high-pass",
        },
        estimators={
            "ASR": "package defaults; cov_estimator contrasted mean vs geometric_median",
            "JugglerASR": "package defaults; strategy contrasted gev vs dbscan",
            "cutoff": "package default for every variant; no sweep",
        },
        selection_rule=(
            "No selection. Arm A changes exactly one declared thing (whether the "
            "calibration segment still contains artifacts). Arm B runs every "
            "variant on the recording distributed with the rASR reference "
            "implementation, cropped deterministically from t=0."
        ),
        notes=(
            "Kim et al. 2025 report reference fractions of 9% (standard), 24% "
            "(GEV) and 42% (DBSCAN) on 205-channel juggling EEG. Those numbers "
            "are quoted for context and were NOT measured here. The repository "
            "has separately documented that the GEV > standard ordering does not "
            "reproduce on synthetic burst substrates "
            "(scripts/run_juggler_parameter_ablation.py)."
        ),
    ))
    log.info("[act 2b] done\n")


# ---------------------------------------------------------------------------
# Act 3 -- trial-average DSS on ERP CORE N170 (faces vs cars)
# ---------------------------------------------------------------------------

# Every number below is declared before any DSS result is inspected.
N170_HP, N170_LP = 0.1, 30.0
N170_SFREQ = 256.0
N170_TMIN, N170_TMAX = -0.2, 0.8
N170_BASELINE = (-0.2, 0.0)
#: N170 measurement window from the ERP CORE reference implementation
#: (Kappenman et al. 2021).
N170_WINDOW = (0.110, 0.150)
N170_EOG = ("HEOG_left", "HEOG_right", "VEOG_lower")
DSS_SEED = 20260101
SPLIT_HALF_REPS_SUBJECT = 50
SPLIT_HALF_REPS_GROUP = 20
CV_SPLITS, CV_REPEATS = 5, 5


def _n170_epochs(subject_dir: Path, subject: str):
    """Minimal, defensible epochs. No ICA, no autoreject, no trial rejection."""
    import mne

    raw = mne.io.read_raw_eeglab(
        subject_dir / "eeg" / f"{subject}_task-N170_eeg.set", preload=True, verbose="ERROR")
    present = {c: "eog" for c in N170_EOG if c in raw.ch_names}
    if present:
        raw.set_channel_types(present)
    raw.set_montage("standard_1020", match_case=False, on_missing="warn", verbose="ERROR")
    raw.filter(N170_HP, N170_LP, picks="all", verbose="ERROR")
    raw.resample(N170_SFREQ, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    code = {}
    for key, val in event_id.items():
        try:
            code[int(key)] = val
        except ValueError:
            continue
    # ERP CORE N170 coding: 1-40 faces, 41-80 cars, 101-180 scrambled.
    face_ids = [code[c] for c in range(1, 41) if c in code]
    car_ids = [code[c] for c in range(41, 81) if c in code]
    ev = events.copy()
    m_face = np.isin(events[:, 2], face_ids)
    m_car = np.isin(events[:, 2], car_ids)
    ev[m_face, 2] = 1
    ev[m_car, 2] = 2
    ev = ev[m_face | m_car]
    return mne.Epochs(
        raw, ev, event_id={"face": 1, "car": 2}, tmin=N170_TMIN, tmax=N170_TMAX,
        baseline=N170_BASELINE, picks="eeg", preload=True, reject=None, proj=False,
        verbose="ERROR",
    )


def _baseline_snr(epochs) -> float:
    """GFP(N170 window) / GFP(baseline) on FACE trials. Reference-independent."""
    evoked = epochs["face"].average()
    times = evoked.times
    gfp = evoked.data.std(axis=0)
    post = (times >= N170_WINDOW[0]) & (times <= N170_WINDOW[1])
    pre = (times >= N170_BASELINE[0]) & (times < N170_BASELINE[1])
    return float(gfp[post].mean() / gfp[pre].mean())


#: Seed and amplitudes for the bias-swap fixture. These match
#: ``deep_dives/04_dss.ipynb`` cell 4 exactly so the talk and the deep dive
#: describe the same data. Declared before any result was inspected: the
#: distractor rhythm is deliberately the STRONGER source.
FIXTURE_SEED = 97
FIXTURE_EVOKED_AMP = 1.1
FIXTURE_ALPHA_AMP = 1.4


def _bias_swap_fixture():
    """Two planted sources: a phase-locked evoked deflection and a stronger alpha."""
    rng = np.random.default_rng(FIXTURE_SEED)
    n_ep, n_ch, n_t, sfreq = 120, 32, 256, 256.0
    times = np.arange(n_t) / sfreq - 0.2
    evoked = -np.exp(-((times - 0.15) ** 2) / 0.002)
    v_evoked = rng.standard_normal(n_ch); v_evoked /= np.linalg.norm(v_evoked)
    v_alpha = rng.standard_normal(n_ch); v_alpha /= np.linalg.norm(v_alpha)
    data = rng.standard_normal((n_ep, n_ch, n_t)) * 0.7
    for e in range(n_ep):
        data[e] += np.outer(v_evoked, evoked) * FIXTURE_EVOKED_AMP
        data[e] += np.outer(v_alpha, np.sin(
            2 * np.pi * 10 * times + rng.uniform(0, 6.28))) * FIXTURE_ALPHA_AMP
    return data, v_evoked, v_alpha, sfreq


def _bias_swap_measurement():
    """Same estimator, same data, three criteria -- what does component 1 find?

    PCA is included as a measurement rather than an assertion: the alpha carries
    more variance, so variance-maximisation should return it.
    """
    import mne

    from mne_denoise.dss import DSS, AverageBias, BandpassBias

    data, v_evoked, v_alpha, sfreq = _bias_swap_fixture()
    # Fit the MNE object, not the raw array: deep_dives/04_dss.ipynb does the
    # same, and the two paths differ in the third decimal. Identical inputs keep
    # the talk and the deep dive quoting identical numbers.
    epochs = mne.EpochsArray(
        data, mne.create_info(data.shape[1], sfreq, "eeg"), tmin=-0.2,
        verbose="ERROR")

    def _score(pattern):
        p = pattern / np.linalg.norm(pattern)
        return {"evoked": float(abs(p @ v_evoked)), "alpha": float(abs(p @ v_alpha))}

    out = {}
    for label, bias in (
        ("AverageBias", AverageBias(axis="epochs")),
        ("BandpassBias", BandpassBias(freq_band=(8.0, 12.0), sfreq=sfreq)),
    ):
        dss = DSS(bias=bias, n_components=4)
        dss.fit(epochs)
        out[label] = _score(dss.patterns_[:, 0])

    flat = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    u, s, _ = np.linalg.svd(flat - flat.mean(axis=1, keepdims=True),
                            full_matrices=False)
    out["PCA"] = _score(u[:, 0])
    out["_explained_variance_ratio"] = (s[:3] ** 2 / np.sum(s ** 2)).tolist()
    out["_amplitudes"] = {"evoked": FIXTURE_EVOKED_AMP, "alpha": FIXTURE_ALPHA_AMP}
    return out


def _ica_equivalence():
    """Does IterativeDSS with a tanh contrast land where FastICA lands?

    The package claims equivalence (``nonlinear.py:646-653``). This measures it
    instead of repeating it, and reports FastICA's own recovery in the same
    table so the gap is visible rather than rounded away.
    """
    from scipy import stats
    from sklearn.decomposition import FastICA

    from mne_denoise.dss import IterativeDSS, TanhMaskDenoiser, beta_tanh

    rng = np.random.default_rng(FIXTURE_SEED)
    n = 4000
    t = np.linspace(0, 8, n)
    sources = np.vstack([
        stats.laplace.rvs(size=n, random_state=1),   # super-Gaussian, sparse
        np.sign(np.sin(3 * t)),                      # square wave, high kurtosis
        np.sin(10 * t),                              # sub-Gaussian
        rng.standard_normal(n),                      # Gaussian
    ])
    sources /= sources.std(axis=1, keepdims=True)
    mixed = rng.standard_normal((8, 4)) @ sources

    dss_src = IterativeDSS(TanhMaskDenoiser(), n_components=4,
                           beta=beta_tanh, random_state=0).fit_transform(mixed)
    ica_src = FastICA(n_components=4, fun="logcosh", random_state=0,
                      max_iter=1000, whiten="unit-variance").fit_transform(mixed.T).T

    def _recovery(rec):
        c = np.abs(np.corrcoef(rec, sources)[: rec.shape[0], rec.shape[0]:])
        return c.max(axis=0).tolist()

    pair = np.abs(np.corrcoef(dss_src, ica_src)[:4, 4:]).max(axis=1)
    return {
        "source_labels": ["laplace", "square", "sinusoid", "gaussian"],
        "iterative_dss_recovery": _recovery(dss_src),
        "fastica_recovery": _recovery(ica_src),
        "matched_pair_r": pair.tolist(),
    }


def _subspace_overlap(a, b):
    """Cosines of the principal angles between two column spaces (1.0 = same)."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)


def _xdawn_projector(fit_data, k):
    """Rank-``k`` Xdawn back-projection matrix, or None if the fit fails.

    A single class is used because ``AverageBias`` also ignores condition. Rows
    of ``filters_`` and ``patterns_`` are components, so the mixing matrix is
    ``patterns_.T`` -- confirmed by checking ``patterns_.T @ filters_ == I`` at
    full rank. Regularisation is fixed at ``ledoit_wolf`` and deliberately NOT
    tuned, so a failed split is recorded, never repaired.
    """
    from mne.decoding import XdawnTransformer

    try:
        xd = XdawnTransformer(n_components=k, reg="ledoit_wolf")
        xd.fit(fit_data, np.zeros(len(fit_data), dtype=int))
    except Exception:
        return None
    return xd.patterns_[0][:k].T @ xd.filters_[0][:k]


def _split_half_reproducibility(data, n_reps, seed):
    """Held-out split-half reproducibility of sensor data vs a DSS projection.

    Per repetition the trials are split in two: one half fits the DSS, the other
    half is split again and used only for evaluation. Nothing that is scored was
    seen by the estimator.

    Three comparators share the rank ``k`` that DSS selected: a plain PCA
    projector, and Xdawn -- the spatial filter MNE-Python already ships for
    exactly this job. Xdawn's regularisation is NOT tuned; it runs at
    ``ledoit_wolf``, which is why its failures are reported as observed rather
    than as a claim about Xdawn.
    """
    from mne_denoise.dss import DSS, AverageBias

    rng = np.random.default_rng(seed)
    n_epochs = data.shape[0]
    out = {"sensor": [], "dss": [], "pca": [], "xdawn": [], "component1": [], "k": []}
    for _ in range(n_reps):
        order = rng.permutation(n_epochs)
        fit_idx, eval_idx = order[: n_epochs // 2], order[n_epochs // 2:]
        half = len(eval_idx) // 2
        a_idx, b_idx = eval_idx[:half], eval_idx[half: 2 * half]

        dss = DSS(bias=AverageBias(axis="epochs"), n_select="auto")
        # DSS's epoched-ndarray convention is (n_channels, n_times, n_epochs).
        dss.fit(data[fit_idx].transpose(1, 2, 0))
        k = int(dss.n_selected_ or 1)
        out["k"].append(k)

        filters, patterns = dss.filters_, dss.patterns_

        def _project(block):
            src = np.einsum("ij,ejt->eit", filters[:k], block)
            return np.einsum("ij,ejt->eit", patterns[:, :k], src)

        # plain PCA control at the same rank, fitted on the same trials
        flat = data[fit_idx].transpose(1, 0, 2).reshape(data.shape[1], -1)
        u, _, _ = np.linalg.svd(flat - flat.mean(axis=1, keepdims=True),
                                full_matrices=False)
        proj = u[:, :k] @ u[:, :k].T

        # Xdawn at the same rank, fitted on the same trials. A single class is
        # used because AverageBias also ignores condition. Rows of filters_ and
        # patterns_ are components, so the back-projection needs patterns_.T --
        # verified by checking patterns_.T @ filters_ == I at full rank.
        proj_xd = _xdawn_projector(data[fit_idx], k)

        arms = [
            ("sensor", data[a_idx].mean(0), data[b_idx].mean(0)),
            ("dss", _project(data[a_idx]).mean(0), _project(data[b_idx]).mean(0)),
            ("pca", proj @ data[a_idx].mean(0), proj @ data[b_idx].mean(0)),
            ("component1",
             np.einsum("j,ejt->et", filters[0], data[a_idx]).mean(0),
             np.einsum("j,ejt->et", filters[0], data[b_idx]).mean(0)),
        ]
        if proj_xd is None:
            out["xdawn"].append(float("nan"))
        else:
            arms.append(("xdawn", proj_xd @ data[a_idx].mean(0),
                         proj_xd @ data[b_idx].mean(0)))

        for key, evk_a, evk_b in arms:
            x, y = np.ravel(evk_a), np.ravel(evk_b)
            out[key].append(float(np.corrcoef(x, y)[0, 1]))

    # Means keep the existing Act 3 numbers bit-identical. Medians are reported
    # alongside because the Xdawn arm is heavy-tailed -- it collapses on a few
    # splits, and a mean alone would hide both the typical case and the failure.
    summary = {k: float(np.nanmean(v)) for k, v in out.items()}
    summary.update({f"{key}_median": float(np.nanmedian(out[key]))
                    for key in ("sensor", "dss", "pca", "xdawn")})
    summary["xdawn_failed"] = int(np.isnan(out["xdawn"]).sum())
    summary["xdawn_worst"] = float(np.nanmin(out["xdawn"])) if any(
        not np.isnan(v) for v in out["xdawn"]) else float("nan")
    return summary


def _condition_auc(data, labels, times, seed):
    """Cross-validated faces-vs-cars AUC, sensor space vs DSS components.

    The DSS is refitted inside every fold on training epochs only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from mne_denoise.dss import DSS, AverageBias

    window = (times >= N170_WINDOW[0]) & (times <= N170_WINDOW[1])
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                 random_state=seed)
    scores = {"sensor": [], "dss": []}
    ks = []
    for train, test in cv.split(data, labels):
        sensor_feat = data[:, :, window].mean(axis=2)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(sensor_feat[train], labels[train])
        scores["sensor"].append(
            roc_auc_score(labels[test], model.decision_function(sensor_feat[test])))

        dss = DSS(bias=AverageBias(axis="epochs"), n_select="auto")
        dss.fit(data[train].transpose(1, 2, 0))
        k = int(dss.n_selected_ or 1)
        ks.append(k)
        src = np.einsum("ij,ejt->eit", dss.filters_[:k], data)
        dss_feat = src[:, :, window].mean(axis=2)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(dss_feat[train], labels[train])
        scores["dss"].append(
            roc_auc_score(labels[test], model.decision_function(dss_feat[test])))
    return (float(np.mean(scores["sensor"])), float(np.mean(scores["dss"])),
            float(np.mean(ks)))


def prepare_dss(root: Path | None = None) -> None:
    """Act 3: trial-average DSS on real evoked data."""
    import mne

    from mne_denoise.dss import DSS, AverageBias

    mne.set_log_level("ERROR")
    log.info("[act 3] DSS -- ERP CORE N170 (faces vs cars)")

    data_root = du.resolve_dataset_root("n170", root)
    subjects = sorted(p.name for p in data_root.glob("sub-*") if (p / "eeg").is_dir())
    log.info("  root      %s", data_root)
    log.info("  subjects  %d available", len(subjects))

    # -- deterministic representative selection on a BASELINE property --------
    snr: dict[str, float] = {}
    cache: dict[str, Any] = {}
    with _timed(f"preprocess {len(subjects)} subjects"):
        for name in subjects:
            # Per-subject epochs are cached so re-running a later stage is fast.
            fif = du.cache_path("n170_epochs", f"{name}-epo.fif")
            if fif.exists():
                epochs = mne.read_epochs(fif, preload=True, verbose="ERROR")
            else:
                epochs = _n170_epochs(data_root / name, name)
                epochs.save(fif, overwrite=True, verbose="ERROR")
            snr[name] = _baseline_snr(epochs)
            cache[name] = epochs
    subject, subject_snr, median_snr = du.nearest_to_median(snr)
    selection_rule = (
        "Subject whose BASELINE evoked SNR -- mean GFP over the 110-150 ms N170 "
        "window divided by mean GFP over the -200-0 ms baseline, on FACE trials "
        "of the uncorrected sensor data -- is closest to the median of that "
        "statistic across all 40 subjects. No DSS output enters the selection."
    )
    log.info("  -> representative: %s (baseline SNR %.3f, cohort median %.3f)",
             subject, subject_snr, median_snr)

    epochs = cache[subject]
    data = epochs.get_data(copy=True)
    labels = (epochs.events[:, 2] == 2).astype(int)  # 0 = face, 1 = car
    times = epochs.times
    log.info("  %s: %d face / %d car, %d channels, %d samples",
             subject, int((labels == 0).sum()), int((labels == 1).sum()),
             data.shape[1], data.shape[2])

    # -- one illustrative fit on all trials (for the figure only) -------------
    with _timed("DSS.fit(Epochs) [illustrative]") as t_fit:
        dss = DSS(bias=AverageBias(axis="epochs"), n_select="auto")
        dss.fit(epochs)
    sources = dss.transform(epochs)
    n_selected = int(dss.n_selected_ or 1)
    log.info("  eigenvalues[:5] = %s ; n_selected_ = %d",
             np.round(dss.eigenvalues_[:5], 4).tolist(), n_selected)

    # -- (A) reproducibility, held-out ----------------------------------------
    with _timed(f"split-half reproducibility ({SPLIT_HALF_REPS_SUBJECT} reps)"):
        repro = _split_half_reproducibility(data, SPLIT_HALF_REPS_SUBJECT, DSS_SEED)
    log.info("  reproducibility r: sensor=%.4f  dss=%.4f  pca(control)=%.4f  comp1=%.4f",
             repro["sensor"], repro["dss"], repro["pca"], repro["component1"])
    log.info("  head-to-head (median r): sensor=%.4f  pca=%.4f  dss=%.4f  xdawn=%.4f",
             repro["sensor_median"], repro["pca_median"], repro["dss_median"],
             repro["xdawn_median"])
    log.info("  xdawn worst split %.4f, hard failures %d/%d -- regularisation NOT tuned",
             repro["xdawn_worst"], repro["xdawn_failed"], SPLIT_HALF_REPS_SUBJECT)

    # -- how much of Xdawn's subspace does DSS actually find? ------------------
    # Answers the obvious maintainer question directly instead of asserting that
    # one method "is" the other.
    proj_ref = _xdawn_projector(data, n_selected)
    if proj_ref is None:
        overlap = []
    else:
        xd_patterns = np.linalg.svd(proj_ref, full_matrices=False)[0][:, :n_selected]
        overlap = _subspace_overlap(dss.patterns_[:, :n_selected], xd_patterns).tolist()
        log.info("  DSS vs Xdawn principal-angle cosines: %s (mean %.3f)",
                 np.round(overlap, 3).tolist(), float(np.mean(overlap)))

    # -- the two framework measurements (synthetic, fast) ----------------------
    with _timed("bias-swap + ICA-equivalence measurements"):
        bias_swap = _bias_swap_measurement()
        ica_equiv = _ica_equivalence()
    log.info("  bias swap: AverageBias->evoked %.3f | BandpassBias->alpha %.3f | "
             "PCA->alpha %.3f",
             bias_swap["AverageBias"]["evoked"], bias_swap["BandpassBias"]["alpha"],
             bias_swap["PCA"]["alpha"])
    log.info("  IterativeDSS(tanh) vs FastICA matched |r|: %s",
             np.round(ica_equiv["matched_pair_r"], 3).tolist())

    # -- (B) held-out condition discriminability ------------------------------
    with _timed(f"condition AUC ({CV_SPLITS}x{CV_REPEATS} folds)"):
        auc_sensor, auc_dss, mean_k = _condition_auc(data, labels, times, DSS_SEED)
    log.info("  faces-vs-cars AUC: sensor=%.4f  dss=%.4f  (delta %+.4f)",
             auc_sensor, auc_dss, auc_dss - auc_sensor)

    # -- group level, every subject, identical pipeline ------------------------
    group: dict[str, Any] = {}
    with _timed(f"group level ({len(subjects)} subjects)"):
        for name in subjects:
            gdata = cache[name].get_data(copy=True)
            glabels = (cache[name].events[:, 2] == 2).astype(int)
            grep = _split_half_reproducibility(gdata, SPLIT_HALF_REPS_GROUP, DSS_SEED)
            gs, gd, gk = _condition_auc(gdata, glabels, cache[name].times, DSS_SEED)
            group[name] = {
                "baseline_snr": snr[name],
                "r_sensor": grep["sensor"], "r_dss": grep["dss"], "r_pca": grep["pca"],
                "r_xdawn": grep["xdawn"],
                "auc_sensor": gs, "auc_dss": gd, "k": gk,
            }

    dr = np.array([g["r_dss"] - g["r_sensor"] for g in group.values()])
    da = np.array([g["auc_dss"] - g["auc_sensor"] for g in group.values()])
    # The honest comparisons: against the controls, not only against doing nothing.
    dp = np.array([g["r_dss"] - g["r_pca"] for g in group.values()])
    dx = np.array([g["r_dss"] - g["r_xdawn"] for g in group.values()])
    ps = np.array([g["r_pca"] - g["r_sensor"] for g in group.values()])
    group_summary = {
        "n_subjects": len(group),
        "reproducibility_gain_median": float(np.median(dr)),
        "reproducibility_gain_iqr": [float(np.percentile(dr, 25)), float(np.percentile(dr, 75))],
        "reproducibility_gain_positive": int((dr > 0).sum()),
        "auc_change_median": float(np.median(da)),
        "auc_change_iqr": [float(np.percentile(da, 25)), float(np.percentile(da, 75))],
        "auc_change_positive": int((da > 0).sum()),
        "dss_over_pca_median": float(np.median(dp)),
        "dss_over_pca_positive": int((dp > 0).sum()),
        "dss_over_xdawn_median": float(np.nanmedian(dx)),
        "dss_over_xdawn_positive": int(np.nansum(dx > 0)),
        "pca_over_sensor_positive": int((ps > 0).sum()),
    }
    log.info("  GROUP (n=%d): median reproducibility gain %+.4f (%d/%d subjects up)",
             len(group), group_summary["reproducibility_gain_median"],
             group_summary["reproducibility_gain_positive"], len(group))
    log.info("  GROUP (n=%d): median AUC change          %+.4f (%d/%d subjects up)",
             len(group), group_summary["auc_change_median"],
             group_summary["auc_change_positive"], len(group))
    log.info("  GROUP: DSS over matched-rank PCA %+.4f (%d/%d) | over Xdawn %+.4f (%d/%d)",
             group_summary["dss_over_pca_median"],
             group_summary["dss_over_pca_positive"], len(group),
             group_summary["dss_over_xdawn_median"],
             group_summary["dss_over_xdawn_positive"], len(group))
    log.info("  GROUP: plain PCA already beats raw sensor space in %d/%d subjects",
             group_summary["pca_over_sensor_positive"], len(group))

    # -- cache -----------------------------------------------------------------
    epochs.save(du.cache_path("dss_demo-epo.fif"), overwrite=True, verbose="ERROR")
    # Best sensor = largest evoked deflection inside the N170 measurement window,
    # not the largest anywhere in the epoch (which lands on a late component).
    win = (times >= N170_WINDOW[0]) & (times <= N170_WINDOW[1])
    best_channel = int(np.argmax(np.abs(epochs.average().data[:, win]).max(axis=1)))
    log.info("  best N170-window sensor: %s", epochs.ch_names[best_channel])
    # One deterministic split-half, cached so the figure can SHOW the metric that
    # Step (A) measures: two independent halves of the trials, averaged.
    rng = np.random.default_rng(DSS_SEED)
    order = rng.permutation(len(data))
    half_a, half_b = order[: len(order) // 2], order[len(order) // 2:]
    du.save_npz(
        du.cache_path("dss_sources.npz"),
        times=times,
        sensor_trials=data[:, best_channel, :],
        sensor_evoked=data[:, best_channel, :].mean(axis=0),
        sensor_half_a=data[half_a, best_channel, :].mean(axis=0),
        sensor_half_b=data[half_b, best_channel, :].mean(axis=0),
        component_trials=sources[:, 0, :],
        component_evoked=sources[:, 0, :].mean(axis=0),
        component_half_a=sources[half_a, 0, :].mean(axis=0),
        component_half_b=sources[half_b, 0, :].mean(axis=0),
        pattern=dss.patterns_[:, 0],
        eigenvalues=dss.eigenvalues_,
        labels=labels,
    )
    du.save_json(du.cache_path("dss_metrics.json"), {
        "subject": subject,
        "baseline_snr_by_subject": snr,
        "cohort_median_baseline_snr": median_snr,
        "best_channel": epochs.ch_names[best_channel],
        "n_selected": n_selected,
        "eigenvalues": dss.eigenvalues_.tolist(),
        "reproducibility": repro,
        "subspace_overlap_dss_xdawn": overlap,
        "bias_swap": bias_swap,
        "ica_equivalence": ica_equiv,
        "auc": {"sensor": auc_sensor, "dss": auc_dss, "mean_k": mean_k},
        "n_face": int((labels == 0).sum()),
        "n_car": int((labels == 1).sum()),
        "runtime_s": {"fit": t_fit.dt},
        "n170_window_s": list(N170_WINDOW),
    })
    du.save_json(du.cache_path("dss_group.json"),
                 {"summary": group_summary, "per_subject": group})
    du.write_manifest("dss", du.build_manifest(
        act="dss",
        dataset="n170",
        subject=subject,
        preprocessing={
            "bandpass_hz": [N170_HP, N170_LP],
            "resample_hz": N170_SFREQ,
            "epochs_s": [N170_TMIN, N170_TMAX],
            "baseline_s": list(N170_BASELINE),
            "reference": "average",
            "picks": "eeg",
            "rejection": "none (no ICA, no autoreject, no trial rejection)",
            "event_coding": "ERP CORE N170: 1-40 faces, 41-80 cars; scrambled ignored",
        },
        estimators={
            "DSS": {"bias": "AverageBias(axis='epochs')", "n_select": "auto"},
            "reproducibility": {
                "method": "split-half, DSS fitted on a held-out half",
                "reps_subject": SPLIT_HALF_REPS_SUBJECT,
                "reps_group": SPLIT_HALF_REPS_GROUP,
                "control": "plain PCA at the same rank",
                "comparator": (
                    "mne.decoding.XdawnTransformer at the same rank, single "
                    "class, reg='ledoit_wolf'. Regularisation deliberately NOT "
                    "tuned -- a poor split is reported as observed, never as a "
                    "claim that Xdawn is unstable."
                ),
            },
            "bias_swap": {
                "fixture": (
                    f"synthetic, seed {FIXTURE_SEED}, 120 trials x 32 ch; a "
                    f"phase-locked evoked source at amplitude "
                    f"{FIXTURE_EVOKED_AMP} and a non-phase-locked 10 Hz rhythm "
                    f"at {FIXTURE_ALPHA_AMP} -- the distractor is the STRONGER "
                    "source, declared before any result was inspected"
                ),
                "scored_by": "|cos| of component 1 against each planted pattern",
                "arms": "DSS(AverageBias), DSS(BandpassBias 8-12 Hz), plain PCA",
            },
            "ica_equivalence": {
                "claim_under_test": (
                    "mne_denoise/dss/nonlinear.py:646-653 -- IterativeDSS is "
                    "'equivalent to FastICA when using ICA contrast functions'"
                ),
                "method": (
                    "4 known sources (laplace, square, sinusoid, gaussian) mixed "
                    "into 8 channels; IterativeDSS(TanhMaskDenoiser, beta_tanh) "
                    "vs sklearn FastICA(logcosh). Both recoveries reported."
                ),
            },
            "discriminability": {
                "cv": f"RepeatedStratifiedKFold({CV_SPLITS}x{CV_REPEATS})",
                "feature": f"mean amplitude {N170_WINDOW[0]*1000:.0f}-"
                           f"{N170_WINDOW[1]*1000:.0f} ms",
                "model": "StandardScaler + LogisticRegression",
                "leakage_control": "DSS refitted inside every fold on training epochs only",
            },
        },
        selection_rule=selection_rule,
        notes=(
            "ERP CORE N170 is the EEG homologue of the M170 arm in the working "
            "draft; ds000117 is not available offline on this machine."
        ),
    ))
    log.info("[act 3] done\n")


# ---------------------------------------------------------------------------
# Act 4 -- attenuation vs preservation on real movement data (ds004505)
# ---------------------------------------------------------------------------

MOVE_CROP_S = 575.0  # longest boundary-free run in sub-01 admits exactly one crop
MOVE_MUSCLE_BAND = (30.0, 55.0)
MOVE_ALPHA_BAND = (8.0, 17.0)
#: Flanking bands used to normalise the alpha change. Without this the "neural"
#: endpoint cannot tell a selective alpha loss from a uniform broadband gain
#: change -- a confound found by adversarial review of the first draft.
MOVE_FLANK_BANDS = ((5.0, 7.0), (18.0, 25.0))
MOVE_COUPLING_WIN_S = 1.0
MOVE_SURROGATE_SHIFTS_S = (37.0, 71.0, 113.0, 149.0, 191.0, 233.0, 277.0, 313.0, 359.0)
#: Circular shift applied to the reference block for the negative control.
MOVE_SCRAMBLE_SHIFT_S = 100.0
#: A CCA over 120 primary + 120 reference channels has 240 dimensions. The 2 s
#: default window supplies 2.08 samples per dimension; this widens the stats
#: window so the same fit gets ~31 instead. Declared, not tuned.
MOVE_WIDE_STATS_S = 30.0
MOVE_POSTERIOR = ("O1", "Oz", "O2", "PO3", "PO4", "POz", "PO7", "PO8", "P1", "P2",
                  "P3", "P4", "P5", "P6", "P7", "P8", "Pz")


def _band_power_series(data, sfreq, lo, hi, win_s=MOVE_COUPLING_WIN_S):
    """log10 band power in non-overlapping windows -> (n_channels, n_windows)."""
    import mne

    filtered = mne.filter.filter_data(
        np.atleast_2d(np.asarray(data, float)), sfreq, lo, hi, verbose="ERROR")
    w = int(round(win_s * sfreq))
    n_w = filtered.shape[-1] // w
    block = filtered[:, : n_w * w].reshape(filtered.shape[0], n_w, w)
    return np.log10((block**2).mean(-1) + 1e-30)


def _zrows(a):
    a = np.atleast_2d(a)
    return (a - a.mean(-1, keepdims=True)) / np.maximum(a.std(-1, keepdims=True), 1e-30)


def _coupling(eeg, sfreq, reference_series, lo, hi):
    """Mean |r| between each channel's log band power and a reference series."""
    e = _zrows(_band_power_series(eeg, sfreq, lo, hi))
    r = (e @ _zrows(reference_series).ravel()) / e.shape[1]
    return r


def _band_db(psd_before, psd_after, freqs, band):
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    return 10.0 * np.log10(psd_after[:, m].mean(axis=1) / psd_before[:, m].mean(axis=1))


def prepare_movement(root: Path | None = None) -> None:
    """Act 4: does artifact attenuation cost the neural endpoint?"""
    import mne

    from mne_denoise.asr import ASR
    from mne_denoise.icanclean import ICanClean

    mne.set_log_level("ERROR")
    log.info("[act 4] movement -- ds004505 Table Tennis (sub-01)")

    data_root = du.resolve_dataset_root("ds004505", root)
    set_path = data_root / "sub-01" / "eeg" / "sub-01_task-TableTennis_eeg.set"
    with _timed("read_raw_eeglab (header)"):
        raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")

    # -- channel typing --------------------------------------------------------
    ref_names = [c for c in raw.ch_names if c.startswith("N-")]
    emg_names = [c for c in raw.ch_names if c in
                 ("LISCM", "LSSCM", "LSTrap", "LITrap", "RITrap", "RISCM", "RSSCM", "RSTrap")]
    aux_names = [c for c in raw.ch_names
                 if any(c.startswith(p) for p in ("CGY", "CWR", "NGY", "NWR"))
                 or "Acc" in c or "Sync" in c]
    eeg_names = [c for c in raw.ch_names
                 if c not in ref_names and c not in emg_names and c not in aux_names]
    # Everything that is not scalp EEG must be retyped, otherwise picks='eeg'
    # silently sweeps the inertial and sync channels into the ASR covariance.
    raw.set_channel_types({c: "misc" for c in ref_names + aux_names}, verbose="ERROR")
    raw.set_channel_types({c: "emg" for c in emg_names}, verbose="ERROR")
    # The BIDS sidecar documents a 1 Hz high-pass; record it so ASR's
    # high-pass precondition check reflects the data's actual provenance.
    with raw.info._unlock():
        raw.info["highpass"] = max(float(raw.info["highpass"]), 1.0)
    n_eeg_typed = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    log.info("  channels: %d EEG, %d N-* references, %d neck EMG, %d inertial/sync "
             "(picks='eeg' resolves to %d)",
             len(eeg_names), len(ref_names), len(emg_names), len(aux_names), n_eeg_typed)
    if n_eeg_typed != len(eeg_names):  # pragma: no cover - guarded by construction
        raise RuntimeError(
            f"channel typing mismatch: picks='eeg' would clean {n_eeg_typed} channels "
            f"but only {len(eeg_names)} are scalp EEG")

    # -- deterministic crop: the longest recording-boundary-free run -----------
    bounds = [(a["onset"], a["onset"] + a["duration"]) for a in raw.annotations
              if "boundary" in a["description"].lower()]
    edges = [0.0] + [b for pair in bounds for b in pair] + [float(raw.times[-1])]
    runs = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    runs = [r for r in runs if r[1] - r[0] >= MOVE_CROP_S]
    if not runs:
        raise RuntimeError(
            f"No recording-boundary-free run of at least {MOVE_CROP_S} s in {set_path}.")
    start = max(runs, key=lambda r: r[1] - r[0])[0]
    log.info("  %d boundary annotations; %d run(s) >= %.0f s; using [%.1f, %.1f] s",
             len(bounds), len(runs), MOVE_CROP_S, start, start + MOVE_CROP_S)

    with _timed("crop + load"):
        raw = raw.crop(start, start + MOVE_CROP_S).load_data(verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    eeg_idx = [raw.ch_names.index(c) for c in eeg_names]
    emg_idx = [raw.ch_names.index(c) for c in emg_names]
    posterior = [i for i, c in enumerate(eeg_names) if c in MOVE_POSTERIOR]
    log.info("  posterior ROI: %d of %d EEG channels", len(posterior), len(eeg_names))

    all_data = raw.get_data()
    eeg0 = all_data[eeg_idx]
    emg = all_data[emg_idx]

    # -- endpoint validity: is the muscle coupling above its own null? ---------
    emg_series = _band_power_series(emg, sfreq, *MOVE_MUSCLE_BAND).mean(axis=0)
    r0 = _coupling(eeg0, sfreq, emg_series, *MOVE_MUSCLE_BAND)
    e_z = _zrows(_band_power_series(eeg0, sfreq, *MOVE_MUSCLE_BAND))
    null = np.array([
        float(np.abs((e_z @ _zrows(np.roll(emg_series, int(s / MOVE_COUPLING_WIN_S))).ravel())
                     / e_z.shape[1]).mean())
        for s in MOVE_SURROGATE_SHIFTS_S
    ])
    coupling0 = float(np.abs(r0).mean())
    log.info("  scalp<->neck-EMG %.0f-%.0f Hz coupling: |r|=%.4f  (null max %.4f) -> %s",
             *MOVE_MUSCLE_BAND, coupling0, null.max(),
             "USABLE" if coupling0 > null.max() else "NOT USABLE")
    if coupling0 <= null.max():
        raise RuntimeError(
            "The muscle-coupling artifact endpoint is not above its surrogate null on "
            "this segment; refusing to report a trade-off built on it.")

    psd_kw = dict(method="welch", fmin=1.0, fmax=90.0, n_fft=int(4 * sfreq))
    base_psd = raw.copy().pick(eeg_names).compute_psd(**psd_kw, verbose="ERROR")
    freqs = base_psd.freqs
    psd0 = base_psd.get_data()

    # -- comparators, each at its package default / declared operating point ---
    runs_spec: list[tuple[str, str, dict]] = [
        ("ASR", "asr", {"method": "standard"}),
        ("rASR", "asr", {"method": "riemannian_windowed"}),
        ("iCanClean", "icc", {}),
        ("iCanClean\n(scrambled ref)", "icc", {"scramble": True}),
        # The same estimator with enough samples to estimate 240 dimensions
        # honestly. Deep dive 03 uses this pair to show that the default-window
        # attenuation above was overfitting, not a shared artifact.
        ("iCanClean\n(30 s stats window)", "icc", {"stats_segment_len": MOVE_WIDE_STATS_S}),
        ("iCanClean\n(30 s, scrambled ref)", "icc",
         {"stats_segment_len": MOVE_WIDE_STATS_S, "scramble": True}),
    ]

    rows: list[dict[str, Any]] = [{
        "method": "uncorrected",
        "coupling_r": coupling0,
        "attenuation_pct": 0.0,
        "alpha_db": 0.0,
        "alpha_vs_background_db": 0.0,
        "variance_retained": 1.0,
        "components_removed": 0.0,
        "runtime_s": 0.0,
    }]
    traces: dict[str, np.ndarray] = {
        "freqs": freqs,
        "psd_uncorrected": psd0[posterior].mean(axis=0),
    }

    for label, kind, opts in runs_spec:
        work = raw.copy()
        if kind == "asr":
            est = ASR(picks="eeg", random_state=du.RANDOM_STATE, **opts)
            with _timed(f"{label.splitlines()[0]} fit_transform") as t:
                out = est.fit_transform(work)
            removed = float(np.mean(est.n_components_reconstructed_))
            extra = {
                "windows_touched": float((est.n_components_reconstructed_ > 0).mean()),
                "samples_repaired": float(est.sample_mask_.mean()),
            }
        else:
            if opts.get("scramble"):
                shift = int(MOVE_SCRAMBLE_SHIFT_S * sfreq)
                data = work.get_data()
                ridx = [work.ch_names.index(c) for c in ref_names]
                data[ridx] = np.roll(data[ridx], shift, axis=1)
                work = mne.io.RawArray(data, work.info, verbose="ERROR")
            icc_kw = {"sfreq": sfreq, "ref_channels": ref_names,
                      "primary_channels": eeg_names}
            if opts.get("stats_segment_len"):
                icc_kw["stats_segment_len"] = opts["stats_segment_len"]
            est = ICanClean(**icc_kw)
            with _timed(f"{label.splitlines()[0]} fit_transform") as t:
                out = est.fit_transform(work)
            removed = float(np.mean(est.n_removed_))
            stats_len = opts.get("stats_segment_len") or est.segment_len
            extra = {
                "n_windows": int(est.n_windows_),
                "mean_r2": float(np.mean(est.correlations_)),
                "stats_segment_len_s": float(stats_len),
                "samples_per_dimension": float(
                    stats_len * sfreq / (len(eeg_names) + len(ref_names))),
            }

        clean = out.copy().pick(eeg_names)
        psd1 = clean.compute_psd(**psd_kw, verbose="ERROR").get_data()
        eeg1 = out.get_data()[[out.ch_names.index(c) for c in eeg_names]]

        r1 = _coupling(eeg1, sfreq, emg_series, *MOVE_MUSCLE_BAND)
        coupling1 = float(np.abs(r1).mean())
        alpha = float(np.median(_band_db(psd0, psd1, freqs, MOVE_ALPHA_BAND)[posterior]))
        flank = float(np.mean([
            np.median(_band_db(psd0, psd1, freqs, b)[posterior]) for b in MOVE_FLANK_BANDS
        ]))
        row = {
            "method": label,
            "coupling_r": coupling1,
            "attenuation_pct": 100.0 * (1.0 - coupling1 / coupling0),
            "alpha_db": alpha,
            "alpha_vs_background_db": alpha - flank,
            "background_db": flank,
            "variance_retained": float(np.var(eeg1) / np.var(eeg0)),
            "components_removed": removed,
            "runtime_s": t.dt,
            **extra,
        }
        rows.append(row)
        traces[f"psd_{label.splitlines()[0].lower()}"] = psd1[posterior].mean(axis=0)
        log.info("  %-22s coupling |r| %.4f (%+5.1f%%) | alpha %+.2f dB, vs background "
                 "%+.2f dB | var kept %.3f | comps %.2f",
                 label.replace("\n", " "), coupling1, row["attenuation_pct"],
                 alpha, row["alpha_vs_background_db"], row["variance_retained"], removed)
        del work, out, clean

    du.save_npz(du.cache_path("movement_traces.npz"), **traces)
    du.save_json(du.cache_path("movement_metrics.json"), {
        "subject": "sub-01",
        "crop_s": [start, start + MOVE_CROP_S],
        "n_eeg": len(eeg_names), "n_reference": len(ref_names), "n_emg": len(emg_names),
        "posterior_roi": [eeg_names[i] for i in posterior],
        "baseline_coupling": coupling0,
        "coupling_null_max": float(null.max()),
        "muscle_band_hz": list(MOVE_MUSCLE_BAND),
        "alpha_band_hz": list(MOVE_ALPHA_BAND),
        "flank_bands_hz": [list(b) for b in MOVE_FLANK_BANDS],
        "rows": rows,
    })
    du.write_manifest("movement", du.build_manifest(
        act="movement",
        dataset="ds004505",
        subject="sub-01",
        preprocessing={
            "crop_s": [start, start + MOVE_CROP_S],
            "filtering": "as distributed (1 Hz highpass + Cleanline per the BIDS sidecar)",
            "channel_types": {"eeg": len(eeg_names), "misc(N-* reference)": len(ref_names),
                              "emg": len(emg_names)},
        },
        estimators={
            "ASR": {"picks": "eeg", "method": "standard", "cutoff": "package default"},
            "rASR": {"picks": "eeg", "method": "riemannian_windowed"},
            "ICanClean": {"ref_channels": "the 120 N-* dual-layer electrodes",
                          "other": "package defaults"},
            "negative_control": f"reference block circularly shifted by "
                                f"{MOVE_SCRAMBLE_SHIFT_S:.0f} s",
            "artifact_endpoint": "mean |r| between per-channel log 30-55 Hz band power "
                                 "and neck-EMG 30-55 Hz band power, 1 s windows; the EMG "
                                 "channels were never given to any method",
            "neural_endpoint": "posterior 8-17 Hz power change MINUS the mean change in "
                               "the 5-7 and 18-25 Hz flanking bands, so a uniform "
                               "broadband gain change scores zero",
        },
        selection_rule=(
            "Only sub-01 of ds004505 is available offline. The analysed segment is the "
            f"first {MOVE_CROP_S:.0f} s of the longest recording-boundary-free run, which "
            "is the only run long enough to admit it -- the choice has no free parameter."
        ),
        notes=(
            "n=1 participant: this act illustrates a measurement principle, not a group "
            "result. The scrambled-reference run is a negative control, not a method."
        ),
    ))
    log.info("[act 4] done\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Act 4 -- iCanClean against genuine EOG references (ERP CORE N170)
# ---------------------------------------------------------------------------

# Declared before any iCanClean result was inspected.
EOG_SUBJECT = "sub-005"            # the Act 3 representative, chosen DSS-blind
EOG_REFS = ("HEOG_left", "HEOG_right", "VEOG_lower")
EOG_BLINK_TMIN, EOG_BLINK_TMAX = -0.5, 0.5
EOG_BLINK_BASELINE = (-0.5, -0.3)
#: Window around each blink for CycleAverageBias, in seconds. A blink lasts
#: 200-400 ms; this brackets it. Declared from the physiology, not swept.
EOG_DSS_WINDOW_S = (-0.2, 0.4)
#: Linear DSS is a closed-form eigendecomposition and returns the same answer
#: every time. IterativeDSS is a fixed-point iteration started from a random
#: init, so it does not -- these seeds are run and the spread is reported
#: rather than one of them being picked.
EOG_DSS_SEEDS = (0, 1, 2, 7, 42, 97)
#: The N170 is measured at these sites in the ERP CORE reference implementation.
EOG_N170_ROI = ("PO7", "PO8")
#: Pseudo-reference filters. iCanClean's original MATLAB mode built the CCA
#: reference from a filtered copy of the EEG itself, so no dedicated electrodes
#: are needed. ``('notch', ...)`` matches MATLAB ``filtYtype='Notch'``; the
#: lowpass is the other obvious way to isolate a slow artifact. BOTH are
#: reported -- neither was chosen after seeing the endpoint.
EOG_PSEUDO_FILTERS = (
    ("pseudo-reference\n(notch 8-30 Hz)", 30.0, 8.0),
    ("pseudo-reference\n(lowpass 5 Hz)", None, 5.0),
)


def _eog_raw(data_root: Path):
    """sub-005 with its EOG channels kept -- ``_n170_epochs`` drops them."""
    import mne

    raw = mne.io.read_raw_eeglab(
        data_root / EOG_SUBJECT / "eeg" / f"{EOG_SUBJECT}_task-N170_eeg.set",
        preload=True, verbose="ERROR")
    raw.set_channel_types({c: "eog" for c in EOG_REFS if c in raw.ch_names},
                          verbose="ERROR")
    raw.set_montage("standard_1020", match_case=False, on_missing="warn",
                    verbose="ERROR")
    raw.filter(N170_HP, N170_LP, picks="all", verbose="ERROR")
    raw.resample(N170_SFREQ, verbose="ERROR")
    # EOGRegression refuses to fit without an average-reference projection, and
    # both arms must share one baseline for the percentages to be comparable.
    raw.set_eeg_reference("average", projection=True, verbose="ERROR")
    return raw


def _attach_pseudo_reference(raw, eeg_names, l_freq, h_freq):
    """Build iCanClean's pseudo-reference: a filtered copy of the EEG itself.

    The original MATLAB mode (``filtYtype='Notch'``) needs no dedicated
    electrodes -- it filters the scalp channels so that what survives is mostly
    artifact, and uses that as the CCA reference. The ``pseudo_ref=True``
    shortcut existed in this package (commit 80b02e0) and was removed in the
    PR-26 refactor, so the reference is constructed explicitly here using only
    the shipped API.
    """
    import mne

    filtered = mne.filter.filter_data(
        raw.copy().pick(eeg_names).get_data(), raw.info["sfreq"],
        l_freq, h_freq, verbose="ERROR")
    names = [f"PSEUDO{i:03d}" for i in range(len(eeg_names))]
    extra = mne.io.RawArray(
        filtered, mne.create_info(names, raw.info["sfreq"], "eeg"), verbose="ERROR")
    return raw.copy().add_channels([extra], force_update_info=True), names


def prepare_eog(root: Path | None = None) -> None:
    """Act 4: iCanClean where the reference is real and the dimensions are sane."""
    import mne
    from mne.preprocessing import EOGRegression

    from mne_denoise.dss import (
        DSS,
        CycleAverageBias,
        IterativeDSS,
        TanhMaskDenoiser,
    )
    from mne_denoise.icanclean import ICanClean

    mne.set_log_level("ERROR")
    log.info("[act 4] iCanClean -- ERP CORE N170 blinks vs genuine EOG references")

    data_root = du.resolve_dataset_root("n170", root)
    with _timed(f"read + preprocess {EOG_SUBJECT}"):
        raw = _eog_raw(data_root)
    eog_names = [c for c in raw.ch_names if c in EOG_REFS]
    eeg_names = [c for c in raw.ch_names if c not in eog_names]
    sfreq = float(raw.info["sfreq"])
    dims = len(eeg_names) + len(eog_names)
    samples_per_dim = 2.0 * sfreq / dims
    log.info("  %d EEG + %d EOG = %d dimensions; the 2 s default window gives "
             "%.1f samples per dimension",
             len(eeg_names), len(eog_names), dims, samples_per_dim)

    blinks = mne.preprocessing.find_eog_events(raw, ch_name="VEOG_lower",
                                               verbose="ERROR")
    log.info("  %d blinks detected (%.1f/min over %.0f s)",
             len(blinks), 60.0 * len(blinks) / raw.times[-1], raw.times[-1])

    # The science endpoint: the faces-vs-cars N170 effect. Blinks are not
    # condition-locked, so honest blink removal should leave this alone or
    # sharpen it. Anything that flattens or inverts it removed brain.
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    code = {}
    for key, val in event_id.items():
        try:
            code[int(key)] = val
        except ValueError:
            continue
    face_ids = [code[c] for c in range(1, 41) if c in code]
    car_ids = [code[c] for c in range(41, 81) if c in code]
    n170_ev = events.copy()
    m_face = np.isin(events[:, 2], face_ids)
    m_car = np.isin(events[:, 2], car_ids)
    n170_ev[m_face, 2] = 1
    n170_ev[m_car, 2] = 2
    n170_ev = n170_ev[m_face | m_car]
    roi = [eeg_names.index(c) for c in EOG_N170_ROI if c in eeg_names]

    def _n170_effect(inst):
        ep = mne.Epochs(inst, n170_ev, event_id={"face": 1, "car": 2},
                        tmin=N170_TMIN, tmax=N170_TMAX, baseline=N170_BASELINE,
                        picks=eeg_names, preload=True, reject=None, proj=True,
                        verbose="ERROR")
        t = ep.times
        win = (t >= N170_WINDOW[0]) & (t <= N170_WINDOW[1])
        diff = ep["face"].average().data - ep["car"].average().data
        return float(diff[roi][:, win].mean()) * 1e6

    def _blink_evoked(inst):
        ep = mne.Epochs(inst, blinks, tmin=EOG_BLINK_TMIN, tmax=EOG_BLINK_TMAX,
                        baseline=EOG_BLINK_BASELINE, picks=eeg_names, preload=True,
                        reject=None, proj=True, verbose="ERROR")
        return ep.average()

    evk0 = _blink_evoked(raw)
    base_p2p = float(np.ptp(evk0.data, axis=1).mean()) * 1e6
    # The channel mean of a blink under an average reference is ~zero (the
    # pattern is dipolar), so the figure must show one channel, not the mean.
    # Pick the channel the blink is largest on, before any cleaning.
    worst = int(np.argmax(np.ptp(evk0.data, axis=1)))
    log.info("  blink is largest on %s (%.1f uV peak-to-peak)",
             eeg_names[worst], float(np.ptp(evk0.data[worst])) * 1e6)
    log.info("  uncorrected blink-locked p2p %.2f uV (mean over %d EEG channels)",
             base_p2p, len(eeg_names))

    base_effect = _n170_effect(raw)
    log.info("  uncorrected faces-vs-cars N170 effect at %s: %+.4f uV",
             "/".join(EOG_N170_ROI), base_effect)

    rows: list[dict[str, Any]] = [{
        "method": "uncorrected", "blink_p2p_uv": base_p2p, "attenuation_pct": 0.0,
        "n170_effect_uv": base_effect, "n170_effect_pct": 100.0,
        "information": "none", "reference_kind": "none",
        "components_removed": None, "runtime_s": None,
    }]
    traces: dict[str, Any] = {"times": evk0.times, "uncorrected": evk0.data}

    def _record(label, out, information, kind, removed=None, runtime=None):
        evk = _blink_evoked(out)
        p2p = float(np.ptp(evk.data, axis=1).mean()) * 1e6
        effect = _n170_effect(out)
        rows.append({
            "method": label, "blink_p2p_uv": p2p,
            "attenuation_pct": 100.0 * (1.0 - p2p / base_p2p),
            "n170_effect_uv": effect,
            "n170_effect_pct": 100.0 * effect / base_effect,
            "information": information, "reference_kind": kind,
            "components_removed": removed, "runtime_s": runtime,
        })
        traces[label] = evk.data
        log.info("  %-34s %-14s blink %6.2f uV (%+5.1f%%) | N170 %+.3f uV",
                 label.replace("\n", " "), information, p2p,
                 rows[-1]["attenuation_pct"], effect)

    scalp = raw.copy().pick(eeg_names)

    # -- tier 1: the method is handed the recorded EOG waveform ---------------
    with _timed("iCanClean (EOG electrodes)") as t:
        est = ICanClean(sfreq=sfreq, ref_channels=eog_names,
                        primary_channels=eeg_names)
        out = est.fit_transform(raw.copy())
    _record("iCanClean\n(EOG electrodes)", out, "EOG waveform", "recorded",
            float(est.n_removed_.mean()), t.dt)

    with _timed("EOG regression") as t:
        out = EOGRegression(picks=eeg_names,
                            picks_artifact=eog_names).fit(raw).apply(raw.copy())
    _record("EOG regression", out, "EOG waveform", "recorded", None, t.dt)

    # -- tier 2: only the blink TIMES, never the EOG waveform -----------------
    # Linear DSS: closed-form GED biased toward blink-locked activity.
    with _timed("DSS linear (CycleAverageBias)") as t:
        dss = DSS(bias=CycleAverageBias(blinks[:, 0], window=EOG_DSS_WINDOW_S,
                                        sfreq=sfreq),
                  n_select="auto", return_type="raw")
        out = dss.fit_transform(scalp.copy())
    k_dss = int(dss.n_selected_ or 1)
    _record("DSS linear\n(CycleAverageBias)", out, "blink times", "event-locked",
            float(k_dss), t.dt)

    # Non-linear DSS at the SAME rank: the criterion is re-estimated from the
    # data each iteration instead of being a fixed covariance.
    # NB: no beta= here. beta_tanh accelerates convergence but leaves filters_
    # non-orthogonal, so patterns_ stops being a valid inverse and
    # inverse_transform silently returns garbage (checked: round-trip relative
    # error 1.1 with beta, 7e-15 without).
    data = scalp.get_data()
    half = int(0.5 * sfreq)
    nonlinear_runs = []
    with _timed(f"DSS non-linear (IterativeDSS + tanh, {len(EOG_DSS_SEEDS)} seeds)") as t:
        for seed in EOG_DSS_SEEDS:
            idss = IterativeDSS(TanhMaskDenoiser(), n_components=data.shape[0],
                                random_state=seed)
            sources = idss.fit_transform(data)
            segs = np.stack([sources[:, s - half:s + half] for s in blinks[:, 0]
                             if s - half >= 0 and s + half < sources.shape[1]])
            # Same information as the linear arm: blink times pick the components.
            worst_components = np.argsort(np.ptp(segs.mean(0), axis=1))[::-1][:k_dss]
            kept = sources.copy()
            kept[worst_components] = 0.0
            run = mne.io.RawArray(idss.inverse_transform(kept), scalp.info.copy(),
                                  verbose="ERROR")
            evk = _blink_evoked(run)
            nonlinear_runs.append({
                "seed": seed,
                "attenuation_pct": 100.0 * (1.0 - float(
                    np.ptp(evk.data, axis=1).mean()) * 1e6 / base_p2p),
                "n170_effect_uv": _n170_effect(run),
                "converged_fraction": float(idss.convergence_info_[:, 1].mean()),
                "evoked": evk.data,
            })
    # The reported arm is the seed closest to the median outcome, and the whole
    # spread is cached so the figure can show it. Picking the best seed would be
    # picking a result.
    effects = np.array([r["n170_effect_uv"] for r in nonlinear_runs])
    median_run = nonlinear_runs[int(np.argmin(np.abs(effects - np.median(effects))))]
    log.info("  non-linear DSS across %d seeds: attenuation %.1f%% (sd %.1f), "
             "N170 %+.3f uV (sd %.3f, range %+.3f to %+.3f)",
             len(EOG_DSS_SEEDS),
             float(np.mean([r["attenuation_pct"] for r in nonlinear_runs])),
             float(np.std([r["attenuation_pct"] for r in nonlinear_runs])),
             float(effects.mean()), float(effects.std()),
             float(effects.min()), float(effects.max()))
    rows.append({
        "method": "DSS non-linear\n(IterativeDSS + tanh)",
        "blink_p2p_uv": base_p2p * (1.0 - median_run["attenuation_pct"] / 100.0),
        "attenuation_pct": median_run["attenuation_pct"],
        "n170_effect_uv": median_run["n170_effect_uv"],
        "n170_effect_pct": 100.0 * median_run["n170_effect_uv"] / base_effect,
        "information": "blink times", "reference_kind": "event-locked",
        "components_removed": float(k_dss), "runtime_s": t.dt,
        "seed_spread": {
            "seeds": list(EOG_DSS_SEEDS),
            "reported_seed": median_run["seed"],
            "attenuation_pct": [r["attenuation_pct"] for r in nonlinear_runs],
            "n170_effect_uv": [r["n170_effect_uv"] for r in nonlinear_runs],
            "converged_fraction": [r["converged_fraction"] for r in nonlinear_runs],
        },
    })
    traces["DSS non-linear\n(IterativeDSS + tanh)"] = median_run["evoked"]
    log.info("  %-34s %-14s blink %6.2f uV (%+5.1f%%) | N170 %+.3f uV (seed %d, median)",
             "DSS non-linear (IterativeDSS + tanh)", "blink times",
             rows[-1]["blink_p2p_uv"], rows[-1]["attenuation_pct"],
             rows[-1]["n170_effect_uv"], median_run["seed"])

    # -- tier 3: no external information at all -------------------------------
    for label, lo, hi in EOG_PSEUDO_FILTERS:
        aug, pseudo_names = _attach_pseudo_reference(raw, eeg_names, lo, hi)
        with _timed(label.replace("\n", " ")) as t:
            est = ICanClean(sfreq=sfreq, ref_channels=pseudo_names,
                            primary_channels=eeg_names)
            out = est.fit_transform(aug).pick(eeg_names)
        _record(label, out, "the EEG itself", "pseudo",
                float(est.n_removed_.mean()), t.dt)

    du.save_npz(du.cache_path("eog_traces.npz"), **traces)
    du.save_json(du.cache_path("eog_metrics.json"), {
        "subject": EOG_SUBJECT,
        "n_eeg": len(eeg_names), "n_reference": len(eog_names),
        "dimensions": dims, "samples_per_dimension": samples_per_dim,
        "n_blinks": len(blinks),
        "blinks_per_min": 60.0 * len(blinks) / float(raw.times[-1]),
        "duration_s": float(raw.times[-1]),
        "reference_channels": list(eog_names),
        "dss_rank": k_dss,
        "dss_window_s": list(EOG_DSS_WINDOW_S),
        "baseline_blink_p2p_uv": base_p2p,
        "baseline_n170_effect_uv": base_effect,
        "n170_roi": list(EOG_N170_ROI),
        "n170_window_s": list(N170_WINDOW),
        "blink_channel": eeg_names[worst],
        "blink_channel_index": worst,
        "rows": rows,
    })
    du.write_manifest("eog", du.build_manifest(
        act="eog",
        dataset="n170",
        subject=EOG_SUBJECT,
        preprocessing={
            "bandpass_hz": [N170_HP, N170_LP],
            "resample_hz": N170_SFREQ,
            "reference": "average (projection=True, required by EOGRegression)",
            "eog_channels": list(EOG_REFS),
        },
        estimators={
            "ICanClean": {
                "ref_channels": list(EOG_REFS),
                "params": "package defaults (segment_len=2.0, threshold=0.7)",
                "samples_per_dimension": samples_per_dim,
            },
            "comparator": {
                "name": "mne.preprocessing.EOGRegression",
                "citation": "Gratton, Coles & Donchin (1983)",
                "note": "same reference channels, same baseline, same control",
            },
            "pseudo_reference": {
                "construction": ("a filtered copy of the scalp channels is used "
                                 "as the CCA reference -- no dedicated electrodes"),
                "filters": [f"{lo}-{hi}" for _, lo, hi in EOG_PSEUDO_FILTERS],
                "provenance": ("iCanClean's MATLAB filtYtype='Notch' mode. The "
                               "pseudo_ref=True shortcut was added in mne-denoise "
                               "80b02e0 and removed in the PR-26 refactor, so the "
                               "reference is built explicitly from the shipped API."),
                "note": ("both filter settings are reported; neither was chosen "
                         "after seeing an endpoint"),
            },
            "dss": {
                "linear": (f"DSS(bias=CycleAverageBias(blink_samples, "
                           f"window={EOG_DSS_WINDOW_S}), n_select='auto') -- a "
                           "closed-form GED biased toward blink-locked activity"),
                "nonlinear": ("IterativeDSS(TanhMaskDenoiser()) at the SAME rank; "
                              "components are ranked by blink-locked amplitude, so "
                              "both DSS arms use identical information and differ "
                              "only in how the criterion is solved"),
                "beta_warning": ("beta=beta_tanh is NOT used: it accelerates "
                                 "convergence but leaves filters_ non-orthogonal, "
                                 "so patterns_ stops being a valid inverse and "
                                 "inverse_transform silently returns garbage "
                                 "(round-trip relative error 1.1 with beta, 7e-15 "
                                 "without)"),
                "information": ("both DSS arms see only blink TIMES, never the EOG "
                                "waveform that iCanClean and regression are given"),
            },
            "endpoints": {
                "attenuation": ("blink-locked evoked peak-to-peak, averaged over "
                                "all EEG channels"),
                "preservation": (
                    f"faces-vs-cars N170 effect at {'/'.join(EOG_N170_ROI)} over "
                    f"{N170_WINDOW[0] * 1000:.0f}-{N170_WINDOW[1] * 1000:.0f} ms. "
                    "Blinks are not condition-locked, so honest removal should "
                    "leave this alone or sharpen it. Chosen because it lies INSIDE "
                    "the band these methods touch -- a posterior-alpha endpoint "
                    "sits outside it and cannot see the damage."),
            },
        },
        selection_rule=(
            f"{EOG_SUBJECT} is the Act 3 representative, selected on baseline "
            "evoked SNR before any estimator was run. No iCanClean output enters "
            "the selection."
        ),
        notes=(
            "n=1 participant. Blinks are a strong, physically real coupling with "
            "dedicated reference electrodes, and 33 dimensions leave the CCA well "
            "conditioned -- the opposite of the ds004505 regime in deep dive 03."
        ),
    ))
    log.info("[act 4] done\n")


# ---------------------------------------------------------------------------

STAGES = {
    "zapline": prepare_zapline,
    "asr": prepare_asr,
    "asr-variants": prepare_asr_variants,
    "dss": prepare_dss,
    "eog": prepare_eog,
    "movement": prepare_movement,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare cached assets for the Meta sprint mne-denoise demo.",
    )
    parser.add_argument("--all", action="store_true", help="run every stage")
    for stage in ("zapline", "asr", "asr-variants", "dss", "eog", "movement"):
        parser.add_argument(f"--{stage}", action="store_true", help=f"run the {stage} stage")
    parser.add_argument("--ds003620-root", default=None, help="path to the ds003620 BIDS root")
    parser.add_argument("--n170-root", default=None, help="path to the ERP CORE N170 BIDS root")
    parser.add_argument("--ds004505-root", default=None, help="path to the ds004505 BIDS root")
    parser.add_argument("--check", action="store_true",
                        help="only report which cached assets exist")
    args = parser.parse_args(argv)

    if args.check:
        report = du.check_demo_assets(verbose=True)
        return 0 if report["ok"] else 1

    requested = [
        s for s in ("zapline", "asr", "asr-variants", "dss", "eog", "movement")
        if getattr(args, s.replace("-", "_"))
    ]
    if args.all or not requested:
        requested = list(STAGES)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.perf_counter()
    for stage in requested:
        fn = STAGES.get(stage)
        if fn is None:
            log.warning("stage %r is not implemented yet -- skipping", stage)
            continue
        root = getattr(args, f"{_ROOT_ARG[stage]}_root".replace("-", "_"), None)
        fn(root)
    log.info("total preparation time: %.1f s", time.perf_counter() - t0)

    report = du.check_demo_assets(verbose=True)
    return 0 if report["ok"] else 1


_ROOT_ARG = {
    "zapline": "ds003620",
    "asr": "ds003620",  # unused: the ASR fixture is synthetic
    "asr-variants": "ds003620",  # unused: fixture + the in-repo SME sample
    "dss": "n170",
    "eog": "n170",
    "movement": "ds004505",
}


if __name__ == "__main__":
    raise SystemExit(main())
