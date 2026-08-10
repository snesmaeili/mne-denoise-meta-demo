"""Smoke tests for the Meta sprint demo helpers.

These run without any dataset and without the prepared cache. They cover the
pure helpers, the fixture generator, the manifest/preflight machinery and every
presentation plotting wrapper, so a broken demo is caught by CI rather than on
stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

import demo_utils as du  # noqa: E402
from _asr_fixture import (  # noqa: E402
    CH_NAMES,
    FixtureSpec,
    build_fixture,
    build_masks,
    mean_channel_corr,
    rrmse,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_nearest_to_median_picks_the_median_and_is_deterministic():
    key, value, median = du.nearest_to_median({"c": 5.0, "a": 1.0, "b": 2.0})
    assert (key, value, median) == ("b", 2.0, 2.0)
    # Ties resolve on sorted key order, not dict insertion order.
    assert du.nearest_to_median({"z": 0.0, "a": 2.0})[0] == "a"
    assert du.nearest_to_median({"a": 2.0, "z": 0.0})[0] == "a"


def test_nearest_to_median_rejects_empty():
    with pytest.raises(ValueError):
        du.nearest_to_median({})


def test_zscore_rows_is_scale_only():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = du.zscore_rows(x)
    assert np.allclose(out * np.std(x), x)


def test_cache_path_creates_parents(tmp_path, monkeypatch):
    monkeypatch.setenv("MNE_DENOISE_META_DEMO_CACHE", str(tmp_path / "nested"))
    path = du.cache_path("a", "b", "c.json")
    assert path.parent.is_dir()


def test_json_and_npz_round_trip(tmp_path):
    payload = {"a": np.float64(1.5), "b": np.arange(3), "c": {"d": Path("x")}}
    path = du.save_json(tmp_path / "m.json", payload)
    back = du.load_json(path)
    assert back["a"] == 1.5 and back["b"] == [0, 1, 2] and back["c"]["d"] == "x"

    npz = du.save_npz(tmp_path / "a.npz", x=np.arange(4.0))
    assert np.array_equal(du.load_npz(npz)["x"], np.arange(4.0))


def test_manifest_has_the_required_provenance_fields():
    manifest = du.build_manifest(
        act="zapline", dataset="ds003620", subject="sub-08",
        preprocessing={"picks": "eeg"}, estimators={"ZapLine": {"n_select": "auto"}},
        selection_rule="nearest median baseline contamination",
    )
    for key in ("act", "dataset", "dataset_doi", "subject", "preprocessing",
                "estimators", "selection_rule", "random_state", "environment",
                "created_utc"):
        assert key in manifest, key
    env = manifest["environment"]
    for key in ("python", "mne_denoise", "mne", "numpy", "scipy", "repo_commit"):
        assert env[key], key


def test_dataset_registry_is_complete():
    for key, spec in du.DATASETS.items():
        assert spec["env"].startswith("META_DEMO_"), key
        assert spec["probe"] and spec["label"] and spec["doi"], key


def test_resolve_dataset_root_rejects_unknown_key():
    with pytest.raises(KeyError):
        du.resolve_dataset_root("not-a-dataset")


def test_resolve_dataset_root_reports_a_bad_explicit_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not contain"):
        du.resolve_dataset_root("ds003620", tmp_path)


def test_resolve_dataset_root_accepts_a_good_explicit_path(tmp_path):
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    assert du.resolve_dataset_root("ds003620", tmp_path) == tmp_path


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def test_preflight_reports_missing_assets(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MNE_DENOISE_META_DEMO_CACHE", str(tmp_path))
    report = du.check_demo_assets(verbose=True)
    assert report["imports_ok"], report["import_error"]
    assert not report["ok"]
    assert set(report["acts"]) == {
        "zapline", "asr", "asr_variants", "dss", "movement"}
    assert "NOT READY" in capsys.readouterr().out


def test_assert_presenter_ready_raises_when_incomplete(tmp_path, monkeypatch):
    monkeypatch.setenv("MNE_DENOISE_META_DEMO_CACHE", str(tmp_path))
    with pytest.raises(RuntimeError, match="prepare_meta_demo"):
        du.assert_presenter_ready()


def test_preflight_passes_once_every_asset_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("MNE_DENOISE_META_DEMO_CACHE", str(tmp_path))
    for asset in du.ASSETS:
        for name in asset.files:
            (tmp_path / name).write_bytes(b"")
    report = du.check_demo_assets(verbose=False)
    assert report["ok"]
    du.assert_presenter_ready()


# --------------------------------------------------------------------------
# ASR fixture
# --------------------------------------------------------------------------


def test_fixture_is_reproducible_and_artifacts_are_additive():
    spec = FixtureSpec(duration_s=20.0, random_state=du.RANDOM_STATE)
    clean_a, cont_a, events, pos, meta = build_fixture(spec)
    clean_b, cont_b, *_ = build_fixture(spec)

    assert np.array_equal(clean_a, clean_b)
    assert np.array_equal(cont_a, cont_b)
    assert clean_a.shape == (len(CH_NAMES), spec.n_times)
    assert pos.shape == (len(CH_NAMES), 3)
    assert meta["background_sd_uv"] == pytest.approx(10.0, rel=0.2)
    # The covariance-shift direction really is outside the calibration subspace.
    assert meta["cov_direction_leakage"] < 1e-8

    # Outside the artifact intervals the two arrays must be identical.
    artifact, _guard, clean_mask = build_masks(events, spec.n_times, spec.sfreq)
    assert np.array_equal(cont_a[:, clean_mask], clean_a[:, clean_mask])
    assert np.abs(cont_a[:, artifact] - clean_a[:, artifact]).max() > 0


def test_fixture_masks_are_disjoint_and_cover_everything():
    spec = FixtureSpec(duration_s=20.0, random_state=du.RANDOM_STATE)
    *_, events = build_fixture(spec)[:3]
    artifact, guard, clean_mask = build_masks(events, spec.n_times, spec.sfreq)
    assert not (artifact & clean_mask).any()
    assert not (artifact & guard).any()
    assert not (guard & clean_mask).any()
    assert (artifact | guard | clean_mask).all()
    assert artifact.sum() and guard.sum() and clean_mask.sum()


def test_fixture_tiles_events_across_a_longer_recording():
    short = FixtureSpec(duration_s=20.0)
    long = FixtureSpec(duration_s=60.0)
    assert len(long.expanded_events()) == 3 * len(short.expanded_events())
    # Same artifact fraction regardless of length.
    for spec in (short, long):
        events = build_fixture(spec)[2]
        artifact = build_masks(events, spec.n_times, spec.sfreq)[0]
        assert artifact.mean() == pytest.approx(0.165, abs=0.01)


def test_scoring_helpers():
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((4, 200))
    mask = np.ones(200, dtype=bool)

    assert rrmse(truth, truth, mask) == pytest.approx(0.0)
    mean, worst = mean_channel_corr(truth, truth, mask)
    assert mean == pytest.approx(1.0) and worst == pytest.approx(1.0)

    # A perturbed estimate must score strictly worse on both endpoints.
    noisy = truth + 0.5 * rng.standard_normal(truth.shape)
    assert rrmse(noisy, truth, mask) > 0.1
    assert mean_channel_corr(noisy, truth, mask)[0] < 1.0

    # The mask really does restrict the comparison.
    half = np.zeros(200, dtype=bool)
    half[:100] = True
    broken = truth.copy()
    broken[:, 100:] += 10.0
    assert rrmse(broken, truth, half) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Presentation plotting
# --------------------------------------------------------------------------


def test_line_noise_triptych():
    freqs = np.linspace(40, 65, 400)
    floor = np.full_like(freqs, 1e-12)
    peak = floor + 5e-12 * np.exp(-((freqs - 50.0) ** 2) / 0.02)
    with du.presentation_theme():
        fig = du.plot_line_noise_triptych(
            freqs, {"original": peak, "notch": floor * 0.2, "zapline+": floor},
            line_freq=50.0, ratios={"original": 1.4, "notch": 0.17, "zapline+": 1.0},
        )
    assert fig.axes and fig.axes[0].get_xlim() == (40.0, 65.0)


def test_adaptive_component_timeline():
    starts = np.arange(0.0, 600.0, 60.0)
    with du.presentation_theme():
        fig = du.plot_adaptive_component_timeline(
            starts, np.arange(len(starts)) % 4,
            contamination=(starts, np.linspace(0.9, 2.5, len(starts))),
        )
    assert len(fig.axes) == 2  # main axis plus the twin


def test_asr_reconstruction_panel_respects_the_display_window():
    rng = np.random.default_rng(0)
    times = np.arange(0, 20, 1 / 250)
    data = rng.standard_normal((8, times.size))
    with du.presentation_theme():
        fig = du.plot_asr_reconstruction_panel(
            times, data, data * 0.5, data * 0.4,
            repair=(np.arange(0.0, 20.0, 0.5), np.zeros(40)),
            channels=(0, 1, 2), ch_names=[f"C{i}" for i in range(8)],
            tlim=(2.0, 8.0),
        )
    assert fig.axes[0].get_xlim()[0] == pytest.approx(2.0, abs=0.05)
    assert fig.axes[0].get_xlim()[1] == pytest.approx(8.0, abs=0.05)


def test_dss_target_panel_with_group_scatter():
    times = np.linspace(-0.2, 0.8, 257)
    wave = np.sin(2 * np.pi * 3 * times)
    with du.presentation_theme():
        fig = du.plot_dss_target_panel(
            times, wave, wave * 1.01, wave * 0.9, wave * 0.91,
            group={"reproducibility_gain": [0.01, 0.05, -0.01],
                   "auc_change": [0.02, -0.03, 0.01]},
            labels={"sensor": "sensor PO7", "dss": "DSS component 1"},
            window=(0.11, 0.15),
        )
    assert len(fig.axes) == 2


def test_dss_framework_panel_draws_both_arguments():
    bias_swap = {
        "PCA": {"evoked": 0.07, "alpha": 1.00},
        "AverageBias": {"evoked": 0.99, "alpha": 0.03},
        "BandpassBias": {"evoked": 0.06, "alpha": 1.00},
        "_amplitudes": {"evoked": 1.1, "alpha": 1.4},
    }
    head = {"sensor_median": 0.9675, "pca_median": 0.9712,
            "dss_median": 0.9747, "xdawn_median": 0.9787}
    group = {"n_subjects": 40, "dss_over_pca_positive": 32,
             "dss_over_xdawn_positive": 15}
    with du.presentation_theme():
        fig = du.plot_dss_framework_panel(bias_swap, head, group=group)
    assert len(fig.axes) == 2
    # Left panel: three criteria x two planted patterns.
    assert len(fig.axes[0].patches) == 6
    # Right panel: the comparator that already exists must be on screen.
    assert len(fig.axes[1].patches) == 4
    assert "Xdawn" in " ".join(t.get_text() for t in fig.axes[1].get_xticklabels())


def test_dss_cache_reports_every_comparator():
    """The honest baselines must survive in the cache, not just in prose."""
    metrics = du.load_json(du.cache_path("dss_metrics.json"))
    repro = metrics["reproducibility"]
    for key in ("sensor", "pca", "dss", "xdawn"):
        assert f"{key}_median" in repro, f"missing {key}_median"
    summary = du.load_json(du.cache_path("dss_group.json"))["summary"]
    for key in ("dss_over_pca_positive", "dss_over_xdawn_positive",
                "pca_over_sensor_positive"):
        assert key in summary, f"missing {key}"
    # Panel A only makes its point if PCA really does return the distractor.
    swap = metrics["bias_swap"]
    assert swap["PCA"]["alpha"] > swap["PCA"]["evoked"]
    assert swap["AverageBias"]["evoked"] > swap["AverageBias"]["alpha"]
    assert swap["BandpassBias"]["alpha"] > swap["BandpassBias"]["evoked"]


def test_attenuation_preservation_includes_every_row():
    rows = [
        {"method": "uncorrected", "attenuation_pct": 0.0, "alpha_vs_background_db": 0.0,
         "variance_retained": 1.0, "components_removed": 0.0},
        {"method": "ASR", "attenuation_pct": -2.4, "alpha_vs_background_db": 0.0,
         "variance_retained": 0.955, "components_removed": 0.03},
        {"method": "iCanClean", "attenuation_pct": 11.5, "alpha_vs_background_db": 0.13,
         "variance_retained": 0.781, "components_removed": 10.9},
        {"method": "iCanClean\n(scrambled ref)", "attenuation_pct": 10.5,
         "alpha_vs_background_db": 0.20, "variance_retained": 0.779,
         "components_removed": 10.8},
    ]
    with du.presentation_theme():
        fig = du.plot_attenuation_preservation(rows)
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == len(rows)
    # Negative attenuation must stay inside the axis, not be clipped away.
    assert ax.get_xlim()[0] < -2.4


def _variant_arms():
    arm_a = {
        "calibration_s": 30.0,
        "artifact_fraction_in_segment": 0.143,
        "methods_identical": True,
        "rows": [
            {"cov_estimator": "mean", "calibration": "clean",
             "artifact_rrmse": 0.157, "clean_rrmse": 0.160, "threshold_median": 4.247},
            {"cov_estimator": "mean", "calibration": "contaminated",
             "artifact_rrmse": 0.173, "clean_rrmse": 0.160, "threshold_median": 5.094},
            {"cov_estimator": "geometric_median", "calibration": "clean",
             "artifact_rrmse": 0.160, "clean_rrmse": 0.160, "threshold_median": 4.164},
            {"cov_estimator": "geometric_median", "calibration": "contaminated",
             "artifact_rrmse": 0.173, "clean_rrmse": 0.160, "threshold_median": 4.524},
        ],
    }
    arm_b = {
        "dataset": "SME sme_1_1",
        "paper_reference_fractions": {"standard": 0.09, "juggler-gev": 0.24,
                                      "juggler-dbscan": 0.42},
        "rows": [
            {"variant": "standard", "calibration_fraction": 0.738,
             "calibration_kind": "window", "variance_removed_pct": 52.1, "runtime_s": 2.0},
            {"variant": "rASR", "calibration_fraction": 0.738,
             "calibration_kind": "window", "variance_removed_pct": 50.8, "runtime_s": 2.0},
            {"variant": "juggler-gev", "calibration_fraction": 0.238,
             "calibration_kind": "sample", "variance_removed_pct": 71.5, "runtime_s": 1.5},
            {"variant": "juggler-dbscan", "calibration_fraction": 0.659,
             "calibration_kind": "sample", "variance_removed_pct": 50.1, "runtime_s": 13.0},
        ],
    }
    return arm_a, arm_b


def test_asr_variant_regimes_renders_both_panels():
    arm_a, arm_b = _variant_arms()
    with du.presentation_theme():
        fig = du.plot_asr_variant_regimes(arm_a, arm_b)
    assert len(fig.axes) == 2
    # One y tick per variant, and the calibration axis stays a percentage.
    assert len(fig.axes[1].get_yticklabels()) == len(arm_b["rows"])
    assert fig.axes[1].get_xlim()[0] == 0


def test_asr_variant_regimes_without_paper_reference():
    arm_a, arm_b = _variant_arms()
    with du.presentation_theme():
        fig = du.plot_asr_variant_regimes(arm_a, arm_b, show_paper_reference=False)
    assert fig.axes[1].get_legend() is None


def test_asr_variant_regimes_tolerates_a_missing_paper_entry():
    arm_a, arm_b = _variant_arms()
    arm_b = {**arm_b, "paper_reference_fractions": {"standard": 0.09}}
    with du.presentation_theme():
        du.plot_asr_variant_regimes(arm_a, arm_b)


def test_contract_screen():
    with du.presentation_theme():
        fig = du.plot_contract_screen()
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    for name in ("ZapLine", "ASR", "iCanClean", "DSS"):
        assert name in texts


def test_presentation_theme_restores_rcparams():
    before = dict(plt.rcParams)
    with du.presentation_theme():
        assert plt.rcParams["font.size"] == 15
    assert plt.rcParams["font.size"] == before["font.size"]
