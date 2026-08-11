# Meta sprint demo — mne-denoise

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snesmaeili/mne-denoise-meta-demo/blob/main/meta_mne_denoise_demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/snesmaeili/mne-denoise-meta-demo/main?labpath=meta_mne_denoise_demo.ipynb)
[![License: BSD-3](https://img.shields.io/badge/code-BSD--3--Clause-blue.svg)](LICENSE)
[![Data: CC BY-SA 4.0](https://img.shields.io/badge/data-CC%20BY--SA%204.0-lightgrey.svg)](NOTICE.md)

**There is no universally correct M/EEG denoiser.** Line noise, transient
movement, reference-correlated artifacts and "enhance the response I care
about" are different problems that need different information.
[`mne-denoise`](https://github.com/mne-tools/mne-denoise) implements each as an
MNE-native estimator — and leaves a fitted state you can interrogate to check
what it actually did.

This repository is that argument, in five minutes and four measurements.

## Run it yourself

Click **Open in Colab** above. The first cell pins `mne-denoise` to the exact
commit the numbers were measured against and downloads a 20 MB bundle of
prepared assets — no OpenNeuro download, no multi-gigabyte wait. The whole
notebook then executes in about ten seconds.

Everything you see was computed by the software. Nothing is a screenshot, and
nothing is transcribed from a manuscript.

Prefer to read rather than run? **[View the executed notebook](executed_demo.html)**.

## Going deeper

Five focused notebooks sit behind the talk. Each runs standalone in Colab, in
under fifteen seconds, mostly on synthetic data so nothing has to be downloaded.

| | Notebook | What it adds |
|---|---|---|
| 01 | [ZapLine](deep_dives/01_zapline.ipynb) | notch vs fixed vs adaptive on a line that *moves*; what `adaptive_results_` records |
| 02 | [ASR](deep_dives/02_asr.ipynb) | all four variants against known ground truth; where each looks for clean calibration data |
| 03 | [iCanClean](deep_dives/03_icanclean.ipynb) | reference-based CCA, the time-shifted control, and the ds004505 recording where the control comes back negative — and why |
| 04 | [DSS](deep_dives/04_dss.ipynb) | two biases on identical data; why the stronger signal is not the one you get |
| 05 | [Spectrum interpolation](deep_dives/05_spectrum_interpolation.ipynb) | phase-preserving line removal, and the rank it does not cost you |

---

A five-minute live demo for the MNE-Python maintainers sprint.
**Different noise. Different assumptions. One MNE-native contract.**

Four regimes on stage — periodic, transient, a target you declare, and
reference-correlated. Everything else, including all four ASR variants and the
DSS-versus-Xdawn comparison, is in the deep dives so the talk stays inside five
minutes.

Built against mne-denoise **`main` @ `f5b821c`** with mne 1.12.1, numpy 2.4.6,
scipy 1.17.1, Python 3.14.

This repository is deliberately separate from
[mne-tools/mne-denoise](https://github.com/mne-tools/mne-denoise): it is a talk,
not a package feature. Fixes discovered while building it go upstream as pull
requests; nothing here is imported by the library.

## Prerequisites

`mne-denoise` must be importable. **Install it from a checkout, not a wheel** —
one act reads the rASR reference recording that is vendored in the source tree,
and the manifests record the exact package commit under test:

```bash
git clone https://github.com/mne-tools/mne-denoise.git
pip install -e ./mne-denoise
```

The demo finds that checkout automatically from `mne_denoise.__file__`. Override
with `META_DEMO_MNE_DENOISE_ROOT` if you have several, or point
`META_DEMO_SME` straight at `sme_1_1.xdf_filt.set` to skip the search.

---

## A. One-time preparation

```bash
export META_DEMO_DS003620=/path/to/ds003620      # Runabout mobile EEG
export META_DEMO_N170=/path/to/erpcore/N170      # ERP CORE N170
export META_DEMO_DS004505=/path/to/raw_bids      # Table Tennis

python prepare_meta_demo.py --all
```

Each stage can be run alone: `--zapline`, `--asr`, `--asr-variants`, `--dss`,
`--movement`.
If a dataset cannot be found the script fails immediately with the exact
environment variable to set and the directory layout it expected. It never
downloads anything.

Approximate wall time on a laptop, cold: **~15 min total** — line noise ~3 min,
ASR ~15 s, DSS ~5 min (first run also builds a 40-subject epoch cache; later
runs ~2.5 min), movement ~5 min. Peak memory ~6 GB during the movement stage.

## B. Preflight

```bash
python prepare_meta_demo.py --check
```

Prints a compact readiness block and exits non-zero if anything is missing. The
notebook's first cell calls `du.assert_presenter_ready()`, which does the same
check and raises before any act runs.

## C. Launching the notebook

```bash
jupyter lab meta_mne_denoise_demo.ipynb
```

Advance cell by cell with <kbd>Shift</kbd>+<kbd>Enter</kbd>. Ten code cells, five
figures. Nothing requires the mouse.

## D. Running entirely offline

The notebook reads only from `~/.cache/mne-denoise/meta-demo`. It performs no
network access and touches no dataset path. Verified by executing the notebook
end to end with no network and a clean environment.

Override the cache location with `MNE_DENOISE_META_DEMO_CACHE` if you want to
carry it on a USB stick.

**Presenter-mode runtime: 7.8 s warm**, slowest cell 2.3 s (imports + preflight).
No cell exceeds 3 s.

> **Warm the kernel before you present.** From a cold filesystem cache the first
> cell takes ~35 s (52.9 s total) purely importing `mne` and `matplotlib`. Run
> the notebook once, then restart the kernel and run it again — the second run is
> the 7.8 s one. This is the single most likely way for the demo to look slow.

## E. What each act proves

| Act | Question | Result |
|---|---|---|
| 1 — line noise | Can two methods remove the same 50 Hz peak at different cost? | R(50 Hz) 1.40 → notch **0.17** (a fifth of the surrounding floor) → ZapLine+ **1.00** (at the floor). ZapLine+ pays 0.087 dB of broadband distortion; the notch pays 0.0005 dB. |
| 2 — ASR | What did ASR detect, and what did it cost? | Artifact-interval RRMSE 0.492 → **0.160**. Artifact-free RRMSE 0.003 → **0.165** — a real cost. `calibration_info_` explains it: at 20 s the calibration has 27 samples per channel dimension and *both* endpoints get worse. |
| 3 — DSS | Why do I need this, when MNE already ships Xdawn and SSD? | Because the criterion is an argument. On one fixture with two planted sources, component 1 follows whatever is declared: PCA → the rhythm (\|cos\| **1.00**, it has more variance), `AverageBias` → the evoked source (**0.995**), `BandpassBias` → the rhythm (**0.999**). *On stage this is one panel and one sentence.* |
| 4 — blink removal | Does attenuation mean the method worked? | **No — the ordering is nearly reversed.** Blink removed / N170 effect (uncorrected −0.74 µV): iCanClean **83.1% / −1.43**, `EOGRegression` 73.0% / −1.15, pseudo-reference notch **92.2% / +0.59** (sign inverted). The arm that removes the most artifact destroys the effect. |

### Measured, but held back for questions

| Where | Result |
|---|---|
| deep dive 02 — ASR variants | Robustness to a contaminated calibration comes from `cov_estimator`, not `method=` — the threshold inflates **+20%** with `mean` and **+9%** with the default `geometric_median`, while `standard` and `riemannian_windowed` are numerically identical. On real mobile EEG the window selector is **not** starving (74%), so Juggler is not indicated. |
| deep dive 04 — DSS head-to-head | DSS beats matched-rank PCA in 32/40 but `XdawnTransformer` in only **15/40**; plain PCA beats raw sensors in **40/40**. Reproducibility improves in 37/40, condition AUC in only 25/40. Xdawn is *not* a special case of DSS — mean principal-angle overlap 0.46. |
| deep dive 03 — DSS on blinks | Same blinks, blink times only: DSS linear **95.9% / −0.56**, DSS non-linear 76.2% / −0.37. The non-linear arm spans −1.05 to −0.02 µV across six seeds; the linear one is a closed form and carries no spread. |

## F. Datasets

| Act | Dataset | Scope |
|---|---|---|
| 1 | OpenNeuro **ds003620**, Runabout mobile EEG (`doi:10.18112/openneuro.ds003620.v1.1.1`) | 32 ch, 500 Hz, `PowerLineFrequency` 50; whole-recording analysis of one participant |
| 2 | Synthetic fixture, `_asr_fixture.py` | 32 ch, 250 Hz, known clean ground truth |
| 2b | Same fixture + **SME `sme_1_1`**, the sample recording shipped with the rASR MATLAB reference | 24-ch Smarting mobile EEG, 250 Hz, first 120 s. In-repo at `refs/asr/repos/rASRMatlab/sampleData/filtered/` |
| 3 | **ERP CORE N170** (`doi:10.18115/D5JW4R`) | all 40 participants, faces vs cars |
| 4 | ERP CORE **N170** (`doi:10.18115/D5JW4R`) | 30 EEG + 3 EOG (HEOG left/right, VEOG lower), **sub-005 only** — the Act 3 representative |
| deep dive 03 | OpenNeuro **ds004505**, Table Tennis (`doi:10.18112/openneuro.ds004505.v1.0.2`) | 120 EEG + 120 dual-layer reference electrodes + 8 neck EMG, **sub-01 only** |

`ds000117` — the MEG M170 arm in the working draft — is not available offline
here. ERP CORE N170 is its EEG homologue: the same face-perception construct,
with 40 participants instead of 16, and it needs no download.

## G. How the representative recordings were chosen

Every selection uses a **baseline property of the uncorrected data**. No
cleaning result, and no endpoint, enters any selection.

- **Act 1** — the recording whose whole-recording median R(50 Hz) is closest to
  the median of that statistic across all available recordings. Selected
  **sub-08** (R = 1.400; cohort median 1.400 across 9 recordings).
- **Act 3** — the participant whose baseline evoked SNR — mean GFP over
  110–150 ms divided by mean GFP over the −200–0 ms baseline, on face trials —
  is closest to the cohort median. Selected **sub-005** (SNR 4.537; cohort
  median 4.532 across 40). The best N170-window sensor resolves to **PO7**, the
  canonical N170 electrode, which is a useful sanity check on the rule.
- **Act 4** — only sub-01 is on disk. The analysed segment is the first 575 s of
  the longest recording-boundary-free run, which is the *only* run long enough
  to admit it, so the choice has no free parameter.
- **Act 2** — synthetic; no selection.

Ties break on sorted identifier, so every rule is deterministic. All of this is
recorded in the manifests.

## H. Where the cache lives

`~/.cache/mne-denoise/meta-demo` (override: `MNE_DENOISE_META_DEMO_CACHE`).

```
zapline_spectra.npz  zapline_adaptive.npz  zapline_demo_raw.fif  zapline_metrics.json
asr_fixture.npz      asr_metrics.json
dss_demo-epo.fif     dss_sources.npz       dss_metrics.json      dss_group.json
movement_traces.npz  movement_metrics.json
n170_epochs/         <- 40 cached epoch files, so re-runs are fast
*_manifest.json      <- one per act
```

Only stable formats: FIF, NPZ, JSON. No pickled estimators. Every act writes a
manifest recording the repository commit, package versions, dataset identifier
and DOI, subject, preprocessing, estimator parameters, the selection rule, the
random seed, and a UTC timestamp.

Total cache size ≈ 250 MB, dominated by the N170 epoch cache. Nothing here
redistributes raw OpenNeuro data.

## I. Regenerating

Delete the cache and re-run preparation:

```bash
rm -rf ~/.cache/mne-denoise/meta-demo
python prepare_meta_demo.py --all
```

Every stage is seeded (`RANDOM_STATE = 97`; DSS resampling uses `20260101`) and
reproduces bit-for-bit on the same machine and package versions.

To rebuild the notebook figures without re-preparing, just re-run the notebook.

---

## The ASR family — where each variant comes from

| Variant | Source | Regime it was built for |
|---|---|---|
| `ASR(method="standard")` | Kothe & Jung 2016; Chang et al. 2020, *IEEE TBME* 67(4) | transient bursts on ordinary EEG |
| `ASR(method="riemannian_windowed")` | Blum et al. 2019, *Front. Hum. Neurosci.* 13:141 | calibration windows themselves contaminated |
| `AdaptiveASR(variant="psp"/"psw"/"mw")` | Tsai et al. (Hebbian / anti-Hebbian) | non-stationary recordings, streaming BCI |
| `JugglerASR(strategy="gev"/"dbscan")` | Kim et al. 2025, *J. Neurosci. Methods* 420:110465 | extreme MoBI — 205-channel juggling EEG |

**rASR is not a sleep method.** Blum et al. developed it on 24-channel *mobile*
EEG recorded with gyroscope, accelerometer and GPS streams, indoors and
outdoors — the recording vendored here as `sme_1_1`. The sleep ASR toolbox is
`dusk2dawn` (Somervail et al. 2023, *Sleep*), a separate lineage that happens to
embed Blum's Riemannian option as an off-by-default flag. That is the likely
source of the confusion.

### Two results from Act 2b worth stating carefully

1. **`method="riemannian_windowed"` changes nothing at package defaults.** Both
   backends already use `cov_estimator="geometric_median"`; the only structural
   difference is partial-block handling, which vanishes when the calibration
   length is a multiple of `blocksize`. Blum's contribution is real and is
   measured here (threshold inflation +9% vs +20% for `mean` when the
   calibration segment is dirty) — but it lives in `cov_estimator`, and it is
   already on by default for every variant.
2. **Kim et al. report reference fractions of 9% (standard), 24% (GEV) and 42%
   (DBSCAN).** We measure 74% / 24% / 66% on `sme_1_1`. GEV matches almost
   exactly; the others do not, because a 24-channel walking recording is far
   less contaminated than 205-channel juggling. The repository had already
   documented that this ordering does not reproduce on synthetic bursts
   (`scripts/run_juggler_parameter_ablation.py`: *"No burst count in the swept
   range produced GEV > standard"*). The paper values are drawn on the figure as
   reference marks and are **not** presented as our measurements.

## Known issues found while building this

These are real, reproduced, and **not worked around** in the demo.

1. **`DSS` silently accepts the wrong epoched-array axis order.** The documented
   convention is `(n_channels, n_times, n_epochs)`. Passing
   `(n_epochs, n_channels, n_times)` — the MNE `get_data()` order — raises no
   error and produces an `n_epochs × n_epochs` "spatial" filter.
   ```python
   DSS(bias=AverageBias(axis="epochs")).fit(np.zeros((40, 12, 120))).filters_.shape
   # (40, 40)   <- silently filtering over epochs
   ```
   The demo transposes explicitly. Worth a shape check in `DSS.fit`.

2. **`DSS` loses `tmin`/`baseline` on its internal biased Epochs.**
   `mne_denoise/dss/linear.py:769` builds `mne.EpochsArray(biased_data, inst.info)`
   without carrying `tmin` or `baseline`, so MNE reports the *internal* object as
   un-baselined and warns — even when the epochs you passed in are correctly
   baselined. The notebook filters that one message with an explicit comment.
   The covariance arithmetic is unaffected; only metadata and the warning.

3. **`TrialAverageBias` does not exist** but is referenced in
   `mne_denoise/dss/linear.py:266` and in the `DSS` docstring example at
   `linear.py:391`. The class is `AverageBias`.

4. **`n_remove=` is dead** but survives in `docs/auto_examples/zapline/*` and
   `examples/tutorials/viz_showcase.ipynb`. The current argument is `n_select`.
   The hand-written `examples/zapline/*.py` are correct; the generated
   `docs/auto_examples` copies are stale.

5. **Adaptive ZapLine+ can remove components in chunks where its own detector
   says no line artifact is present.** On the Act 1 recording, 9 of 92 chunks
   (331 s, 8.7% of the recording) had `artifact_present=False` yet removed 6
   components — exactly `int(32 × max_prop_remove)`. The cause is
   `n_select='auto'` on a flat eigenvalue spectrum: `auto_select` flags nearly
   every component, and only the `max_prop_remove` cap bounds it.
   `artifact_present` currently only lowers the *minimum* removal count
   (`core.py:905`); it does not gate removal. Measured broadband distortion
   stayed small (0.087 dB), so this is a robustness issue rather than a
   correctness failure — but it is worth a decision, and it would change
   published numbers, so the demo documents it rather than changing behaviour.

6. **`ASR(method='riemannian_windowed')` can be numerically identical to
   `'standard'`.** Both use `cov_estimator='geometric_median'` by default; the
   only structural difference is `covariance_kind` (`"padded"` vs `"standard"`),
   which matters only when the calibration length is not a multiple of
   `blocksize`. In Act 4 the two agree to three decimals on every endpoint.

7. **`ASR.picks_` is `None` for ndarray input** although the class docstring
   documents it unconditionally as an ndarray of channel indices.

8. **`ASR(picks=...)` is assigned and never read.** Channel selection is done
   entirely by `_get_homogeneous_picks` (`utils.py:37`), which takes the first
   present type in the order **mag → grad → eeg**. On a mixed EEG+MEG object
   `picks="eeg"` still selects the MEG channels. Always `raw.pick("eeg")` before
   fitting. Act 4 asserts the resolved channel count for this reason.

9. **`ASR(random_state=...)` is inert** — there is no randomness anywhere in the
   ASR pipeline; output is bitwise identical across seeds. `copy` and `n_jobs`
   are inert too.

10. **`adaptive.py:762` contains a stray `print("CAUGHT:", repr(exc))`** inside
    the moving-window exception handler, which will dump a traceback to stdout
    if an MW window fails to calibrate. The demo avoids `variant="mw"`.

11. **Several documented defaults are wrong** — `ref_max_bad_channels` is 0.075
    not 0.2, `ref_tolerances` is `(-inf, 5.5)` not `(-3.5, 5.0)`,
    `calibration_window_length` is 1.0 not 0.5, `max_mem_mb` is 512 not 200,
    and `AdaptiveASR`'s `mw_mode` options are `final_state`/`sliding`, not
    `cumulative`. Read the signatures, not the docstrings.

---

## Night-before-the-talk checklist

- [ ] `git rev-parse HEAD` → matches the commit in the manifests
      (`python -c "import json,pathlib; print(json.loads((pathlib.Path.home()/'.cache/mne-denoise/meta-demo/zapline_manifest.json').read_text())['environment']['repo_commit'])"`)
- [ ] `python prepare_meta_demo.py --check` → `META DEMO READY`
- [ ] Open the notebook and **Run All once to warm the import cache** (a cold
      first run takes ~50 s, almost all of it in cell 1)
- [ ] **Restart Kernel and Run All** again — this run should finish in ~8 s with
      five figures and no traceback
- [ ] Turn **off** wifi and run it again — it must behave identically
- [ ] Confirm `PRESENTER_MODE = True`, `RECOMPUTE = False`, `SHOW_DIAGNOSTICS = False`
- [ ] Set the browser/JupyterLab zoom so the smallest axis label is readable
      from the back of the room
- [ ] Clear all outputs before presenting, so the figures appear as you run
- [ ] Have `SHOW_DIAGNOSTICS = True` ready for the adaptive-chunk panel in case
      someone asks how ZapLine+ tracks contamination over the recording

### Numbers worth knowing cold

- Act 1 — R: 1.40 → 0.17 (notch) → 1.00 (ZapLine+)
- Act 2 — artifact RRMSE 0.49 → 0.16; artifact-free 0.00 → 0.17
- Act 2b — threshold inflation +20% (`mean`) vs +9% (`geometric_median`);
  calibration supply on mobile EEG 74% / 74% / 24% / 66%
- Act 3 — bias swap: PCA → alpha 1.00, `AverageBias` → evoked 0.995,
  `BandpassBias` → alpha 0.999. Median held-out reproducibility: sensors 0.9675,
  PCA 0.9712, DSS 0.9747, **Xdawn 0.9787**. DSS beats PCA 32/40, Xdawn 15/40;
  PCA beats sensors 40/40. Reproducibility up 37/40; discriminability up 25/40.
- Act 4 — N170 effect uncorrected **−0.74 µV**. Blink removed / N170 effect:
  iCanClean (EOG electrodes) 83.1% / **−1.43**, EOG regression 73.0% / −1.15,
  DSS linear **95.9% / −0.56**, DSS non-linear 76.2% / −0.37,
  pseudo-reference notch **92.2% / +0.59** (sign inverted), pseudo-reference
  lowpass 89.8% / +0.36. The non-linear DSS arm spans −1.05 to −0.02 µV across
  six seeds — it is a fixed-point iteration, not a closed form.
- deep dive 03 — ds004505 real reference 11.5%, scrambled 10.5%, at **2.08**
  samples per dimension; widen to a 30 s stats window (31.2 samples/dim) and
  mean R² falls 0.296 → 0.026 with **nothing** removed

### Likely questions

- *"Why DSS when MNE has Xdawn and SSD?"* — Not because it scores better; on the
  N170 arm Xdawn wins on the median (15/40 for DSS). Because
  `max_w (wᵀR_biased w)/(wᵀR_baseline w)` leaves `R_biased` as an argument, and
  Xdawn/SSD/CSP each freeze it. Swap the argument and the same estimator answers
  a different question — that is the left panel of the Act 3 figure.
- *"Isn't DSS just PCA/ICA?"* — Identity bias gives eigenvalues of exactly 1.0
  (deep dive 04: max deviation 3e-15), so the bias is the entire content. With a
  non-linear contrast, `IterativeDSS` + `TanhMaskDenoiser` lands on FastICA's
  components at \|r\| = 1.000 / 0.999 / 0.895 / 0.894 — close, not identical, and
  FastICA recovered all four sources where DSS got three.
- *"Why is the non-linear DSS arm an error bar?"* — Linear DSS is a closed-form
  generalised eigendecomposition, so it returns the same answer every run.
  `IterativeDSS` is a fixed-point iteration from a random init: over six seeds
  its N170 lands between −1.05 and −0.02 µV while attenuation stays at
  77.4% ± 2.8. Reporting one seed would be reporting a choice. Note also that
  `beta=beta_tanh` must **not** be used when reconstructing — it speeds
  convergence but leaves `filters_` non-orthogonal, so `patterns_` stops being a
  valid inverse and `inverse_transform` silently returns garbage (round-trip
  relative error 1.1 with it, 7e-15 without).
- *"Where did `pseudo_ref=True` go?"* — It was added to `ICanClean` in
  mne-denoise `80b02e0` (with `filter_ref`, matching MATLAB
  `filtYtype='Notch'`) and removed in the PR-26 refactor with no changelog
  entry; `tests/test_icanclean.py:605` now asserts it raises `TypeError`. Act 4
  therefore builds the pseudo-reference explicitly from the shipped API. On
  `main` there is currently **no** reference-free CCA path at all — `LaggedCCA`
  (PR #49) would provide one but is an unmerged draft.
- *"Why doesn't iCanClean work on ds004505?"* — It is not the method, it is the
  operating point. 120 primary + 120 reference channels is 240 dimensions, and
  the 2 s default window supplies 500 samples — **2.08 samples per dimension**.
  A CCA that under-determined finds a "shared" subspace in noise; the scrambled
  reference even scores a marginally *higher* mean R² (0.2966 vs 0.2959). Widen
  the stats window to 30 s and the subspace evaporates (R² 0.026, nothing
  removed). The real fix is to reduce reference dimensionality first. Deep dive
  03 has the numbers.
- *"Isn't rASR for sleep?"* — No; that's `dusk2dawn`. See the ASR family table.
- *"Why doesn't rASR beat standard?"* — Because its robustness is already the
  default here. Show the `cov_estimator` panel.
- *"Did you reproduce Kim et al.?"* — GEV yes (24% vs 24%); the ordering no, and
  the repo documented that before this demo existed.

### If something goes wrong

Every act's cell is independent: they all read the cache and share no state
except the imports in cell 1. If one act fails, skip its cell and keep going —
nothing downstream depends on it.
