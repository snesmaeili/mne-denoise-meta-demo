# mne-denoise — Meta sprint demo, executed run record

This is a **verbatim record of one full run** of `meta_mne_denoise_demo.ipynb`,
exported for review. Every number below was produced by the code shown directly
above it; nothing is transcribed by hand.

**Read this as the script for a 5-minute live talk**, not as a paper. Specifics
that are deliberate rather than oversights:

- Acts 3 and 4 are **n = 1 participant**, stated on the slides. They demonstrate
  a measurement principle, not a group result. Group-level counts across 40
  participants appear in Act 3.
- The demo **reports negative and null results on purpose** — a method losing to
  a baseline, a control failing, an endpoint that cannot see damage. Those are
  the argument, not defects.
- Figures are omitted here (placeholders mark where they appear). The rendered
  version with figures is `executed_demo.html`.
- Cells marked *(cached)* replay a result prepared offline by
  `prepare_meta_demo.py`; the raw datasets are too large to ship. The
  preparation code is in the repository.

---


# Real M/EEG is messy — and the noise is never the same shape

### There is no universally correct denoiser. `mne-denoise` matches the method to the structure of the contamination, and lets you audit what it did.

```
        PERIODIC              TRANSIENT           REFERENCE-CORRELATED        A TARGET YOU DECLARE
     power-line noise      movement bursts        recorded noise channels      reproducible response
            │                     │                        │                            │
            ▼                     ▼                        ▼                            ▼
   ZapLine / SpectrumInterp      ASR                   iCanClean                       DSS
```

Every act below asks one question, runs the real estimator, and reports **two**
numbers: did the artifact go down, and did the neural signal survive.

MNE-Python maintainers sprint &nbsp;·&nbsp; Meta Paris

## 0 — Setup

On Colab this installs a pinned `mne-denoise` and fetches a 20 MB data bundle.
Locally it is a no-op — everything is already cached.

```python
# Colab / fresh-environment bootstrap. Does nothing on a prepared machine.
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
DEMO_REPO = "https://github.com/snesmaeili/mne-denoise-meta-demo.git"
MNE_DENOISE_PIN = "f5b821cc2a535e84ed46085d45ea5a356dd8d548"
MNE_DENOISE_SPEC = (
    f"mne-denoise @ git+https://github.com/mne-tools/mne-denoise.git@{MNE_DENOISE_PIN}"
)

def _pip(spec):
    # subprocess, not %pip: line magics do not interpolate {braces}, so an
    # f-string here is the only way the pin actually reaches pip.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], check=True)

if IN_COLAB and not Path("demo_utils.py").exists():
    done = subprocess.run(["git", "clone", "-q", "--depth", "1", DEMO_REPO],
                          capture_output=True, text=True)
    if done.returncode != 0:
        print("Could not clone the demo repository.")
        print("git said:", (done.stderr or "").strip() or done.returncode)
        raise SystemExit(
            "If the repository is still private, the Colab VM has no credentials "
            "for it -- authorising Colab lets it OPEN a notebook, not clone the "
            "repo. Make the repository public, or run this notebook locally."
        )
    os.chdir("mne-denoise-meta-demo")
    sys.path.insert(0, os.getcwd())

try:
    import mne_denoise  # noqa: F401
except ImportError:
    _pip(MNE_DENOISE_SPEC)

# Pull the prepared assets only if the cache is incomplete. The existence check
# is deliberately cheap so a prepared machine pays nothing here.
import os

_cache = Path(os.environ.get("MNE_DENOISE_META_DEMO_CACHE",
                             Path.home() / ".cache" / "mne-denoise" / "meta-demo"))
if not (_cache / "zapline_metrics.json").exists():
    subprocess.run([sys.executable, "fetch_demo_data.py"], check=False)
else:
    print(f"demo assets already present in {_cache}")
```

```
demo assets already present in C:\Users\s\.cache\mne-denoise\meta-demo
```

```python
PRESENTER_MODE = True      # stage settings: quiet, fast, no network
LIVE = True                # False -> load the cached result instead of computing it
RECOMPUTE = False          # True re-runs the slow paths instead of loading the cache
SHOW_DIAGNOSTICS = False   # extra panels, only if someone asks
RANDOM_STATE = 97

import sys, warnings, logging
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt
import mne

import demo_utils as du

mne.set_log_level("ERROR")
logging.getLogger("mne_denoise").setLevel(logging.ERROR)
if PRESENTER_MODE:
    # Two narrowly-scoped filters. Warnings about rank, calibration, or any
    # algorithmic failure stay switched on deliberately.
    warnings.filterwarnings("ignore", message=".*figure layout has changed.*")
    # DSS builds its biased Epochs with mne.EpochsArray(data, info) and does not
    # carry tmin/baseline across, so MNE reports the *internal* object as
    # un-baselined. The epochs we pass in are baseline corrected -- verified in
    # README.md ("Known issues"). Filtered so a library metadata bug does not
    # look like a data problem on stage.
    warnings.filterwarnings(
        "ignore", message=".*Epochs are not baseline corrected.*")
%matplotlib inline

du.assert_presenter_ready()
```

```
META DEMO READY
  cache        C:\Users\s\.cache\mne-denoise\meta-demo
  mne-denoise  f5b821cc2a53
  demo repo    70a764c46de2
  [x] package imports
  [x] ZapLine+ / line noise
  [x] ASR transient reconstruction
  [x] ASR variant regimes
  [x] DSS target enhancement
  [x] iCanClean vs EOG references
  [x] Movement attenuation/preservation
  [x] offline (no network required)
```

---
## 1 — Can two methods remove the same 50 Hz peak at very different cost?

*Mobile EEG recorded while walking a university campus — OpenNeuro `ds003620`, 32 channels, 50 Hz mains.*

```python
# The real estimator call, run live on a 60 s slice of the recording.
from mne_denoise.zapline import ZapLine
from mne_denoise.qa import noise_surround_ratio

m = du.load_json(du.cache_path("zapline_metrics.json"))
raw = mne.io.read_raw_fif(du.cache_path("zapline_demo_raw.fif"), preload=True, verbose="ERROR")

if LIVE:
    zap = ZapLine(sfreq=raw.info["sfreq"], line_freq=50.0, adaptive=True, n_select="auto")
    clean = zap.fit_transform(raw.copy())

    chunks = zap.adaptive_results_["chunk_info"]
    print(f"detected {zap.adaptive_results_['line_freq']:.0f} Hz, removed "
          f"{zap.n_removed_} component(s) over {len(chunks)} adaptive chunk(s)")

    psd_kw = dict(method="welch", fmin=1.0, fmax=125.0, n_fft=8192, verbose="ERROR")
    for name, obj in (("before", raw), ("after ", clean)):
        p = obj.compute_psd(**psd_kw)
        r = np.median(noise_surround_ratio(p.freqs, p.get_data(), 50.0, peak_bw=0.5))
        print(f"   R(50 Hz) {name} = {r:.2f}")
else:
    lc = m["live_crop"]
    print(f"(cached) 60 s window at {lc['start_s']:.0f} s, "
          f"uncorrected R(50 Hz) = {lc['window_ratio']:.2f}")
```

```
detected 50 Hz, removed 1 component(s) over 1 adaptive chunk(s)
   R(50 Hz) before = 1.20
   R(50 Hz) after  = 1.10
```

```python
# Full-recording result, prepared offline (see prepare_meta_demo.py --zapline).
m = du.load_json(du.cache_path("zapline_metrics.json"))
S = du.load_npz(du.cache_path("zapline_spectra.npz"))

with du.presentation_theme():
    fig = du.plot_line_noise_triptych(
        S["freqs"],
        {"original": S["psd_original"], "notch": S["psd_notch"], "zapline+": S["psd_zapline"]},
        line_freq=m["line_freq"], ratios=m["ratio"], fmin=40.0, fmax=65.0,
    )
plt.show()

print(f"R = residual peak power / local spectral floor      (R = 1 means 'at the floor')")
for k in ("original", "notch", "zapline+"):
    print(f"   {k:<10s} R = {m['ratio'][k]:.2f}")
```

*[1 figure(s) rendered here — omitted from this text export]*

```
R = residual peak power / local spectral floor      (R = 1 means 'at the floor')
   original   R = 1.40
   notch      R = 0.17
   zapline+   R = 1.00
```

> **Say:** both remove the peak. The difference is what else they touch. The notch suppresses
> the whole band well below its surrounding spectral floor; ZapLine+ brings the peak back
> *to* the floor by modelling the **spatial** line subspace instead of imposing a fixed
> spectral depth.

```python
if SHOW_DIAGNOSTICS:
    A = du.load_npz(du.cache_path("zapline_adaptive.npz"))
    with du.presentation_theme():
        fig = du.plot_adaptive_component_timeline(
            A["chunk_start"], A["chunk_n_removed"],
            contamination=(A["contamination_times"], A["contamination_ratio"]),
        )
    plt.show()
    print(f"{m['n_chunks']} chunks, {m['chunks_with_zero_removed']} of them removed nothing.")
    print("Contamination over the recording spans "
          f"R = {m['contamination_range'][0]:.2f} to {m['contamination_range'][1]:.2f}.")
```

---
## 2 — What did ASR actually detect, and what did it cost?

*A synthetic recording where the clean signal is known exactly, plus blinks, muscle bursts,
electrode pops and a covariance shift.*

```python
F = du.load_npz(du.cache_path("asr_fixture.npz"))
a = du.load_json(du.cache_path("asr_metrics.json"))
head = a[f"{a['headline_duration_s']:.0f}s"]["standard"]

with du.presentation_theme():
    fig = du.plot_asr_reconstruction_panel(
        F["times"], F["contaminated"], F["cleaned"], F["clean"],
        repair=(F["window_times"], F["n_components_reconstructed"]),
        channels=(0, 14, 22), ch_names=list(a["headline_ch_names"]),
        tlim=(1.0, 19.0),   # display window only; metrics use the whole 60 s
    )
plt.show()

print(f"artifact intervals   RRMSE {head['artifact_rrmse_before']:.3f} "
      f"-> {head['artifact_rrmse_after']:.3f}")
print(f"artifact-free data   RRMSE {head['clean_rrmse_before']:.3f} "
      f"-> {head['clean_rrmse_after']:.3f}   <- the cost")
```

*[1 figure(s) rendered here — omitted from this text export]*

```
artifact intervals   RRMSE 0.492 -> 0.160
artifact-free data   RRMSE 0.003 -> 0.165   <- the cost
```

```python
# The fitted object says where it acted, and how much reference data it had.
print(f"windows modified            {head['fraction_windows_modified']:6.1%}")
print(f"samples repaired            {head['fraction_samples_repaired']:6.1%}")
print(f"recall on true artifacts    {head['sample_mask_recall_on_artifact']:6.1%}")
print(f"calibration                 {head['calibration_samples_per_dim']:.0f} samples "
      f"per channel dimension")

short = a["20s"]["standard"]
print(f"\nSame estimator, same defaults, a 20 s recording instead of 60 s:")
print(f"   calibration              {short['calibration_samples_per_dim']:.0f} samples/dim")
print(f"   artifact RRMSE           {short['artifact_rrmse_before']:.3f} "
      f"-> {short['artifact_rrmse_after']:.3f}")
print(f"   artifact-free RRMSE      {short['clean_rrmse_before']:.3f} "
      f"-> {short['clean_rrmse_after']:.3f}")
```

```
windows modified             20.2%
samples repaired             26.4%
recall on true artifacts     98.2%
calibration                 253 samples per channel dimension

Same estimator, same defaults, a 20 s recording instead of 60 s:
   calibration              27 samples/dim
   artifact RRMSE           0.471 -> 0.341
   artifact-free RRMSE      0.003 -> 0.318
```

> **Say:** here we know the clean target, so we can price the cleaning. ASR identified about
> 98% of the contaminated samples. It also touched data that needed nothing — and
> `calibration_info_` tells us why: starve the calibration and both endpoints get worse
> together.
>
> The package ships four ASR variants for four calibration regimes. Choosing between them is
> a whole conversation on its own — it is deep dive 02, and I am happy to open it in
> questions.

---
## 3 — Sometimes the goal is not to remove noise, but to keep a structure you can name

DSS maximises a ratio you *declare*: $\max_w \; w^{\top} R_{\text{biased}} w \,/\, w^{\top} R_{\text{baseline}} w$. PCA, Xdawn, SSD and CSP are all this problem with $R_{\text{biased}}$ **frozen**. DSS leaves it as an argument.

```python
# One fixture, two planted sources, three criteria. DSS fits in a fraction of a
# second, so the estimator call runs live.
from mne_denoise.dss import DSS, AverageBias

d = du.load_json(du.cache_path("dss_metrics.json"))
epochs = mne.read_epochs(du.cache_path("dss_demo-epo.fif"), preload=True, verbose="ERROR")

if LIVE:
    dss = DSS(bias=AverageBias(axis="epochs"), n_select="auto")
    dss.fit(epochs)
    print(f"DSS on {len(epochs)} real trials: kept {dss.n_selected_} components, "
          f"leading bias scores {np.round(dss.eigenvalues_[:3], 3)}")

b = d["bias_swap"]
with du.presentation_theme():
    fig = du.plot_dss_framework_panel(b)
plt.show()

amp = b["_amplitudes"]
print(f"planted: evoked at {amp['evoked']}, alpha at {amp['alpha']} — "
      f"the distractor is the stronger source")
for name in ("PCA", "AverageBias", "BandpassBias"):
    print(f"   {name:<13s} evoked {b[name]['evoked']:.3f}    alpha {b[name]['alpha']:.3f}")
```

*[1 figure(s) rendered here — omitted from this text export]*

```
DSS on 160 real trials: kept 8 components, leading bias scores [0.577 0.422 0.14 ]
planted: evoked at 1.1, alpha at 1.4 — the distractor is the stronger source
   PCA           evoked 0.069    alpha 1.000
   AverageBias   evoked 0.995    alpha 0.025
   BandpassBias  evoked 0.060    alpha 0.999
```

> **Say:** same data, same estimator, one argument changed — and the answer moves from the
> evoked source to the rhythm. PCA returns the rhythm too, because the rhythm has more
> variance; it has no way to be *asked* for anything else. Xdawn cannot become SSD.
>
> On real N170 data DSS does not beat Xdawn at this — that comparison, and the 40-participant
> result, are in deep dive 04.

```python
if SHOW_DIAGNOSTICS:
    # The head-to-head that is cut from the live path: DSS against the tools
    # that already exist, on real N170 data, plus the 40-participant counts.
    G = du.load_json(du.cache_path("dss_group.json"))
    g, per = G["summary"], G["per_subject"]
    r = d["reproducibility"]

    with du.presentation_theme():
        fig = du.plot_dss_framework_panel(b, r, group=g)
    plt.show()

    print("median split-half reproducibility, held out:")
    for key, label in (("sensor", "raw sensors"), ("pca", "PCA, matched rank"),
                       ("dss", "DSS AverageBias"), ("xdawn", "Xdawn (already in MNE)")):
        print(f"   {label:<24s} {r[f'{key}_median']:.4f}")
    print(f"\nAcross all {g['n_subjects']} participants:")
    print(f"   reproducibility improved in   {g['reproducibility_gain_positive']}/{g['n_subjects']}")
    print(f"   discriminability improved in  {g['auc_change_positive']}/{g['n_subjects']}")
    print(f"   DSS beat Xdawn in             {g['dss_over_xdawn_positive']}/{g['n_subjects']}")
    print(f"   plain PCA beat raw sensors in {g['pca_over_sensor_positive']}/{g['n_subjects']}")

    ov = d["subspace_overlap_dss_xdawn"]
    print(f"\nDSS vs Xdawn principal-angle cosines: {np.round(ov, 3).tolist()}")
    print(f"   mean {np.mean(ov):.3f} — the leading direction agrees, the rest does not,")
    print("   so Xdawn is NOT a special case of DSS as implemented.")
```

---
## 4 — Does artifact attenuation mean the method worked?

*Blinks on ERP CORE N170, `sub-005` — the same participant as Act 3.*

Six methods, ordered by how much they are told. **The EOG waveform:** iCanClean, and MNE's own `EOGRegression`. **Only the blink times:** DSS with a linear bias, and DSS with a non-linear one. **Nothing at all:** iCanClean's pseudo-reference, built from a filtered copy of the EEG itself.

```python
E = du.load_json(du.cache_path("eog_metrics.json"))
T = du.load_npz(du.cache_path("eog_traces.npz"))

# Four arms on stage. The two DSS blink arms are measured and cached too --
# they are in deep dive 03, because they open a second idea mid-act.
SHOWN = ["uncorrected", "EOG regression", "iCanClean\n(EOG electrodes)",
         "pseudo-reference\n(notch 8-30 Hz)"]

with du.presentation_theme():
    fig = du.plot_icanclean_control_panel(
        T["times"], {k: T[k] for k in SHOWN}, E["rows"],
        channel=E["blink_channel_index"], channel_name=E["blink_channel"],
        only=SHOWN,
    )
plt.show()

print(f"{E['n_blinks']} blinks. Before cleaning, the faces-vs-cars N170 at "
      f"{'/'.join(E['n170_roi'])} is {E['baseline_n170_effect_uv']:+.2f} µV.\n")
print(f"{'arm':<34s} {'sees':<15s} {'blink removed':>13s} {'N170':>10s}")
for r in E["rows"]:
    if r["method"] not in SHOWN or r["method"] == "uncorrected":
        continue
    print(f"{r['method'].replace(chr(10), ' '):<34s} {r['information']:<15s} "
          f"{r['attenuation_pct']:12.1f}% {r['n170_effect_uv']:+9.2f} µV")
```

*[1 figure(s) rendered here — omitted from this text export]*

```
593 blinks. Before cleaning, the faces-vs-cars N170 at PO7/PO8 is -0.74 µV.

arm                                sees            blink removed       N170
iCanClean (EOG electrodes)         EOG waveform            83.1%     -1.43 µV
EOG regression                     EOG waveform            73.0%     -1.15 µV
pseudo-reference (notch 8-30 Hz)   the EEG itself          92.2%     +0.59 µV
```

> **Say:** the pseudo-reference is the appealing one — no electrodes at all, you notch the
> brain band out of the EEG and use that as the reference. And it wins on attenuation: 92%
> of the blink, against 83% for the recorded EOG electrodes.
>
> It also destroys the experiment. The faces-versus-cars N170 is a *negative* deflection.
> With the recorded electrodes it goes from −0.74 to −1.43 µV — removing blink noise sharpens
> it. With the pseudo-reference it comes back **positive**. The effect is not weakened, it is
> gone, because a reference built by filtering the EEG shares the signal's own low
> frequencies, and the N170 lives there.
>
> On this participant, iCanClean removed substantial blink contamination while retaining the
> expected N170; the arms with the largest attenuation did not. So:
> **artifact attenuation is not a score.**

---
## 5 — You have seen four regimes. The same contract covers the rest.

| What you have | Method | What information it uses |
|---|---|---|
| power-line noise, possibly non-stationary | **ZapLine / ZapLine+** | narrowband spatial structure at the line frequency |
| line noise, conservative + phase-preserving | **SpectrumInterpolation** | the spectral neighbourhood of the peak |
| large transient / movement artifacts | **ASR**, AdaptiveASR, JugglerASR | abnormal covariance vs a clean baseline |
| recorded noise reference channels | **iCanClean** | correlation between scalp and reference channels |
| channel-specific sensor noise | **SNS** | what neighbouring sensors agree on |
| a target response you can define | **DSS** | a bias you declare (trial average, band, period) |

```python
with du.presentation_theme():
    fig = du.plot_contract_screen()
plt.show()
```

*[1 figure(s) rendered here — omitted from this text export]*

> **Say:** different assumptions, different information, different failure modes — but one
> MNE-native estimator contract. MNE objects in, MNE objects out, and in every case the
> fitted object is the thing that let us check the claim.

---

```python
ZapLine(...).fit_transform(raw)
ASR(...).fit_transform(raw)
ICanClean(...).fit_transform(raw)
DSS(bias=...).fit_transform(epochs)
```

## Run exactly what I just ran

```python
from IPython.display import Image, display

# The QR opens this exact notebook in Colab. Regenerate with make_qr.py if the
# repository ever moves.
display(Image(filename="qr_colab.png", width=300))
print("  https://colab.research.google.com/github/snesmaeili/mne-denoise-meta-demo")
print("  package:  github.com/mne-tools/mne-denoise")
print()
print("  5 deep dives in this repo cover every method and every control cut for time:")
print("     01 ZapLine   02 ASR (all four variants)   03 iCanClean   04 DSS   "
      "05 SpectrumInterpolation")
```

*[1 figure(s) rendered here — omitted from this text export]*

```
  https://colab.research.google.com/github/snesmaeili/mne-denoise-meta-demo
  package:  github.com/mne-tools/mne-denoise

  5 deep dives in this repo cover every method and every control cut for time:
     01 ZapLine   02 ASR (all four variants)   03 iCanClean   04 DSS   05 SpectrumInterpolation
```

---
---

# Appendix — the ASR family in full

*Not part of the five minutes. Run these during questions or at the sprint table.*

## A1 — What ASR accepts

```python
import numpy as np, mne
from mne_denoise.asr import ASR

rng = np.random.default_rng(RANDOM_STATE)
info = mne.create_info([f"EEG{i:03d}" for i in range(8)], 250.0, "eeg")
raw_demo = mne.io.RawArray(rng.standard_normal((8, 2500)), info, verbose="ERROR")
raw_demo.filter(1.0, None, verbose="ERROR")   # ASR's documented precondition
arr = raw_demo.get_data()
epo_demo = mne.make_fixed_length_epochs(raw_demo, duration=2.0, preload=True, verbose="ERROR")

for label, obj, kw in [
    ("ndarray (n_ch, n_times)", arr, dict(sfreq=250.0)),
    ("mne.io.Raw", raw_demo, {}),
    ("mne.Epochs", epo_demo, {}),
    ("mne.Evoked", epo_demo.average(), {}),
]:
    try:
        out = ASR(**kw).fit_transform(obj if not hasattr(obj, "copy") else obj.copy())
        print(f"   {label:<26s} -> {type(out).__name__}")
    except Exception as exc:
        print(f"   {label:<26s} -> {type(exc).__name__}: {str(exc)[:58]}")
```

```
   ndarray (n_ch, n_times)    -> ndarray
   mne.io.Raw                 -> RawArray
   mne.Epochs                 -> Epochs
   mne.Evoked                 -> ValueError: ASR.fit() does not support Evoked calibration data
```

> `Evoked` is rejected for *calibration* by design — there is no within-trial variability to
> estimate a clean covariance from. A 2-D array needs `sfreq`; MNE objects carry their own.

## A2 — What you can switch

```python
import inspect
from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR, GuidedASR

def params(cls):
    return {n: p.default for n, p in inspect.signature(cls.__init__).parameters.items()
            if n != "self"}

sets = {c.__name__: params(c) for c in (ASR, AdaptiveASR, JugglerASR, GuidedASR)}
shared = set.intersection(*(set(v) for v in sets.values()))

print(f"{'estimator':<14s} {'params':>7s}   variant-specific knobs")
print("-" * 78)
for name, ps in sets.items():
    own = sorted(set(ps) - shared)
    print(f"{name:<14s} {len(ps):>7d}   {', '.join(own)}")
print(f"\n{len(shared)} parameters are shared by all four — one contract, four calibration"
      f" strategies.")
```

```
estimator       params   variant-specific knobs
------------------------------------------------------------------------------
ASR                 31   calibration, cov_estimator, experimental, filter_kind, method, window_overlap
AdaptiveASR         31   learning_rate, mw_mode, mw_window_length, tau, update_window_length, variant
JugglerASR          35   cov_estimator, dbscan_eps, dbscan_min_samples, dbscan_top_k, filter_kind, gev_grid_size, min_reference_fraction, selection_filter_kind, strategy, window_overlap
GuidedASR           34   artifact_biases, calibration, cov_estimator, experimental, filter_kind, guidance_strength, preserve_biases, reconstruction, window_overlap

25 parameters are shared by all four — one contract, four calibration strategies.
```

The package's own decision guide (`docs/asr.rst`):

- **reference-compatible start** — `ASR(method="standard")`, then validate the cutoff
- **robust calibration with a working cutoff** — `ASR(method="riemannian_windowed")`
- **online / streaming BCI** — `AdaptiveASR(variant="psw")` or `"psp"`
- **extreme MoBI / high motion** — `JugglerASR(strategy="gev")` or `"dbscan"`

## A3 — Validated against the MATLAB originals

```python
# This demo lives in its own repository, so ask the installed package where
# its source tree is rather than assuming a directory layout.
root = du.mne_denoise_root()
if root is None:
    print("mne-denoise is installed from a wheel; parity fixtures ship with the "
          "source checkout only.")
else:
    fixtures = sorted((root / "tests" / "parity" / "matlab_reference").glob("*.mat"))
    tests = sorted((root / "tests" / "parity").glob("test_*.py"))
    print(f"{len(fixtures)} MATLAB reference fixtures, {len(tests)} parity test modules:")
    for t in tests:
        print(f"   {t.name}")
```

```
38 MATLAB reference fixtures, 6 parity test modules:
   test_aasr_parity.py
   test_asr_parity.py
   test_asr_python_references.py
   test_dss0_parity.py
   test_riemannian_windowed_parity.py
   test_zapline_parity.py
```

> **Say:** the variants are not reimplementations we hope are right — standard ASR, adaptive
> ASR and the Riemannian backend are each pinned against fixtures generated from the original
> MATLAB code.

## A4 — Where each variant comes from

| Variant | Source | Regime it was built for |
|---|---|---|
| `ASR(method="standard")` | Kothe & Jung 2016; Chang et al. 2020 | transient bursts on ordinary EEG |
| `ASR(method="riemannian_windowed")` | Blum et al. 2019 | calibration windows themselves contaminated |
| `AdaptiveASR(variant=...)` | Tsai et al. | non-stationary recordings, streaming BCI |
| `JugglerASR(strategy=...)` | Kim et al. 2025 | extreme MoBI — 205-channel juggling EEG |

Blum's rASR was developed on 24-channel *mobile* EEG with gyroscope, accelerometer and GPS
streams, indoors and outdoors — not sleep. The sleep ASR toolbox is `dusk2dawn`
(Somervail et al. 2023), a separate lineage that happens to embed Blum's Riemannian option.
