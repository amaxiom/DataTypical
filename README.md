# RobustPixelMaker (RPM)

[![License: MIT](https://img.shields.io/badge/license-MIT-440154.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-414487.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.2.0-31688e.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-217-21918c.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-98%25-22a884.svg)](tests/)
[![PyPI](https://img.shields.io/pypi/v/robustpixelmaker.svg?color=2a788e)](https://pypi.org/project/robustpixelmaker/)

**Reproducible, leakage-safe feature selection that works directly on image pixels.**

RPM finds the informative region of a set of registered scientific images, working on the
pixels themselves with no collapse to engineered features, and returns a robust predictor
that reads only that region, a reproducible selected region, and an honest performance
estimate. It is the image sibling of
[RobustModelMaker](https://github.com/amaxiom/RobustModelMaker).

```bash
pip install robustpixelmaker
```

- Literature positioning and novelty analysis: [RELATED_WORK.md](RELATED_WORK.md)
- Worked scientific examples: [examples/EXAMPLES.md](examples/EXAMPLES.md)
- Benchmark protocol and results: [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md),
  with the conclusions and negative results in
  [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md)

## Why

Scientific images are mostly not signal: a micrograph is largely substrate, an MRI largely
skull and background, a microscopy field largely plasma. Training on all of it spends
capacity on noise and invites the model to learn the background instead of the object.

The usual answers disappoint. Ranking pixels by variance keeps whatever fluctuates most,
which is not what predicts: the shipped control plants two regions of equal variance and
only one drives the label. Post-hoc saliency explains one fitted model and moves when you
retrain. Collapsing to radiomics or ROI features pre-commits the representation, so the
selection then partly describes the recipe rather than the data.

RPM asks instead which pixels are predictively sufficient *and reproducibly so*, and
returns a region together with a number saying how much to trust it.

## Quick start

```python
from robustpixelmaker import RobustPixelMaker

# X: (n_images, height, width) in a common coordinate frame. y: labels or values.
rpm = RobustPixelMaker(registered=True, random_state=0).fit(X, y)

rpm.summary()             # nested-CV score, stability, and selection diagnostics
rpm.pi_map()              # selection-frequency heatmap, in image space
rpm.stability_report()    # ambiguous_fraction: does a reproducible region exist?
rpm.coverage()            # fraction of the image retained
rpm.predict(X_new)        # the robust predictor, reading only the selected region
rpm.save("run_output/")
```

Task (binary, multiclass, regression) is inferred from `y`. The one argument to own
explicitly is `registered`: a single shared mask only means something when the images
share a coordinate frame, so `registered=False` raises rather than guessing.

The model fitted on the selected region is chosen with `model=`, independently of the
selection itself:

```python
RobustPixelMaker(model="rf")                    # a random forest on the selected patches
RobustPixelMaker(model=GradientBoostingClassifier())   # or any sklearn estimator
```

`None` (the default) keeps the mask's own head, which is what every published result
here used. Named options are `mask`, `rf`, `logreg`/`ridge`, `gb`, `svm`, `knn`. The
choice never changes WHICH patches are selected, only what is fitted on them, and a test
pins that invariant.

The mask's own head is configured separately, and these reach every selector:

| argument | default | what it does |
|---|---|---|
| `head` | `"linear"` | `"mlp"` adds one hidden layer, so a gate can be kept for an interaction |
| `hidden` | `16` | width of that layer |
| `lam_scale` | `"fixed"` | `"auto"` searches `lam` instead of trusting it, needed when an expressive head leaves every gate open |
| `auto_target` | `0.15` | with `"auto"`, the fraction of the mask to aim to leave open |

`auto_target` is a **prior on coverage**, not a neutral knob, and it is **coarse**:
the lam ladder has six geometric rungs, so 0.25 and 0.40 select the same rung and give
identical answers. The default was 0.4 and is now 0.15, because at 0.4 against a true
informative fraction of 0.062 recovery F1 on the shipped control halved, 1.000 to
0.441, while 0.15 recovers it exactly and still closes a mask that a fixed `lam`
leaves wide open (FINDINGS sec.19, sec.21). Set it near the coverage you expect. If
you have no expectation, `target_coverage` is the more direct control and skips the
search. It is a request rather than a guarantee: `pi_` takes at most `n_bootstrap`+1
distinct values, so the reachable coverages are quantised and a heavily tied map can
make the nearest one far from what you asked. `stability_report()["coverage_target_met"]`
says whether the target was reached, and a miss of more than 25% warns (FINDINGS
sec.22). `top_patches(k)` forces exactly k patches when you need the match enforced.

Three selector strengths are available via `selector=`:

| `selector` | what it does | when |
|---|---|---|
| `"single"` | one mask fit | fast look, large clean data |
| `"bootstrap"` | selection frequencies over resamples | the default working choice |
| `"ensemble"` | double bootstrap over resamples **and** representations | when the answer must not depend on the lens |

## How it works

1. **A learnable patch mask.** Hard-Concrete L0 gates over a patch grid, trained in two
   phases: fit the head on all patches, freeze it, then prune. Freezing matters, because
   trained jointly the head simply grows to cancel a shrinking mask, and the sparsity
   penalty collapses the mask without changing predictions.

   The head is `linear` by default. `head="mlp"` adds one hidden layer, so a patch can
   earn its place through an interaction rather than a main effect. It is opt-in, costs
   more, and is **not** uniformly better: it recovers a saturating rule the linear head
   misses entirely (5.0x chance against 0.000), and it does not rescue the purely
   interactive rule, where it reaches 1.2x chance and the linear head reaches zero
   (FINDINGS sec.16, sec.19). That limit is structural rather than a tuning problem,
   because the freeze-base trick prunes any gate that does not help a head fitted with
   every gate open, and on a pure interaction no single gate does.

   On real images it reads as the weakest arm in the headline table, 0.876 against
   0.950, but that is a coverage difference and not a method difference: it needs
   `lam_scale="auto"` to close its mask at all on multiclass data, which lands it near
   9% coverage where the other selectors sit near 33%. Compared at **matched**
   coverage it is statistically indistinguishable from random-forest importance on
   score (p = 0.28), while losing clearly on stability, 0.47 against 0.75, in 19 of 20
   operating points. Its earlier and better-looking real-data numbers were largely the
   full-image baseline, because it had kept every patch (sec.20, sec.23).
2. **Bootstrap stability selection.** Meinshausen-Buehlmann selection frequencies over
   resamples (bootstrap, half-subsample, or Shah-Samworth complementary pairs), so a patch
   that survives only on one particular sample drops out.
3. **Marginalisation over representations.** The distinguishing step: the selection is
   integrated over an ensemble of lenses as well as over data resamples, treating the
   representation as a nuisance variable rather than a choice to defend.
4. **Absent, not zero.** Dropped regions use renormalised (partial) convolution, so a
   mask boundary does not fabricate an edge.

Selection happens at patch granularity rather than per pixel, deliberately: neighbouring
pixels are near-equivalent, so a pixel-level mask never stabilises.

## Modules

| module | purpose |
|---|---|
| `core.py` | the `RobustPixelMaker` facade and `run_pipeline` |
| `nested_cv.py` | leakage-safe nested CV, task inference, splitters, the fold-estimator contract |
| `masking.py` | `SoftMaskSelector`: the learnable mask, analytic gradients, L0 or L1 gates, linear or MLP head |
| `downstream.py` | the model fitted on the selected region, chosen with `model=` |
| `synthetic.py` | `make_synthetic_images`: the control with a known informative region and an equal-variance decoy |
| `selection.py` | `BootstrapMaskSelector` and `RepresentationEnsembleSelector` |
| `representations.py` | the lenses: patch means, texture statistics, scales, random convolutions |
| `maskfill.py` | renormalised absent masking and the `fabricated_edge` diagnostic |
| `regimes.py` | `PatchGrid` granularity and input validation |
| `metrics.py` | scoring, chance-corrected stability, paired baseline verdicts |
| `reproducibility.py`, `config.py` | deterministic seed streams, determinism switches, device detection |
| `results.py` | the result container and its JSON/CSV/npy exports |

## Evidence, including what did not work

The library ships its negative results, because a selection method that cannot say "no
region here" is not trustworthy:

- **Bootstrap aggregation is a stability estimator, not a generator.** On a well-powered
  compact task the frequency map is sharply bimodal (ambiguous fraction 0.14); on an
  n=780 ultrasound set it is diffuse (0.57), and more resampling does not manufacture a
  region. `stability_report()` exposes this as a go/no-go diagnostic.
- **Raw Jaccard stability is confounded by coverage**, so `adjusted_jaccard` is provided
  and preferred when comparing operating points.
- **Random-forest importance matches RPM on real-data score** and is leaner (33%
  coverage against 45%). Paired under the suite's own judge the two are statistically
  indistinguishable (p = 0.87), so the earlier claim that RF was better was reading an
  unpaired difference of 0.002 as a result.
- **The choice of evaluator moves the verdict by more than the methods differ.** Scoring
  the same selections with a panel of four downstream models shifts the RPM-minus-RF
  margin significantly (p < 0.0001 for a linear judge), which is why every benchmark now
  reports the panel rather than a single random forest. FINDINGS sec.14.
- **The learned mask adds nothing over RF importance under the same wrapper.** Holding
  the bootstrap aggregation fixed and swapping only the base ranker: +0.0003, p = 1.000,
  and the mask is the less stable of the two (0.67 against 0.89). Every earlier
  comparison confounded the mask with the wrapper around it. FINDINGS sec.15.
- **A hyperparameter search inside the aggregation quietly spent its stability.**
  `lam_scale="auto"` was forwarded into every bootstrap resample, so each replicate
  chose its own penalty and the frequency map mixed "which patches matter" with "which
  `lam` this replicate landed on" while still being thresholded as the former.
  Calibrating once per fit costs one extra fit, removes `B` ladders (at least 3.4x
  faster), and the stability gain it bought, +0.081 at p = 0.097, is recorded as a
  direction rather than a result. FINDINGS sec.18.
- **A request that could not be met used to report as though it had been.**
  `target_coverage` asks for a coverage the frequency map may not be able to reach: on
  one dataset the lowest reachable value was 0.245 and a request for 0.12 silently
  returned double it. A shipped example relied on that to compare two datasets at
  matched coverage, and was comparing 0.245 against 0.122 while saying otherwise. The
  conclusion survived enforcement, but it had not been tested as stated. Now reported,
  warned, and enforced with `top_patches`. FINDINGS sec.22.
- **A method compared at its own operating point is not a method comparison.** The
  headline tables report each selector at whatever coverage it lands on, which is right
  for them to report and is not a like-for-like test whenever the coverages differ. The
  MLP arm reads as worst in the table at 0.876 and is indistinguishable from RF at
  matched coverage (p = 0.28). `run_coverage_frontier.py` is the instrument for the
  comparison, and it flips the verdict. FINDINGS sec.23.
- **A documented option the documented entry point could not set.** `head="mlp"` was
  described in this README while `RobustPixelMaker` took no `head` argument and
  forwarded none, so the only way to use it was to bypass the facade. Fixed, with
  `hidden`, `lam_scale` and `auto_target`, and validated at construction. FINDINGS
  sec.19.
- **The shipped synthetic control is linear, and the mask's head was linear.** The
  benchmark where RPM wins largest is one where its inductive bias is correct by
  construction; on interactive labels the linear head recovered exactly nothing. The
  nonlinear label axis is now part of even the smoke tier. FINDINGS sec.16.

## Development

Requires Python 3.9+ with numpy, scipy and scikit-learn; `pip install -e ".[all]"` adds
joblib, matplotlib, pandas and pytest. `torch` is **not** required: everything runs on
CPU with analytic gradients, and device detection falls back gracefully.

```bash
pip install -e ".[all]"
pytest tests -q
```

The package ships a `py.typed` marker, so its annotations are used by your type
checker. 81% of the public surface is annotated and the package checks clean under
mypy, which a test enforces: an unclean package would put its own diagnostics into
the output of anyone checking their code.

217 tests, 98% statement coverage (97.9% of 1422 package statements). Benchmarks reproduce with one command:

```bash
python benchmarks/run_all.py --tier smoke
```

`--tier standard` reproduces the FINDINGS configurations, `--tier full` adds every
registered dataset. Benchmark datasets download and cache on first use.

## The RobustMaker family

RPM is one of three **RobustMaker** libraries, which apply the same method to three
kinds of data:

| library | selects over | repository |
|---|---|---|
| RobustModelMaker (RMM) | tabular feature columns | [amaxiom/RobustModelMaker](https://github.com/amaxiom/RobustModelMaker) |
| **RobustPixelMaker (RPM)** | image pixels and patches | [amaxiom/RobustPixelMaker](https://github.com/amaxiom/RobustPixelMaker) |
| RobustSignalMaker (RSM) | signals and spectra | [amaxiom/RobustSignalMaker](https://github.com/amaxiom/RobustSignalMaker) |

All three do leakage-free bootstrap stability selection, return a reproducible selected
set with an explicit stability number, and report an honest nested-CV score. They share
deterministic seeding, the same result-object shape, and a policy of publishing the
negative results alongside the positive ones.

They also share defects, which is worth knowing: a bug found in one is worth hunting in
the others, and a fix verified in one is **not** verified in the others. Both directions
have happened here. An adversarial sweep found eight defects in RPM by deliberately
hunting the families recorded for a sibling, and a later one records a defect that
travelled the other way along a port, plus a contrast-normalisation rule that is correct
in RPM and measurably wrong in RSM because their lens sets and criteria differ. Both are
written up in CHANGELOG.md under the versions that fixed them.

Two further libraries by the same author share the contract without being RobustMakers:
[BenchMake](https://github.com/amaxiom/benchmake) for archetypal train/test splits, and
[MissLearn](https://github.com/amaxiom/MissLearn) for learning from incomplete data.

## Licence

MIT. Copyright 2026 Amanda S. Barnard.
