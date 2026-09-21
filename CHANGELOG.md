# Changelog

All notable changes to RobustPixelMaker. Versions follow semantic versioning.

## 0.2.0 (2026-09-15)

A minor version rather than a patch, because one default moves: `auto_target` goes from
0.4 to 0.15, which changes results for anyone using `lam_scale="auto"`. Everything else
is additive, and **selection behaviour is unchanged by default**. Every option added
here is opt-in, and the defaults are pinned by tests asserting they reproduce previous
output exactly.

The full standard-tier suite was rerun to confirm that: across 957 paired rows, score,
coverage and stability are identical to the previous release for every arm except
`rpm_mlp`, and identical on all 759 synthetic rows including that one. `rpm_mlp`'s
earlier real-data numbers were substantially the full-image baseline, and its corrected
numbers are lower and honest (see the evidence section).

The other theme is that four separate calls could not do what was asked and returned
something plausible instead of saying so: a penalty search that ran inside the
aggregation it fed, a mask that never closed, a default prior that quietly imposed the
wrong operating point, and a threshold that could not be reached. All four are fixed or
reported. The Fixed entries below record each one.

### Added

- **Type annotations on the public surface, with a `py.typed` marker.** 81% of public
  functions are now fully annotated, up from 32%, and the package type checks clean
  under mypy. Both halves matter: `py.typed` makes RPM's annotations part of every
  user's type-check run, so a diagnostic in RPM's own source surfaces in the output of
  anyone checking their code. Before the cleanup a consumer type checking a six-line
  script saw 44 errors, 43 of them ours. The marker and the clean run are one
  commitment, and two tests pin it. `mypy` is now a `dev` extra.
- **`CITATION.cff`**, so GitHub offers the "Cite this repository" button, matching
  RobustSignalMaker. The ORCID line is present and commented, ready to fill in.
- **`model=` on `RobustPixelMaker`**, the parity gap with RobustModelMaker. The model
  fitted on the selected region is now a choice: `None` (the mask's own head, the
  default and previous behaviour), a name from `mask`, `rf`, `logreg`, `ridge`, `gb`,
  `svm`, `knn`, or any scikit-learn estimator, which is cloned rather than mutated
  across folds. It is fitted inside the same leakage-safe folds. A test pins the
  invariant that makes it safe: choosing a model never changes WHICH patches are
  selected, only what is fitted on them. New module `downstream.py`.
- **`head="mlp"` on `SoftMaskSelector`** and the aggregators: one hidden layer under the
  same freeze-base scheme, with hand-derived gradients verified against central
  differences to 5e-9. Repairs the case where a purely interactive signal made the
  linear head prune the entire informative region (FINDINGS sec.16, 17).
- **`lam_scale="auto"`**, which searches `lam` for a mask that actually discriminates
  rather than trusting a fixed penalty weight. Needed because a more expressive frozen
  head uses every input, so at the linear head's `lam` the MLP mask never closed at all.
  `lam_used_` reports the operating point.
- **`ranking()` and `top_patches()`** on the aggregators, breaking selection-frequency
  ties on `strength_`, the mean gate value over the same fits. `pi_` takes at most B+1
  values and saturates, so a small top-k request previously returned patches in index
  order. Worth +0.11 AUC at 5% coverage on MNIST.
- **`score_panel`** in the benchmark suite: every selection scored by four downstream
  models spanning different inductive biases, not one random forest.
- **`run_coverage_frontier.py`**, `run_nonlinear_control.py`, `run_timing.py`,
  `machine_state.py`, and the `rf_boot` and `rpm_mlp` competitor arms.
- **Every benchmark summary table now carries a `selects` column**, `no` wherever an
  arm kept the whole image (coverage at or above 0.999), with a note saying such a row
  must be compared against `full` rather than against the selectors. The suite
  published an arm that selected nothing and no column said so: `rpm_mlp` returned
  coverage 1.000 on both multiclass datasets, scoring the full-image score with
  stability 1.000, and read down the score column it looked tied for best. A metric
  that rewards not selecting will be won by a method that does not select, and the
  protection is a column in the table rather than knowing about the trap. `rfe`,
  `anova` and `lasso` reach coverage 1.000 on some datasets legitimately and are
  marked identically. FINDINGS sec.20.

- **`head`, `hidden`, `lam_scale` and `auto_target` reach every selector and the
  `RobustPixelMaker` facade.** The README documented `head="mlp"` as a feature while
  the facade took no such argument and forwarded none, so the documented option was
  unreachable from the documented entry point: using it meant bypassing the facade and
  constructing a selector by hand. All four are now plumbed through the facade, both
  fold estimators and both aggregators, and validated at construction in each, because
  `_one_fit` deliberately swallows resample-fit exceptions and would otherwise report a
  mistyped `head` as "every one of the N resample fits failed". FINDINGS sec.19.
- **`coverage_realised_` and `coverage_target_met_` on both aggregators**, reported in
  `stability_report()`, with a warning when `target_coverage` is missed by more than 25%
  relative. `pi_` takes at most B+1 distinct values, so reachable coverages are
  quantised and ties make the grid coarse: on blood at B=12 the lowest reachable
  coverage is 0.245, so a request for 0.12 silently returned double it.
  `coverage_target_met_` is None when nothing was requested, so no request never reads
  as a failed one. FINDINGS sec.22.
- **Documentation guards**, because three documents had rotted quietly. Every doc
  quoting "N tests" is checked against `pytest --collect-only`; every backtick-quoted
  name in a README argument table must be a parameter the facade accepts; forwarding is
  checked separately from acceptance, since accepting an argument and dropping it before
  it reaches the selector is the half-fix that would keep the README true and change
  nothing; the benchmarks method table must list every registered selector; and a count
  written beside a list must match the list.
- Regression tests in `tests/test_maskfill.py` pinning the above: a level filter must
  return the constant on constant data for four kernel shapes including three
  non-uniform ones, contrast filters must return exactly zero on constant data for
  non-uniform kernels, the denominator is checked to be mass and not count on a kernel
  chosen so the two rules disagree, the contrast branch is checked against a directly
  computed L2 reference, a window with evidence but no filter mass must be invalid,
  and uniform kernels must be unchanged. Every pre-existing level-filter test used
  `ones / 9`, the one kernel shape for which the old code was right, which is why the
  suite stayed green through a release.

### Changed

- **`auto_target` now defaults to 0.15, not 0.4.** It is a prior on coverage used by
  `lam_scale="auto"`, and 0.4 is far too high for sparse selection, which is what this
  library is for. At 0.4, against a true informative fraction of 0.062, recovery F1 on
  the shipped synthetic control halved (1.000 to 0.441) and coverage inflated to 0.234.
  At 0.15 the control is recovered exactly as well as under a fixed `lam` (F1 1.000,
  coverage 0.062) while a multiclass mask that a fixed `lam` leaves wide open still
  closes. There is no trade-off between the two; the old default was simply wrong.
  Note the dial is coarse: the `lam` ladder has six geometric rungs, so 0.25 and 0.40
  select the same rung and give identical answers. FINDINGS sec.21.

### Fixed

- **Five places where the code knew a type the signature did not**, all found by type
  checking rather than by testing, and all fixed by stating the invariant rather than
  silencing the checker: `classes` is now derived inside the branch that indexes it
  instead of beside a separate condition; `mask_img` is converted once like `X` already
  was, in three fill helpers; `patch_ids` is no longer rebound from an array-like to a
  set; and the two attributes that are only valid after `fit` now say so. None was a
  live bug, and each was a place a reader had to reconstruct an invariant the code
  never stated.
- **`lam_scale="auto"` ran its penalty search inside every bootstrap resample, so
  the aggregation measured two things and reported one.** `_inner_kwargs` forwarded
  `lam_scale` to each resample fit, which meant each replicate chose its own `lam`.
  Stability selection counts how often a patch survives at a FIXED operating point,
  so `pi_` was confounding "which patches matter" with "which `lam` this replicate
  landed on" while still being thresholded and plotted as the former. `lam` is now
  calibrated once per `fit`, on the data `fit` was given (the outer training split
  under nested CV, so no leakage), and held fixed across resamples and lenses.
  Measured on 20 matched-coverage cells: mean pairwise Jaccard 0.375 to 0.455
  (+0.081, better in 13/20, p = 0.097, a direction rather than a result), score
  unchanged (p = 0.70), and at least 3.4x faster because `B` ladders become one.
  `lam_scale="fixed"`, the default, constructs no probe and is unchanged. Two tests
  pin it. FINDINGS sec.18.

- **`maskfill.renormalised_convolve` rescaled by kept pixel COUNT, which is exact
  only for a uniform kernel.** For any other kernel it charges a dropped pixel the
  average weight instead of its own. On a flat image beside a straight gap edge a
  uniform 7x7 kernel was exact, a Gaussian 7x7 sigma 1.5 was 74% wrong and a
  Gaussian 9x9 sigma 1.0 was 99.9% wrong, fabricating exactly the edge this module
  exists to prevent, and reporting it as valid. The rescaling is now by kept filter
  MASS, and the correct mass differs by branch:
  - LEVEL filters use the observed filter weight `sum_kept f / sum f`. This is an
    identity, not a tuning choice: on constant data the numerator is `x * mass_kept`,
    so the constant comes back exactly for any kernel.
  - CONTRAST filters use the observed L2 norm `sqrt(sum_kept f^2 / sum f^2)`. Signed
    weights cancel, and the absolute mass is not the right substitute either: both
    count and L1 mass go as N/c for a random kept subset and so over-correct by
    roughly a square. Measured over 8 seeds recovering the full-data response of a
    random zero-mean 5x5 filter on smooth data, L2 gives 79.9% RMS error against
    count's 93.2% under 30% scattered dropout, and 57.3% against 75.2% under patch
    dropout. Level filters improve 2.2x (scattered) and 1.4x (patch dropout).
- **Validity required only one kept pixel, not enough filter mass to divide by.** A
  window whose kept pixels all sit where the kernel is near zero now returns 0.0 with
  `valid` False instead of a large number manufactured from a near-zero denominator.

  Uniform kernels are bit-comparable before and after, which is what makes this a fix
  rather than a change of definition, and a regression test pins that. No benchmark
  result changes: `renormalised_convolve` is called only by `masked_conv_features`
  and the `fabricated_edge` diagnostic, neither of which feeds the recorded runs, and
  the M3 diagnostic is unchanged (renorm 4.8e-16 against zero-fill 1.91) because on a
  flat image the contrast re-centring returns zero whatever the scaling.

  Found by porting the sibling fix from RobustSignalMaker, which hit the same defect
  on 2026-09-01. RSM normalises its contrast branch by L1 mass and keeps doing so:
  the L2 rule was ported there on 2026-09-07 and lost on that library's own criterion
  (truth recovery of the selection, not fidelity of the response), 21 paired wins to 9
  with 10 ties, because its default lens set includes a Savitzky-Golay derivative for
  which the exchangeable-residual argument behind L2 does not hold. An earlier draft of
  this entry called RSM's calibration "weaker", which the measurement refutes; the two
  libraries differ deliberately, each matching its own lens set. The level half is
  correct in both.
- The frequency map's tie-breaking, the MLP head's non-pruning, and a one-off gradient
  rescale that was tried, measured, found to move the cross-head `lam` ratio only from
  3.0x to 2.5x, and removed rather than kept because it looked principled.
- **`target_coverage` missed silently, and the test that should have caught it
  permitted the miss.** `examples/ex4` compares the SHAPE of two selections, which only
  means something at matched coverage, and asked for 0.12 on both datasets. It got
  0.245 on blood and 0.122 on pneumonia, and described the comparison as matched.
  Contiguity is exactly what coverage inflates: pneumonia scores 0.50 at coverage 0.122
  and 0.83 at 0.245. Forced to exactly 6 patches each with `top_patches`, blood is 1.00
  and pneumonia 0.50, the same gap, so the claim was true and untested as stated. The
  test was named `..._hits_requested_operating_point` and asserted
  `abs(coverage - target) < 0.15`, which at target 0.1 admits a miss larger than the
  target itself. It now asserts the real contract, that no reachable coverage is nearer
  than the one chosen. FINDINGS sec.22.

- **Six source lines in `FINDINGS.md` cited run logs that are not published.** Console
  logs are gitignored, 879 KB of transcript whose substance is already in the CSVs, so
  all six resolved on this machine and nowhere else. Two of them were the only cited
  evidence for sec.20. Fixed in both directions: four now cite the tracked CSV that
  already carried the numbers, and the three small logs that carry numbers nothing else
  does are now tracked by an explicit `.gitignore` exception. A test refuses any
  citation of an unpublished file.
- **Twenty-seven result files had no provenance entry.** `benchmarks/results/` holds
  runs from three months and several defaults, and the files do not say which is which:
  `run_coverage_frontier.py` alone was run six times, every run writing the same generic
  title and the same table shape, three of them superseded by a later fix. Opening the
  wrong one returns a number that was true of a version that no longer exists.
  `PROVENANCE.md` now names every tracked file with its date and its section, and a test
  fails on any that it does not.

### Changed, in the evidence rather than the code

Three conclusions in `FINDINGS.md` were corrected by better experiments, and the
corrections matter more than the additions:

- **Section 14**: the suite scored every selection with a random forest, which shares a
  model family with the `rf` competitor. Judged by a linear model instead, the
  RPM-minus-RF margin shifts significantly (p < 0.0001). Section 6's verdict that RF
  won was reading an unpaired 0.002 difference; paired, the two are indistinguishable
  (p = 0.87).
- **Section 15**: no comparison in the suite separated the learned mask from the
  bootstrap wrapper around it. Held fixed, the mask adds nothing over RF importance on
  real data (+0.0003, p = 1.000) and is less stable (0.67 against 0.89).
- **Section 16**: the shipped synthetic control's label is linear and the mask's head
  was linear, so the benchmark where RPM wins largest was one where its inductive bias
  is correct by construction. On interactive labels the linear head recovered exactly
  nothing. The nonlinear axis is now a standing component of the smoke tier.
- **Section 20**: the suite published an arm that selected nothing. `rpm_mlp` returned
  coverage 1.000 on both multiclass datasets, scoring the full-image score with
  stability 1.000, and read down the score column it looked tied for best. Corrected,
  its real-data score falls from 0.954 to 0.912 on blood and from 0.984 to 0.922 on
  organa, at coverage 0.105 and 0.087. The lower numbers are the honest ones.
- **Section 23**: comparing methods at their own operating points is not a method
  comparison. `rpm_mlp` reads as worst in the headline table, 0.876 against 0.950, while
  sitting at roughly a quarter of everyone else's coverage. At matched coverage it is
  statistically indistinguishable from random-forest importance (p = 0.28), though it
  loses clearly on stability in 19 of 20 cells. `run_coverage_frontier.py` is the
  instrument for that comparison, and it flips the verdict.

## 0.1.0 (2026-09-02)

First release. Registered-regime (`registered=True`) pixel selection, complete with the
evaluation and packaging the rest of the RobustMaker family shares.

### Added

- **`RobustPixelMaker`** estimator facade and the `run_pipeline` function, with
  `fit`/`predict`/`predict_proba`, `summary`, `coverage`, `pi_map`, `stability_report`,
  `selected_pixels`, `compare_to_baseline` and `save`.
- **Leakage-safe nested cross-validation** (`nested_cv.py`) in which the selection is
  refitted inside every outer fold, with task inference, `GroupKFold` support and
  out-of-fold prediction accumulation.
- **Learnable patch mask** (`masking.py`): Hard-Concrete L0 gates (default) or
  sigmoid with L1, trained VTF-style by freezing the head before pruning the mask.
  Binary, multiclass and regression, with an optional spatial-smoothness prior.
- **Bootstrap stability selection** (`selection.py`): per-patch selection frequencies
  over bootstrap, half-subsample or Shah-Samworth complementary-pair resamples, an
  explicit `target_coverage` operating point, and the `ambiguous_fraction` go/no-go
  diagnostic for whether a reproducible region exists at all.
- **Representation marginalisation** (`representations.py`, `RepresentationEnsembleSelector`):
  the double bootstrap over data resamples and lenses, with per-lens frequency maps and
  a measured `representation_agreement`.
- **Renormalised absent masking** (`maskfill.py`): partial convolution extended with
  automatic filter re-centring for zero-mean contrast filters, plus the
  `fabricated_edge` diagnostic on a known-answer flat image.
- **Chance-corrected stability** (`adjusted_jaccard`, `expected_jaccard`), because raw
  Jaccard rewards simply retaining more of the image.
- **Reproducibility controls** (`reproducibility.py`, `config.py`): deterministic
  per-stream seed derivation, `enable_determinism`, and device detection that falls
  back to CPU with a warning rather than failing.
- Six worked scientific examples, a nine-component benchmark suite runnable with one
  command, 142 tests at 98% statement coverage at the time of that release, and MIT licensing.

### Known limitations

- `registered=False` (per-image conditional masks for unregistered fields of view)
  raises `NotImplementedError`; it is a documented extension point, not a silent guess.
- Input is single-channel: RGB images must be converted before use.
- No torch or GPU backend yet. The numpy backends are designed to be extended by one,
  not replaced.
- No corruption or distribution-shift robustness delta is reported yet.

### Notes on evidence

Benchmark conclusions, including the negative results (bootstrap aggregation does not
rescue an underpowered study; a random forest selected leaner regions than RPM on two
of the shipped datasets; the coverage confound in raw stability), are recorded in
`benchmarks/FINDINGS.md`. Two adversarial bug sweeps and every defect they found are
recorded in the Fixed entries above.
