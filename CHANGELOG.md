# Changelog

All notable changes to DataTypical are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.1] - 2026-09-23

A bug-fix release. The first entry makes archetypal analysis work again on a
current NumPy; without it, `pip install datatypical==0.8.0` produces a library
whose headline method raises.

### Fixed

- **`archetypal_method='aa'` failed on every fresh install.** `py_pcha` calls
  `np.mat`, which NumPy removed in 2.0, so a fresh `pip install` (which pulls
  NumPy 2.x) made PCHA raise and v0.8.0 correctly refused to substitute another
  method. The alias is now restored before `py_pcha` is imported. `np.mat` is
  exactly `np.asmatrix`, so PCHA returns bit-identical archetypes: verified at
  0.9160 and 0.6489 on the same data under NumPy 1.26.4 and 2.4.6.

  **This was worse before v0.8.0.** Through v0.7.7 the same failure fell through
  to ConvexHull silently, so a fit requested as archetypal analysis could return
  ConvexHull output with nothing to distinguish it. **Any `'aa'` result produced
  by v0.7.7 or earlier on NumPy 2 is a ConvexHull approximation.** Audit a saved
  fit without re-running it: `nmf_model_` set means NMF ran, `nmf_model_` None
  with a numeric `reconstruction_error_` means PCHA, both absent means
  ConvexHull.

  No version pin was added. Pinning `numpy<2` would strand every user on a
  superseded NumPy to work around one removed alias in a transitive dependency.

- **`stereotypical_rank` was scale-dependent.** The normalisation was guarded by
  `max_dist > 1e-12`, an absolute threshold applied to a user-supplied column in
  the user's own units. A rank is scale-free, but any stereotype column whose
  spread fell below that (concentrations, mole fractions, probability
  differences) took the else branch and every sample was assigned rank 1.0, so
  every sample was reported as maximally stereotypical. No warning was emitted.
  The threshold is now relative to the column's own magnitude, and the collapse
  case warns. A column rescaled by 1e-13 now produces an identical ranking.

- **`exact_formative_prototypical` absorbed NaN and returned confident values.**
  A NaN cell makes that row's norm NaN, and the selection step `vals > 0.0`
  compares False against NaN, so the affected similarities were dropped rather
  than propagated. The function returned an all-finite, fully rankable vector
  computed from whichever cells happened to be clean, and the NaN-bearing sample
  took an ordinary position in the ranking. Both sibling exact functions
  propagate NaN; only this one hid it. Non-finite input now raises
  `DataTypicalError` naming the count.

### Added

- Tests that assert **which backend actually ran**, not merely that a fit
  succeeded. The v0.8.0 suite had 738 tests and none of them would have caught
  the NumPy 2 failure, because none of them checked `archetypal_backend_`. That
  blind spot is the reason a silent backend substitution can reach published
  results.

---

## [0.8.0] - 2026-09-21

Nineteen fixes and four additions. Two of the defects invalidate results
silently, so anyone who has run `archetypal_method='aa'`, or fitted data whose
values are not of order 1, should read the first two entries.

The version is a minor bump rather than a patch because the release adds
`formative_method`, `archetypal_method='auto'`, the `archetypal_backend_` and
`split_half_rho` attributes, and two public functions. Nothing was removed and
no existing call signature changed, so v0.7.7 code runs unmodified, with the one
deliberate exception recorded under `archetypal_method='aa'` below.

### Fixed

- **Silent substitution of NMF for archetypal analysis.** Through v0.7.7, when
  `py_pcha` was not installed, `archetypal_method='aa'` fell through to
  ConvexHull (itself skipped whenever `n_features > 20`) and then to NMF. The
  only notice was behind `verbose`, which is `False` by default, while the
  verbose method label still read "Archetypal Analysis (PCHA+ConvexHull)". A fit
  reported as archetypal analysis could therefore return NMF output with nothing
  to distinguish it.

  `archetypal_method='aa'` now raises `ConfigError` when PCHA is unavailable or
  fails, rather than substituting another method.

  **Any `'aa'` result produced by v0.7.7 or earlier in an environment without
  `py_pcha` is an NMF approximation and should be re-run.** A saved fit can be
  checked without re-running it: `nmf_model_ is not None` means NMF ran,
  `nmf_model_ is None` with a numeric `reconstruction_error_` means PCHA ran,
  and `nmf_model_ is None` with `reconstruction_error_ is None` means ConvexHull
  ran.

- **The working `dtype` was applied to the raw input, before scaling.** The
  feature matrix was read with `to_numpy(dtype=self.dtype)`, which defaults to
  `float32`, and only then imputed and scaled. Because the scaled matrix lives
  in [0, 1], that early cast bought no memory and cost a great deal:

    * values above roughly 3.4e38 became `inf`, and the fit died inside sklearn
      with a bare "Input X contains infinity or a value too large for
      dtype('float32')" that never named DataTypical or the column;
    * values below roughly 1.2e-38 underflowed to zero, every column then read
      as constant, and the fit died complaining about `nmf_rank`, which points
      at the wrong thing entirely;
    * most quietly of all, a feature carrying its signal beyond the 7th
      significant digit lost that signal outright. A measurement of order 1000
      varying in its 8th digit had its spread collapse to exactly zero and was
      then dropped as a constant column, with no error and, unless `verbose` was
      on, no warning.

  Reading, imputing and scaling now happen in float64, and only the scaled
  result is cast to the working `dtype`. The memory contract is unchanged, since
  the large matrices passed downstream are still stored at the requested
  precision. `dtype` is now a memory setting only, and no longer changes results:
  `float32` and `float64` fits agree.

  DataTypical's ranks are scale-free by construction, and they now demonstrably
  are. Rescaling or offsetting a feature leaves every rank bit-identical, where
  before it moved them by up to 3e-5.

- **Dropping a feature column was announced only under `verbose`, and losing
  every column reported the wrong cause.** Dropping a feature materially changes
  an explainability result, so it now always warns. The warning also separates
  the two cases that were previously conflated: a genuinely constant column, and
  one that varies by less than `MinMaxScaler` can resolve (a range under about
  2e-15), which is real structure being discarded. If every column goes this
  way, the fit now raises naming both lists and what to do about it, instead of
  failing further downstream with a message about `nmf_rank`.

- **The formative Shapley ranking does not converge at the default
  permutation count, and nothing said so.** Two fits of the same data differing
  only in `random_state`, at the documented default of 100 permutations,
  produced archetypal formative rankings with a Spearman correlation of **0.02**
  and a top-ten overlap of **1 of 10**, where chance is 2 of 10. The single most
  formative instance differed between the two. Raising the count to 1000 gave
  rho 0.32, still far from reproducible.

  The cause is not a coding error. The estimator is an unbiased Monte Carlo
  Shapley estimator and the values it produces sum correctly, which is what the
  existing `additivity_error` checks. But summing correctly says nothing about
  whether the *order* of the samples is stable, and the order is what gets
  reported as the formative instances.

  Every Shapley estimate now carries a **`split_half_rho`** in `shapley_info_`:
  the permutations are accumulated in two independent halves and the rank
  correlation between them is reported. It costs one extra array and no extra
  sampling. When it falls below 0.7 a `RuntimeWarning` says the ranking is
  largely sampling noise at that permutation count, and distinguishes the values
  from the ordering so the message cannot be misread.

  **Any formative result computed at the default 100 permutations should be
  re-examined**, checking `split_half_rho` and raising
  `shapley_n_permutations` until it stops moving.

- **Two properties of the archetypal measure that were never stated.** Neither
  is a coding error and neither has been changed, because changing either would
  move every archetypal rank the library has produced. Both now announce
  themselves, and both are documented in the README under "Reading the ranks".

    * **A row that is the minimum in every retained feature ranks last.** After
      MinMax scaling such a row sits at the origin, where neither backend can
      define an archetype membership: both return a zero weight vector. Its
      score then falls to the corner term alone and it lands at the bottom of
      `archetypal_rank`, below every other row, despite being an extreme point.
      Measured on both `nmf` and `aa`. A `RuntimeWarning` now names how many
      rows are affected and says to read their archetypal scores as undefined
      rather than low.

    * **The measure inverts once a single row dominates a feature's range.**
      MinMax scaling compresses the remaining rows against one end, and that end
      is itself a corner of the unit cube, so the corner term rewards the
      compressed bulk as much as the extreme point. Measured across six seeds:
      an outlier at 3 to 5 standard deviations ranks above the cloud median
      every time; one at 30 standard deviations ranks *below* it every time. A
      `RuntimeWarning` now fires when the middle 98% of any feature spans under
      a fifth of its full range, which separates the two regimes cleanly, and
      suggests a log or rank transform. This regime is common in assay and
      biomarker data.

- **The ranks are scores, not percentiles, and only one of the three spans
  [0, 1].** `archetypal_rank` has a structural floor near 0.4 at `nmf_rank=3`,
  because its membership term is the maximum of a row-normalised vector over
  `nmf_rank` entries and so cannot fall below `1/nmf_rank`, while its corner
  term is normalised by `sqrt(d)` when the largest attainable distance is
  `0.5*sqrt(d)` and so occupies only [0.5, 1]. `prototypical_rank` has a similar
  floor. Only `stereotypical_rank` uses the full interval. The README now gives
  the attainable range of each and says to compare ranks against each other
  rather than against zero. The arithmetic is unchanged.

- **`auto_n_prototypes` recognised exactly one spelling.** Only the literal
  string `'kneedle'` did anything. A near miss such as `'knee'`, or any other
  value, was accepted and then ignored, so a caller who asked for automatic
  selection quietly got none. It now accepts `None` or `'kneedle'` and raises
  on anything else.

- **`register_ideal()` stored a vector that nothing reads.** No scoring path
  touches `ideals_`, so registering an ideal never changed a rank, an
  explanation or any other output, despite the name. It is a leftover from the
  pre-v0.4 stereotype mechanism. The method still validates and stores, so
  existing code keeps working, but it now emits a `DeprecationWarning` saying
  plainly that it has no effect and pointing at `stereotype_column` with
  `stereotype_target`, which is the supported mechanism.

- **`scale`, `distance_metric` and `similarity_metric` did nothing at all.**
  All three were declared as constructor parameters, carried in `to_config()`
  and `get_params()`, and **never read anywhere in the module**. Any value was
  accepted, including nonsense such as `scale='wibble'`, and the pipeline went
  on using MinMax scaling, Euclidean distance and cosine similarity regardless.
  A caller asking for standardised features, or a cosine distance, silently got
  neither and had no way to tell.

  Rather than change the numerics in a bug-fix release, the implemented value is
  now the only one accepted: `scale='minmax'`, `distance_metric='euclidean'`,
  `similarity_metric='cosine'`. Anything else raises `ConfigError` explaining
  what the code actually does. Results for anyone using the defaults are
  unchanged, because the defaults were always what ran.

- **A non-finite formative value was reported as a confident 0.5.** The rank
  normaliser compared `max - min > 1e-12` to decide whether the values varied.
  Every comparison against NaN is False, so a NaN landed in the equal-values
  branch and *every sample* was reported as 0.5, a perfectly ordinary mid-rank
  standing in for "could not be computed". Non-finite values now stay NaN, the
  finite rows are still ranked among themselves, and a warning names the
  significance type and counts the affected rows.

- **`auto_n_prototypes='kneedle'` silently cut the prototype set to one.**
  Facility-location gains fall away steeply, so the first prototype carries most
  of the coverage and the knee sits at index 1. On every dataset tried, asking
  for 15 prototypes and enabling Kneedle returned exactly 1, which changes every
  prototypical result and every figure drawn from it. The truncation now warns
  when it keeps fewer than two prototypes, or under a quarter of those selected,
  and names both counts. The Kneedle maths is unchanged.

- **`knee_` never reported the value that shaped the result.** It was assigned
  the Kneedle index and then overwritten unconditionally by a second,
  unrelated second-difference heuristic. When Kneedle runs, `knee_` now reports
  what it chose, which is also `len(prototype_indices_)`. Otherwise it keeps the
  diagnostic it has always carried.

- **State from one fit survived into the next on the same estimator.**
  Several attributes were written only on the path that produced them and read
  back with `hasattr`, so nothing cleared them:

    * refitting on clean data still reported the previous fit's
      `dropped_columns_` and `missingness_`;
    * refitting with `stereotype_column=None` kept the previous stereotype
      source;
    * fitting text after tabular left `_df_original_fit`, `feature_columns_`,
      `keep_mask_` and `scaler_` pointing at the tabular frame, so `heatmap()`
      and `profile_plot()` would have described the *wrong data* instead of
      refusing, and `transform_text()` would not have noticed the missing
      vectorizer.

  Every public fit entry point now clears this state first.

- **`fast_mode` was only honoured on the first fit.** The presets were applied
  once per object behind an `_fast_mode_applied` flag, so changing `fast_mode`
  and refitting left the previous mode's `archetypal_method`,
  `shapley_n_permutations`, `shapley_top_n` and `shapley_compute_formative` in
  place. They are now reconsidered on every fit. A value the presets filled in
  is released only while it still holds exactly what they put there, so an
  explicit choice made in between is preserved. The one case that cannot be
  distinguished, assigning the same value the preset already chose, is
  documented in the tests.

- **`to_config()` silently dropped `feature_weights`.** A configuration saved
  and reloaded therefore produced a *different fit*. On a five-feature frame the
  round trip moved `archetypal_rank` by 0.33 and `prototypical_rank` by 0.50, on
  a scale where both live in [0, 1]. That is a different answer, not a rounding
  difference. `to_config()` now carries every init parameter, and a test fails if
  a future parameter is added without being included.

- **`transform()` clipped out-of-range data onto the training boundary in
  silence.** The fitted scaler is built with `clip=True`, so anything beyond the
  range seen during `fit` is pinned to 0 or 1. Two samples ten times apart from
  each other, both far beyond the training maximum, therefore received identical
  ranks, with nothing to indicate it. For a library whose purpose is identifying
  extreme instances that is precisely the wrong silent failure.

  The clipping itself is kept, because unclipped values break the [0, 1]
  geometry the archetypal scores assume. What is new is that it reports itself:
  how many values across how many rows were clipped, which column is worst, and
  how many training ranges beyond the limit it extends. Missing values, which
  are imputed rather than clipped, are not counted.

- **`max_memory_mb` was only validated on the chunked path.** A nonsensical
  budget such as `0` was accepted without complaint on any dataset small enough
  to skip chunking, then raised much later on a larger one. It is now checked at
  fit time.

- **An edge naming a node outside `range(n_nodes)` was accepted in silence.**
  NetworkX simply added the extra node, so every topology feature was then
  computed on a larger graph than the node features described. Pagerank over the
  requested nodes summed to less than one (0.82 in a three-node example with one
  stray index), and betweenness and closeness were measured against phantom
  nodes. The returned columns looked perfectly ordinary. Out-of-range and
  negative indices now raise, naming the offending values.

  A misshapen edge array now raises as well, and the one genuinely ambiguous
  case, a (2, 2) array that could be two edges as rows or two nodes as columns,
  warns which reading it used instead of choosing silently.

- **`UnboundLocalError` in stereotypical Shapley explanations.** The
  stereotypical value function was defined inside the core-samples branch of
  `_fit_shapley_explanations` but was also called from the secondary-samples
  branch. When `subsample_indices` was supplied, `core_samples` came from
  `_union_core_samples` and need not intersect the requested samples, so it
  could be empty while `secondary_samples` was not. The crash fired for
  `shapley_mode=True` with a `stereotype_column` and `shapley_top_n` less than
  the number of rows, and never at `shapley_top_n == len(df)`, which is why it
  went unnoticed.

- **A non-numeric `stereotype_column` failed deep inside pandas.** A reasonable
  request such as `stereotype_column='Menopause'` on a Yes/No column raised
  `ValueError: could not convert string to float: 'No'` several frames below the
  caller, with no mention of the column at fault. The column is now validated
  where it is selected, at fit time, whatever `shapley_mode` is set to. Ordered
  categoricals are encoded by category order and booleans become 0.0/1.0;
  anything else raises `ConfigError` naming the column and the offending values.

- **A `stereotype_column` on text data was ignored when no `text_metadata` was
  supplied.** `_get_stereotype_source_text` held the right error but nothing
  called it, so the fit silently fell back to extremeness. This is the same
  class of silent substitution as the archetypal backend defect. Found while
  writing the v0.8.0 test suite.

- **Verbose text and graph fits crashed on a `stereotype_column`.** The verbose
  reporting block guarded `_df_original_fit` with `hasattr` alone, but the
  attribute exists and is `None` for text and graph fits, so the branch
  dereferenced `None`. Found while writing the v0.8.0 test suite.

### Added


- **`formative_method='exact'`**, a closed-form computation of the archetypal
  formative Shapley values with no sampling at all.

  The value function is a mean of "minimum" games, one per archetype, and
  Shapley is linear in the value function, so the whole thing has a closed form.
  For one archetype, sorting the samples by distance gives, for the sample at
  ascending rank r,

      phi_r = -d_r / n  +  sum over k > r of (d_k - d_r) / ((k + 1) * k)

  Both sums are suffix sums, so it costs O(n log n) per archetype: **0.006
  seconds at n = 5000**, against minutes of sampling that still would not
  converge. Verified against exhaustive coalition enumeration to machine
  precision, and the values satisfy efficiency exactly.

  **The prototypical game is exact too, and it was the one that needed it
  most.** It is a mean of maxima, so it does not collapse the way the other two
  do, but it decomposes. The maximum is attained at the first neighbour present
  in the coalition, so

      max(0, max_j s_ij) = sum over r of s_i,(r) * 1[l_(r) in S, l_(1..r-1) not in S]

  which rewrites the value function as a weighted sum of games of the form
  `1[A subset of S, B disjoint from S] / |S|` with `|A| = 2`. Shapley is linear
  in the value function, and each of those little games has a closed form
  depending only on `|B|` and the player's role. Accumulating the "neither"
  term as a scalar and the "in B" term as a suffix sum gives **O(n^2)** after
  the sorts, against O(2^n) for enumeration. Exposed as
  `exact_formative_prototypical`.

  Measured cost: 0.6 s at n = 500, 7 s at n = 2000, 21 s at n = 5000, 94 s at
  n = 10,000, 372 s at n = 19,000. Memory is bounded by computing the coefficient tables in blocks
  and by never forming the n by n similarity matrix, which would be 16 GB at
  n = 45,000.

  Why it mattered: on the Wine dataset at 100 permutations, five seeds gave
  **five different top formative prototypes**, with pairwise rank correlations
  of 0.03 to 0.17 and a top-ten overlap of 0 or 1 out of 10 in nine of the ten
  seed pairs. Verified against exhaustive coalition enumeration, using the
  shipped value function itself, on random, positive-orthant, antipodal,
  duplicated, zero-row and all-identical inputs.

  **This is the default from v0.8.0.** Sampling is still reachable with
  `formative_method='monte_carlo'`, which reproduces a pre-0.8.0 result:

      DataTypical(shapley_mode=True, shapley_compute_formative=True,
                  formative_method='monte_carlo')

  **This changes numbers.** A formative ranking produced by v0.7.7 or earlier
  will not match one produced by v0.8.0, and the v0.8.0 one is correct: the
  sampled ordering correlated 0.40 to 0.55 with the exact answer on real
  cohorts, and 0.19 with itself across seeds.

  It also covers the **stereotypical** formative values, whose value function is
  a plain mean over the coalition and so reduces to

      phi_i = (1/n) * [ c_i + (c_i - mbar_i) * (H_n - 1) ]

  with `c_i` the deviation from the median, `mbar_i` the mean of `c` over the
  other samples and `H_n` the nth harmonic number. A numeric `stereotype_target`
  adds a constant term. Verified against enumeration for all three target modes.

  The **prototypical** game does not reduce: its value is a mean of per-member
  maxima over the other members, so a coalition does not collapse to an order
  statistic. It stays Monte Carlo, and the split-half diagnostic still applies
  to it.

  New public functions `exact_formative_archetypal(X, archetypes)` and
  `exact_formative_stereotypical(target_values, target, median)`.

- **`archetypal_method='auto'`**, the permissive cascade PCHA to ConvexHull to
  NMF that `'aa'` used to perform silently. Each downgrade now emits a
  `RuntimeWarning`.

- **`archetypal_backend_`**, a public attribute recording which backend actually
  produced the archetypes: `'pcha'`, `'convexhull'` or `'nmf'`. It is also
  written into `settings_` alongside the requested `archetypal_method`, so an
  archived fit can be audited after the fact.

- **`py_pcha` is now a declared dependency** in `requirements.txt` and
  `install_requires`. It was required for the headline method but was declared
  nowhere, which is how the silent fallback went unnoticed.

- **A pytest suite**, discoverable through `pytest.ini`, with `.coveragerc`
  configured to measure broadly and filter at report time. 748 tests, 99.5%
  statement coverage (`datatypical.py` 99.4%, `datatypical_viz.py` 100%),
  measured with `NUMBA_DISABLE_JIT=1` so that the compiled kernels are visible
  to the line tracer. With compilation active the same suite reports 94.4%,
  because those kernels then execute as machine code and are not traced at all.

- **Invariant tests** asserting the properties the ranks claim: exact row-order
  invariance, exact scale and offset invariance, agreement between
  `fit_transform` and `fit` then `transform`, agreement between a selective fit
  and the matching column of a full fit, and Shapley efficiency, symmetry and
  dummy. The v0.7.7 streaming rewrite was also checked against the generic
  re-evaluation path it replaced, and agrees to 2e-16, as claimed.

- **Correctness tests for the parts that had none**: graph topology values
  against hand-computed graphs, the facility-location selector against brute
  force on a small problem (greedy reaches 98% of the optimum, well above the
  0.632 submodularity bound), chunked distances against unchunked across six
  memory budgets, and each visualisation against the data it claims to show.

### Changed

- `to_config()` reports `version` as `"0.8.0"`.
- The verbose archetypal method label now names the policy in force rather than
  always claiming PCHA+ConvexHull.

### Documentation

- **The stereotypical formative ranking carries no information beyond the
  stereotype column.** The closed form above is *affine* in `c_i`, so the
  ranking is a monotone transform of the column itself, with a measured Pearson
  correlation of 1.0000000000. Both axes of the stereotypical dual-perspective
  plot are therefore driven by the same single column, which makes that panel a
  curve rather than a scatter. This is a property of the value function, not of
  the new implementation: the Monte Carlo estimate was always a noisy version of
  the same quantity. It is the same circularity already noted for
  `stereotypical_rank` itself, now shown to apply to the formative axis too.
  The archetypal formative axis is genuinely distinct by comparison, correlating
  0.83 with a simple distance-to-nearest-archetype proxy rather than 1.0.

- Recorded that `stereotypical_rank` is a deterministic, monotone function of
  `stereotype_column` alone (Methods 2.3.1, eq. 26):
  `s_stereo_i = 1 - |y_i - tau| / max_j |y_j - tau|`. Placing an outcome-derived
  quantity in that column and then evaluating the resulting rank against that
  outcome on held-out rows is circular, and returns a perfect correlation that
  means nothing. This is by design: stereotypical significance depends on an
  external specification of what counts as interesting, unlike archetypal and
  prototypical significance, which derive from the intrinsic geometry of the
  data.

### Not a defect

- `stereotypical_rank` was investigated and found to match the published
  equation exactly. Measured on a real cohort with `stereotype_column='Age'` and
  `stereotype_target=55`, the Spearman correlation with `-|Age - 55|` is
  +1.000000 and no value of Age maps to more than one rank. The behaviour is
  pinned by tests so it is not "fixed" by mistake later.

### Known gaps

- Ten statements in `datatypical.py` are not covered by the test suite because
  they are unreachable: defensive guards on a zero-width feature subset inside
  the explanation closures (the coalition walk never produces one), a
  `stereotype_target` string other than `'min'`/`'max'` after validation has
  already rejected those, a dtype branch in `_fit_components` whose input is
  always cast to float64 first, and a knee branch whose condition contradicts
  its enclosing test. They are left in place and left uncovered rather than
  marked as excluded.

---

## [0.7.7] - 2026-06-03

### Changed

- Streaming formative-Shapley computation. Each Monte Carlo permutation now
  updates the value functions incrementally along the growing coalition instead
  of recomputing them from scratch at every prefix. Per-fit complexity drops
  from O(M*n^2) to O(M*n) for archetypal and stereotypical significance, and
  from O(M*n^3) to O(M*n^2) for prototypical. Output is numerically identical to
  v0.7.6; only runtime changes. The formative step at n = 10,000 completes in
  seconds rather than hours.
- Console output is ASCII-only, so verbose logs and the test suites run cleanly
  under any terminal encoding, including Windows cp1252.

---

## [0.7.6]

### Added

- `selected_significance` parameter, to compute one significance type at a time.

### Fixed

- Prototype feature storage, so `transform()` on new data uses the correct
  vectors.
- Full Shapley analysis (formative and explanations) now runs on text data
  paths.
- Iterator exhaustion in all text fit and transform methods.
- Local/global index mismatch in stereotypical Shapley explanations.
- Clearer error messages when a significance type was not fitted.

---

## [0.7]

### Added

- `shapley_mode` parameter. When `True`, computes explanations and formative
  instances.
- Dual rankings: `*_rank` (actual) alongside `*_shapley_rank` (formative).
- Value functions for convex hull, coverage and extremeness.
- Parallel Shapley computation.
- `fast_mode` presets for exploration versus publication.

---

## [0.6]

### Added

- Local explanations via `get_shapley_explanations()`.
- Global explanations identifying formative instances.

---

## [0.5]

### Added

- Tabular, text and graph support through a unified API.
- Label column preservation.
- Graph topology features.

---

## [0.4]

### Added

- User-configurable stereotypes.
