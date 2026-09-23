# DataTypical

**Scientific Data Significance Rankings with Shapley Explanations**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/datatypical.svg)](https://pypi.org/project/datatypical/)
[![Tests](https://img.shields.io/badge/tests-692%20passing-21918c.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-99.5%25-22a884.svg)](tests/)

DataTypical analyzes datasets through three complementary lenses: archetypal (extreme), prototypical (representative), and stereotypical (target-like), with Shapley value explanations revealing why instances matter and which ones create your dataset's structure.

---

## Key Features

- **Three Significance Types**: Archetypal, prototypical, stereotypical (all computed simultaneously, or selectively)
- **Shapley Explanations**: Feature-level attributions for why samples are significant
- **Formative Discovery**: Distinguish samples that ARE significant from those that CREATE structure
- **Publication Visualizations**: Dual-perspective scatter plots, heatmaps, and profile plots
- **Multi-Modal Support**: Tabular data, text, and graph networks through unified API
- **Performance Optimized**: Fast exploration mode and efficient Shapley computation

---

## Quick Start

### Installation

```bash
pip install datatypical
```

`py_pcha` is installed with the package and is required for archetypal analysis
(`archetypal_method='aa'`). If it is missing, `'aa'` raises rather than quietly
substituting another method. See [Archetypal backends](#archetypal-backends).

### Basic Usage

```python
from datatypical import DataTypical
from datatypical_viz import significance_plot, heatmap, profile_plot
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Analyze with explanations
dt = DataTypical(shapley_mode=True)
results = dt.fit_transform(data)

# Three significance perspectives (bounded in [0, 1]; see Reading the ranks)
print(results[['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']])

# Visualize: which samples are critical vs replaceable?
significance_plot(results, significance='archetypal')

# Understand: which features drive significance?
heatmap(dt, results, significance='archetypal', top_n=20)

# Explain: why is this sample significant?
top_idx = results['archetypal_rank'].idxmax()
profile_plot(dt, top_idx, significance='archetypal')
```

---

## What DataTypical Does

### Three Complementary Lenses

| Lens | Finds | Use Cases |
|------|-------|-----------|
| **Archetypal** | Extreme, boundary samples | Edge case discovery, outlier detection, range understanding |
| **Prototypical** | Representative, central samples | Dataset summarization, cluster centers, typical examples |
| **Stereotypical** | Target-similar samples | Optimization, goal-oriented selection, phenotype matching |

**The Power**: All three computed simultaneously—different perspectives reveal different insights.

### Dual Perspective (with Shapley)

When `shapley_mode=True`, DataTypical reveals two views:

**Actual Significance** (`*_rank`): Samples that ARE significant  
**Formative Significance** (`*_shapley_rank`): Samples that CREATE the structure

**Four Quadrants**:
```
     Formative High
          │
  Gap     │  Critical
  Fillers │  (irreplaceable)
──────────┼──────────────── Actual High
Redundant │ Replaceable
          │  (keep one)
     Formative Low
```

This distinction—between what IS significant vs what CREATES structure—is a genuinely novel contribution.

---

## Example: Drug Discovery

```python
# Analyze compound library
dt = DataTypical(
    shapley_mode=True,
    stereotype_column='activity',  # Target property
    fast_mode=False
)
results = dt.fit_transform(compounds)

# Find critical compounds (high actual + high formative)
critical = results[
    (results['stereotypical_rank'] > 0.8) &
    (results['stereotypical_shapley_rank'] > 0.8)
]
print(f"Found {len(critical)} critical compounds")

# Find redundant compounds (high actual + low formative)
redundant = results[
    (results['stereotypical_rank'] > 0.8) &
    (results['stereotypical_shapley_rank'] < 0.3)
]
print(f"Found {len(redundant)} replaceable compounds")

# Understand alternative mechanisms
for idx in critical.index:
    profile_plot(dt, idx, significance='stereotypical')
    # Each shows different feature pattern → different mechanism
```

**Discovery**: Multiple structural pathways to high activity!

---

## Performance

### Formative-Shapley speed (v0.7.7)

In publication mode (`shapley_mode=True`, `fast_mode=False`) the cost of the
formative-instance computation now scales linearly (archetypal, stereotypical)
or quadratically (prototypical) in the number of samples, instead of
quadratically/cubically. Rankings are numerically identical to v0.7.6 — only
runtime changes.

| Samples | Formative step, v0.7.6 | Formative step, v0.7.7 |
|---------|------------------------|------------------------|
| 1,000   | ~40 seconds            | < 0.1 seconds          |
| 2,000   | ~6.5 minutes           | ~0.3 seconds           |
| 10,000  | ~13 hours (est.)       | ~8 seconds (est.)      |

*Measured single-threaded, M = 30 permutations, d = 8 features, summed over the
archetypal, prototypical, and stereotypical value functions. The 10,000-sample
row is extrapolated from the measured scaling.*

The remaining publication-mode cost is the per-sample feature **explanations**
(a separate Shapley computation). Bound this with `shapley_top_n` to explain only
the most significant samples; it is the main lever on full-pipeline runtime once
the formative step is no longer the bottleneck.

### Optimization Strategy

**Phase 1**: Fast exploration (`fast_mode=True`, no Shapley) to identify
interesting samples.

**Phase 2**: Detailed analysis (`shapley_mode=True`) to generate formative
rankings, explanations, and publication figures. Set `shapley_top_n` to cap how
many samples receive feature-level explanations.

---

## Key Parameters

```python
DataTypical(
    # Enable explanations and formative analysis
    shapley_mode=False,           # True for explanations

    # Speed vs accuracy
    fast_mode=True,               # False for publication quality

    # Significance types
    n_archetypes=8,               # Number of extreme corners
    n_prototypes=8,               # Number of representatives
    stereotype_column=None,       # Target column for stereotypical
    stereotype_target='max',      # 'max', 'min', or numeric value

    # Archetypal backend
    archetypal_method=None,       # 'aa' (PCHA, required), 'auto' (cascade),
                                  # 'nmf', or None to follow fast_mode

    # Selective computation
    selected_significance=None,   # 'archetypal', 'prototypical', 'stereotypical', or None (all)

    # Shapley optimization
    shapley_top_n=500,            # Limit explanations to top N
    shapley_n_permutations=100,   # Number of permutations (30 in fast_mode)

    # Reproducibility
    random_state=None,            # Set for reproducible results

    # Memory management
    max_memory_mb=8000            # Memory limit for operations
)
```

### Parameters with a single implemented value

`scale`, `distance_metric` and `similarity_metric` exist on the constructor but
only one value of each is implemented:

| Parameter | Accepted value | What the code does |
| --- | --- | --- |
| `scale` | `'minmax'` | MinMax scaling to [0, 1] |
| `distance_metric` | `'euclidean'` | Euclidean distance |
| `similarity_metric` | `'cosine'` | cosine similarity |

`auto_n_prototypes` is the same shape of thing: `None` or `'kneedle'`, nothing
else. Anything outside these sets raises `ConfigError`.

`register_ideal()` is a leftover from the pre-v0.4 stereotype mechanism. It
stores a vector in `ideals_` that no scoring path reads, so it has no effect on
any output. Use `stereotype_column` with `stereotype_target` instead.

> **Changed in v0.8.0.** Through v0.7.7 these three were never read anywhere in
> the module. Any value was accepted, including nonsense, and the pipeline used
> MinMax, Euclidean and cosine regardless. A fit that asked for standardised
> features silently got MinMax instead. Results for anyone using the defaults
> are unchanged, because the defaults were always what ran.

### Formative instances and convergence

The formative ranking is a Monte Carlo Shapley estimate, and it converges far
more slowly than the default suggests. Measured on synthetic data, two fits
differing only in `random_state`:

| Permutations | Spearman rho between seeds | Top-10 overlap |
| --- | --- | --- |
| 100 (the default) | **0.02** | **1 of 10** (chance is 2) |
| 1000 | 0.32 | 4 of 10 |

The values are not wrong: they sum to the correct total, which is what the
`additivity_error` in `shapley_info_` has always reported. What is unstable is
the *ordering*, and the ordering is what you read as the formative instances.

**The archetypal formative values have a closed form.** The value function is
a mean of minimum games, one per archetype, and Shapley is linear in the value
function, so no sampling is needed at all. **Since v0.8.0 this is the default**,
because the sampled ordering did not converge:

```python
dt = DataTypical(shapley_mode=True, shapley_compute_formative=True)
results = dt.fit_transform(data)          # formative_method='exact'
```

It costs O(n log n) per archetype, about 0.006 seconds at n = 5000, and is
verified against exhaustive coalition enumeration. The result is identical
whatever `random_state` you use.

The stereotypical formative values reduce the same way. The prototypical game
does not reduce, but it decomposes: the maximum is attained at the first
neighbour present in the coalition, which turns the value function into a
weighted sum of indicator games, each with a closed form. That costs O(n^2)
rather than O(n log n), and it is exact.

**All three formative games are therefore exact by default, and the formative
ranking no longer depends on `random_state` at all.** To reproduce a result
computed before v0.8.0, pass `formative_method='monte_carlo'`.

The prototypical game was the least convergent of the three under sampling. On
Wine at the documented 100 permutations, five seeds produced five different top
formative prototypes, with pairwise rank correlations of 0.03 to 0.17 and a
top-ten overlap of 0 or 1 out of 10 in nine of the ten seed pairs.

Both closed forms are available directly, if you want the values without a fit:

```python
from datatypical import (exact_formative_archetypal,
                         exact_formative_prototypical,
                         exact_formative_stereotypical)

exact_formative_archetypal(scaled_X, dt.H_)
exact_formative_prototypical(scaled_X)
exact_formative_stereotypical(target_values, target='max')
```

The same setting also makes the **stereotypical** formative values exact: that
value function is a plain mean over the coalition, which has its own closed
form. The **prototypical** game does not reduce, so it stays Monte Carlo and the
advice below still applies to it.

> **Worth knowing about the stereotypical formative axis.** Its closed form is
> affine in the deviation from the median, so the ranking is a monotone
> transform of `stereotype_column` and carries no information beyond it. Both
> axes of the stereotypical dual-perspective plot are driven by that one column,
> which makes the panel a curve rather than a scatter. This is a property of the
> published value function, not of the exact implementation. The archetypal
> formative axis is genuinely distinct by comparison.

**Check `split_half_rho` before reporting a Monte Carlo formative result:**

```python
dt = DataTypical(shapley_mode=True, shapley_compute_formative=True,
                 shapley_n_permutations=100)
dt.fit_transform(data)

dt.shapley_info_['archetypal_formative']['split_half_rho']
```

It splits the permutations into two independent halves and correlates the two
rankings. Below about 0.7 the ordering is mostly sampling noise, and a
`RuntimeWarning` says so. Raise `shapley_n_permutations` until the number stops
rising. The actual ranks, `archetypal_rank` and the rest, are fully
deterministic and are not affected by any of this.

### Reading the ranks

All three ranks are bounded in [0, 1], but they are **scores, not percentiles**,
and only `stereotypical_rank` uses the full range. The other two have structural
floors, so a value near the bottom of what you observe is the minimum the
measure can produce, not an especially low score:

| Rank | Definition | Attainable range |
| --- | --- | --- |
| `archetypal_rank` | `0.7 * membership concentration + 0.3 * corner proximity` | roughly [0.4, 1] at `nmf_rank=3`; the floor rises with smaller `nmf_rank` |
| `prototypical_rank` | `0.5 * (1 - normalised distance to nearest prototype) + 0.5 * cosine to it` | roughly [0.4, 1] |
| `stereotypical_rank` | `1 - |y - tau| / max|y - tau|` | exactly [0, 1] |

The membership term of `archetypal_rank` is the maximum of a row-normalised
weight vector over `nmf_rank` archetypes, so it cannot fall below `1/nmf_rank`.
The corner term is normalised by `sqrt(d)` where the largest attainable distance
is `0.5*sqrt(d)`, so it occupies [0.5, 1].

**Compare ranks against each other, not against 0.** If you need values that span
the full interval, rank-transform the column yourself.

### When a few values own a feature's range

MinMax scaling maps the extremes of each feature to 0 and 1. When one value sits
far above the rest, everything else is compressed against the low end, and that
end is itself a corner of the unit cube, so the archetypal corner term loses
resolution exactly where most of the data is.

The practical effect, measured rather than assumed:

- On synthetic data with a **single dominant feature**, an outlier at 3 to 5
  standard deviations ranks above the cloud median on 6 of 6 seeds. One at 30
  standard deviations ranks **below** it on 6 of 6. The ordering inverts.
- On a **real assay cohort** with 43 features, no inversion appears: patients
  with the highest marker values still rank above those at the median. But the
  ranking becomes **unstable**. Log-transforming the compressed columns changed
  **12 of the 20 most archetypal instances**, with a Spearman correlation of
  0.82 between the two orderings.

So with many features the risk is not a sign flip but a ranking that depends on
a preprocessing choice you may not have thought of as one. Since v0.8.0 a
`RuntimeWarning` fires when the middle 98% of any feature spans under a fifth of
its full range. **Fit both ways and compare** before reporting which instances
are archetypal.

> **One further case.** A row that is the minimum in *every* retained feature
> sits at the origin of the scaled space and has no defined archetype
> membership: both backends return a zero weight vector for it. Its
> `archetypal_rank` then falls to the bottom of the column even though it is an
> extreme point. This warns too. Treat those scores as undefined rather than
> low.

### Archetypal backends

`archetypal_method` controls how the archetypes are computed, and how strictly
that choice is enforced.

| Value | Behaviour |
| --- | --- |
| `'aa'` | Principal Convex Hull Analysis (PCHA), the method named in the manuscript. If `py_pcha` is unavailable or PCHA fails, this raises `ConfigError`. It never substitutes another method. |
| `'auto'` | Permissive cascade PCHA to ConvexHull to NMF, with a `RuntimeWarning` at each downgrade. |
| `'nmf'` | NMF approximation, computed directly and without warnings. |
| `None` | Follows `fast_mode`: `'nmf'` when `fast_mode=True`, `'aa'` when `fast_mode=False`. |

Whichever route ran is recorded on the fitted estimator and in `settings_`:

```python
dt = DataTypical(archetypal_method='aa')
dt.fit(data)

dt.archetypal_backend_          # 'pcha', 'convexhull' or 'nmf'
dt.settings_['archetypal_backend']
```

> **Changed in v0.8.0.** Through v0.7.7, `'aa'` fell silently through to
> ConvexHull and then to NMF when `py_pcha` was missing, with the only notice
> behind `verbose=False`. A fit reported as archetypal analysis could return NMF
> output with nothing to distinguish it. Any `'aa'` result produced by v0.7.7 or
> earlier in an environment without `py_pcha` is an NMF approximation and should
> be re-run. A saved fit can be checked without re-running it: `nmf_model_ is not
> None` means NMF ran; `nmf_model_ is None` with a numeric
> `reconstruction_error_` means PCHA ran; `nmf_model_ is None` with
> `reconstruction_error_ is None` means ConvexHull ran.

### `stereotype_column`

Stereotypical significance is the one lens that depends on an external
specification of what counts as interesting, rather than on the intrinsic
geometry of the data. The rank is a monotone rescaling of the distance from the
target on that one column and nothing else:

```
s_stereo_i = 1 - |y_i - tau| / max_j |y_j - tau|
```

That has a consequence worth stating plainly: **putting an outcome-derived
quantity in `stereotype_column` and then evaluating the resulting rank against
that outcome on held-out rows is circular.** The correlation is guaranteed by
construction and means nothing. Use a criterion you specify in advance, not one
derived from the label you intend to predict.

The column must be numeric, boolean, or an ordered categorical. Ordered
categoricals are encoded by category order; anything else raises `ConfigError`
naming the column.

### `selected_significance`

When you only need one significance type, set `selected_significance` to skip the others entirely—saving substantial compute time:

```python
# Only compute archetypal (skip prototypical and stereotypical)
dt = DataTypical(selected_significance='archetypal', shapley_mode=True)
results = dt.fit_transform(data)
# → archetypal_rank computed; prototypical_rank and stereotypical_rank are NaN
```

---

## Visualization

### Three Core Plots

```python
from datatypical_viz import significance_plot, heatmap, profile_plot

# 1. Overview: Actual vs Formative scatter
significance_plot(results, significance='archetypal')

# 2. Feature patterns: Which features matter?
heatmap(dt, results,
        significance='archetypal',
        order='actual',  # or 'formative'
        top_n=20)

# 3. Individual explanation: Why is this sample significant?
profile_plot(dt, sample_idx,
             significance='archetypal',
             order='local')  # or 'global'
```

See [docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md) for detailed interpretation.

---

## Multi-Modal Support

### Tabular Data (Default)
```python
df = pd.DataFrame(...)
dt = DataTypical()
results = dt.fit_transform(df)
```

### Text Data (Auto-Detected)
```python
texts = ["document 1", "document 2", ...]
dt = DataTypical()
results = dt.fit_transform(texts)
```

### Graph Networks (Protein Interactions, Molecules)
```python
node_features = pd.DataFrame(...)
edges = [(0, 1), (1, 2), ...]
dt = DataTypical()
results = dt.fit_transform(node_features, edges=edges)
```

---

## Use Cases

### Scientific Discovery
- **Alternative mechanisms**: Formative instances reveal different pathways
- **Boundary definition**: Which samples define system limits
- **Quality control**: Distinguish novel variation from known patterns
- **Coverage analysis**: Identify sampling gaps

### Dataset Curation
- **Size reduction**: Remove redundant samples while preserving diversity
- **Representative selection**: Choose samples spanning full space
- **Redundancy detection**: Find clusters of similar samples
- **Gap identification**: Locate undersampled regions

### Model Understanding
- **Feature importance**: Global and local significance patterns
- **Individual explanations**: Why specific samples matter
- **Pattern recognition**: Discover multiple pathways to outcomes
- **Interpretability**: Explanations in original feature space

---

## Documentation

**New Users**:
- [docs/START_HERE.md](docs/START_HERE.md) — Friendly introduction and first steps
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) — Daily reference for parameters and workflows
- [docs/EXAMPLES.md](docs/EXAMPLES.md) — Complete worked examples across domains

**Visualization**:
- [docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md) — Comprehensive guide to plots and interpretation

**Advanced**:
- [docs/INTERPRETATION_GUIDE.md](docs/INTERPRETATION_GUIDE.md) — Interpreting complex patterns
- [docs/COMPUTATION_GUIDE.md](docs/COMPUTATION_GUIDE.md) — Implementation details and algorithms

**Benchmarks**:
- [benchmarks/](benchmarks/) — Structural-significance benchmarks (model-free comparison, bounded metrics, synthetic ground truth) and the v0.7.7 scaling study. See its `README.md`, and run `Benchmarks.ipynb` for the same experiments with figures shown inline.
- Superseded benchmark drafts are retained under [DRAFTS/](DRAFTS/).

---

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- SciPy ≥ 1.7
- scikit-learn ≥ 1.0
- Matplotlib ≥ 3.3
- Seaborn ≥ 0.11
- Numba ≥ 0.55 (for performance)
- py_pcha ≥ 0.1.3 (required for `archetypal_method='aa'`)

---

## Citation

If you use DataTypical in your research, please cite:

```bibtex
@software{datatypical2026,
  author = {Barnard, Amanda S.},
  title = {DataTypical: Scientific Data Significance Rankings with Shapley Explanations},
  year = {2026},
  url = {https://github.com/amaxiom/DataTypical},
  version = {0.8.1}
}
```

---

## What Makes DataTypical Different

### From Traditional Methods

**Outlier Detection**: Only finds extremes → DataTypical finds extremes AND explains why

**Clustering**: Groups samples, picks centroids → DataTypical finds representatives maximizing coverage

**Feature Selection**: Ranks features → DataTypical explains which features matter for which samples

**PCA/t-SNE**: Projects to low dimensions → DataTypical maintains interpretability in original space

### The Novel Contribution

**Formative instances** are genuinely new. The distinction between samples that ARE significant vs samples that CREATE structure emerges from the Shapley mechanism and enables:

- Redundancy detection even among significant samples
- Finding structurally important but non-extreme samples
- Understanding irreplaceable vs interchangeable samples
- Quality control based on structural contribution

This dual perspective transforms instance significance from pure ranking into causal understanding.

---

## Development Status

**Current Version**: 0.8.1

**v0.8.1** fixes archetypal analysis on NumPy 2. `py_pcha` calls `np.mat`, removed in NumPy 2.0, so a fresh install of v0.8.0 could not run `archetypal_method='aa'` at all. Through v0.7.7 the same failure fell through to ConvexHull **silently**, so any `'aa'` result from v0.7.7 or earlier on NumPy 2 is a ConvexHull approximation. See [CHANGELOG.md](CHANGELOG.md).

**Recent Updates (v0.8.0)**, nineteen fixes and four additions, see [CHANGELOG.md](CHANGELOG.md):
- **Fixed a results-invalidating defect**: `archetypal_method='aa'` silently
  returned NMF output when `py_pcha` was missing. It now raises. New
  `archetypal_method='auto'` keeps the old permissive cascade with warnings, and
  the new `archetypal_backend_` attribute records what actually ran. `py_pcha`
  is now a declared dependency.
- **Fixed a second results-affecting defect**: the working `dtype` (float32 by
  default) was applied to the raw input before scaling, so large values
  overflowed to infinity, small values underflowed to zero, and a feature
  carrying its signal beyond the 7th significant digit lost it and was dropped
  as constant. Scaling now happens in float64 and only the scaled result is
  cast, so `dtype` is a memory setting only and no longer changes results. The
  ranks are now exactly invariant to rescaling and offsetting a feature.
- **Added `formative_method`, defaulting to `'exact'`**: exact algorithms for
  all three formative games, with no sampling and no seed dependence. O(n log n)
  for the archetypal and stereotypical games, O(n^2) for the prototypical one. This **changes the formative ranking** any
  earlier version produced, which is the point: the sampled ordering did not
  converge. Pass `formative_method='monte_carlo'` to reproduce an older result.
- **The formative rankings do not converge at the default 100 permutations.**
  Two fits differing only in `random_state` gave rankings correlating 0.02, with
  a top-ten overlap of 1 of 10 against a chance level of 2. Every estimate now
  reports a `split_half_rho` and warns when the ordering is sampling noise. See
  [Formative instances and convergence](#formative-instances-and-convergence).
- **Documented two properties of the archetypal measure that were never stated,
  and made both announce themselves.** A row that is the minimum in every
  feature ranks last despite being extreme, and the measure inverts once one row
  dominates a feature's range. Neither arithmetic has changed. See
  [Reading the ranks](#reading-the-ranks).
- **`scale`, `distance_metric` and `similarity_metric` were accepted and then
  ignored entirely.** They now reject any value other than the one the code
  actually implements, instead of silently substituting it.
- `auto_n_prototypes` recognised only the exact string `'kneedle'`; anything
  else, including `'knee'`, was ignored. It now raises.
- `register_ideal()` stores a vector that no scoring path reads, so it never
  affected any output. It now says so, and points at `stereotype_column`.
- **Fixed a non-finite formative value being reported as a confident 0.5** for
  every sample. It now stays NaN, with a warning.
- `auto_n_prototypes='kneedle'` silently reduced the prototype set to one on
  every dataset tried. It now warns when the truncation is drastic, and `knee_`
  finally reports the value that shaped the result.
- **Fixed state leaking between fits on the same estimator.** Fitting text
  after tabular left the tabular feature metadata in place, so the
  visualisations would have plotted the wrong data rather than refusing.
  Refitting also carried over the previous `dropped_columns_` and stereotype
  source, and `fast_mode` was only honoured on the first fit.
- **Fixed a silent config defect**: `to_config()` dropped `feature_weights`, so
  saving and reloading a configuration produced a different fit.
- `transform()` now reports when data falls outside the range seen during `fit`
  and is being clipped to it, instead of collapsing those rows onto the boundary
  in silence.
- Graph edges naming a node outside `range(n_nodes)` now raise instead of
  silently enlarging the graph, which used to distort pagerank, betweenness and
  closeness for every node.
- Dropping a feature column now always warns rather than only under `verbose`,
  and distinguishes a genuinely constant column from one that varies by less
  than the scaler can resolve.
- Fixed an `UnboundLocalError` that crashed stereotypical Shapley explanations
  whenever `shapley_top_n` was less than the number of rows.
- A non-numeric `stereotype_column` now fails at fit time with a message naming
  the column, instead of failing deep inside pandas. Ordered categoricals and
  booleans are accepted.
- Fixed two further silent or crashing paths found while writing the test suite:
  a `stereotype_column` supplied for text data without `text_metadata` was
  ignored, and verbose text or graph fits with a `stereotype_column` crashed.
- Added a pytest suite: 692 tests, 99.5% statement coverage.

**Recent Updates (v0.7.7)**:
- Streaming formative-Shapley computation: each Monte Carlo permutation now updates the value functions incrementally along the growing coalition instead of recomputing them from scratch at every step. Per-fit complexity drops from O(M·n²) to O(M·n) for archetypal and stereotypical significance, and from O(M·n³) to O(M·n²) for prototypical. Rankings are numerically identical to v0.7.6 — only runtime changes.
- The formative step at n = 10,000 now completes in seconds rather than hours, making publication-mode fits on large datasets practical.
- Console and verbose output is now ASCII-only, so logs and the test suites run cleanly under any terminal encoding (including Windows cp1252).

**Recent Updates (v0.7.6)**:
- Added `selected_significance` parameter for selective computation of one significance type
- Fixed prototype feature storage so `transform()` on new data uses correct prototype vectors
- Full Shapley analysis (formative + explanations) now runs correctly on text data paths
- Fixed iterator exhaustion in all text fit/transform methods
- Fixed local/global index mismatch in stereotypical Shapley explanations when subsampling
- Improved error messages when a significance type was not fitted

**Stability**: Production-ready for research use

---

## License

MIT License — See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Amanda S. Barnard

---

## Support

- **Documentation**: See [docs/](docs/) folder or links above
- **Issues**: Report bugs via [GitHub Issues](https://github.com/amaxiom/DataTypical/issues)
- **Questions**: Open a [GitHub Discussion](https://github.com/amaxiom/DataTypical/discussions)

---

## Acknowledgments

DataTypical builds on foundational work in:
- Archetypal analysis (Cutler & Breiman, 1994)
- Facility location optimization (Nemhauser et al., 1978)
- Shapley value theory (Shapley, 1953)
- PCHA optimization (Mørup & Hansen, 2012)

Special thanks to the scientific Python community.

---

## Quick Links

[Documentation](docs/)  
[Quick Start](#quick-start)  
[Examples](docs/EXAMPLES.md)  
[Visualization Guide](docs/VISUALIZATION_GUIDE.md)  
[Report Issues](https://github.com/amaxiom/DataTypical/issues)  
[Discussions](https://github.com/amaxiom/DataTypical/discussions)

---

**Ready to explore your data?**

```bash
pip install datatypical
```

Then see [docs/START_HERE.md](docs/START_HERE.md) for your first analysis!
