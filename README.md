# DataTypical

**Scientific Data Significance Rankings with Shapley Explanations**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/datatypical.svg)](https://pypi.org/project/datatypical/)
[![Tests](https://img.shields.io/badge/tests-981%20passing-21918c.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-22a884.svg)](tests/)

DataTypical analyzes datasets through three complementary lenses: archetypal (extreme), prototypical (representative), and stereotypical (target-like), with Shapley value explanations revealing why instances matter and which ones create your dataset's structure.

---

## Key Features

- **Three Significance Types**: Archetypal, prototypical, stereotypical (all computed simultaneously, or selectively)
- **Shapley Explanations**: Feature-level attributions for why samples are significant, computed exactly
- **Formative Discovery**: Distinguish samples that ARE significant from those that CREATE structure
- **Publication Visualizations**: Dual-perspective scatter plots, heatmaps, and profile plots, for tabular, text and graph fits
- **Multi-Modal Support**: Tabular data, text, and graph networks through unified API
- **Exact by Default**: Closed forms for every Shapley game, so results do not depend on a random seed

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

**The Power**: All three computed simultaneously, and different perspectives reveal different insights.

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

This distinction, between what IS significant and what CREATES structure, is a genuinely novel contribution.

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

Since v0.8.2 every Shapley game DataTypical plays has a closed form, so a
publication-mode fit samples nothing. Measured on an Intel Core Ultra 9 285K
(24 threads, 64 GB) under Windows 11, Python 3.13.9, NumPy 2.3.5 and Numba
0.62.1, on 2026-10-22, with ten features, best of two fits after a warm-up:

| Samples | Ranks only | Ranks, `fast_mode` | Shapley | Shapley, `fast_mode` |
| --- | --- | --- | --- | --- |
| 1,000 | 0.30 s | 0.13 s | 0.47 s | 0.21 s |
| 5,000 | 1.65 s | 0.85 s | 5.2 s | 1.0 s |
| 10,000 | 4.0 s | 3.0 s | 19.6 s | 2.8 s |

"Shapley" is `shapley_mode=True` with a `stereotype_column`, so formative
values and explanations for all three significance types. With both computed
by permutation sampling instead (`'monte_carlo'`, 100 permutations, as through
v0.7.7), the same 1,000-sample Shapley fit took 445 seconds, and 103 with
`fast_mode`. These are v0.8.2 timings, and they stand for v0.8.3: its
explanation games add a few milliseconds to an exact fit, and it no longer
computes, and discards, a nearest-neighbour pass over all pairs of samples on
every fit.

Of the Shapley computations, the prototypical formative values are the one
that grows as n^2, so they come to dominate as n grows; the other formative
values cost O(n log n) and the explanations O(d) per sample.

`fast_mode=True` swaps archetypal analysis for NMF and skips the formative
values, which is where the difference in the table comes from. Its other
settings (30 permutations, explanations for the top half of the samples) only
matter when sampling is asked for with `formative_method='monte_carlo'` or
`explanation_method='monte_carlo'`.

### Numba and its NumPy fallback

The distance and coalition kernels are compiled with Numba when it works, and
`datatypical.NUMBA_AVAILABLE` says whether they are in use. Without a working
Numba, vectorised NumPy versions of the same arithmetic run instead, with a
warning at import. Results are identical, and at the sizes above a fit took
at most 4 per cent longer, sometimes less time, because the kernels are a small
part of a fit. Through v0.8.1 there was no fallback, and the kernels ran as
interpreted loops, thousands of times slower.

Compiled kernels are cached on disk, in `NUMBA_CACHE_DIR` when it is set and
otherwise beside the installed module. Where that folder cannot be written,
such as a system-wide install, caching is turned off and the kernels compile
once per session, about a second and a half. Through v0.8.1 such an install
made `import datatypical` hang.

### Optimization Strategy

**Phase 1**: Fast exploration (`fast_mode=True`, no Shapley) to identify
interesting samples.

**Phase 2**: Detailed analysis (`shapley_mode=True`) to generate formative
rankings, explanations, and publication figures.

---

## Key Parameters

The defaults, with the ones most often changed:

```python
DataTypical(
    # Enable explanations and formative analysis
    shapley_mode=False,           # True for explanations and formative ranks

    # Speed vs accuracy
    fast_mode=False,              # True for exploration: NMF, no formative values

    # Significance types
    nmf_rank=8,                   # Number of archetypes (extreme corners)
    n_prototypes=20,              # Number of representatives
    stereotype_column=None,       # Target column for stereotypical
    stereotype_target='max',      # 'max', 'min', or numeric value

    # Archetypal backend
    archetypal_method=None,       # 'aa' (PCHA, required), 'auto' (cascade),
                                  # 'nmf', or None to follow fast_mode

    # Selective computation
    selected_significance=None,   # 'archetypal', 'prototypical', 'stereotypical', or None (all)

    # How the Shapley values are computed
    formative_method='exact',     # or 'monte_carlo' for permutation sampling
    explanation_method='exact',   # or 'monte_carlo' for permutation sampling
    explanation_game='fitted',    # or 'legacy' for the 0.8.2 explanation games
    shapley_top_n=None,           # explain only the top N (or fraction) of samples
    shapley_n_permutations=100,   # used only by 'monte_carlo' (30 in fast_mode)

    # Reproducibility
    random_state=42,

    # Memory management
    max_memory_mb=2048            # Memory limit for chunked operations
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

Through v0.7.7 the formative ranking was a Monte Carlo Shapley estimate, and it
converges far more slowly than the default permutation count suggests. Measured
on synthetic data, two fits differing only in `random_state`:

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

The stereotypical formative values reduce the same way: that value function is
a plain mean over the coalition, which has its own closed form. The
prototypical game does not reduce, but it decomposes: the maximum is attained
at the first neighbour present in the coalition, which turns the value function
into a weighted sum of indicator games, each with a closed form. That costs
O(n^2) rather than O(n log n), and it is exact.

**All three formative games are therefore exact by default, and the formative
ranking no longer depends on `random_state` at all.** To reproduce a result
computed before v0.8.0, pass `formative_method='monte_carlo'`.

The prototypical game was the least convergent of the three under sampling. On
Wine at the documented 100 permutations, five seeds produced five different top
formative prototypes, with pairwise rank correlations of 0.03 to 0.17 and a
top-ten overlap of 0 or 1 out of 10 in nine of the ten seed pairs.

All three closed forms are available directly, if you want the values without
a fit:

```python
from datatypical import (exact_formative_archetypal,
                         exact_formative_prototypical,
                         exact_formative_stereotypical)

exact_formative_archetypal(scaled_X, dt.H_)
exact_formative_prototypical(scaled_X)
exact_formative_stereotypical(target_values, target='max')
```

> **Worth knowing about the stereotypical formative axis.** Its value function
> is `v(S) = |median - tau| - mean over S of |y_i - tau|`, with `tau` the column
> maximum, minimum or numeric target, exactly as `stereotypical_rank` resolves
> it. The closed form is affine in `|y_i - tau|`, so the formative ranking is
> the same ordering as `stereotypical_rank` and carries no information beyond
> `stereotype_column`. Both axes of the stereotypical dual-perspective plot are
> driven by that one column, which makes the panel a curve rather than a
> scatter. This is a property of the published value function, not of the exact
> implementation. The archetypal formative axis is genuinely distinct by
> comparison.
>
> **Changed in v0.8.2.** For `stereotype_target='max'` or `'min'` the formative
> game used to score the excess over the median instead, clipped at zero, which
> tied every sample on the far side of the median. Numeric targets are
> unchanged.

**Check `split_half_rho` before reporting a Monte Carlo formative result:**

```python
dt = DataTypical(shapley_mode=True, shapley_compute_formative=True,
                 formative_method='monte_carlo', shapley_n_permutations=100)
dt.fit_transform(data)

dt.shapley_info_['archetypal_formative']['split_half_rho']
```

It splits the permutations into two independent halves and correlates the two
rankings. Below about 0.7 the ordering is mostly sampling noise, and a
`RuntimeWarning` says so. Raise `shapley_n_permutations` until the number stops
rising. The actual ranks, `archetypal_rank` and the rest, are fully
deterministic and are not affected by any of this.

Sampling stops early only when the estimate has settled, and never before
`max(50, 20 + 10 * shapley_early_stopping_patience)` permutations: 120 at the
default patience, so at the default 100 permutations every run uses all 100.
With `verbose=True` a fit says so, and states how many coalition values the
sampling will compute, before starting it.

### Feature-level explanations

The explanations, which feature makes each sample significant, are Shapley
values too: one game per sample, played over its features. **Since v0.8.3 the
archetypal and prototypical games are the significance scores themselves**,
with the fitted archetypes and prototypes held fixed
(`explanation_game='fitted'`, the default), so a sample's attributions
decompose its rank:

- **Archetypal.** `archetypal_rank` is 0.7 times the sample's largest share of
  archetype membership plus 0.3 times a corner term on two features. The
  membership share comes from a linear projection onto the archetypes, so each
  feature's part of it is exact, and the corner term is a game on two players.
  The attributions sum to `archetypal_rank`.
- **Prototypical.** `prototypical_rank` is half closeness to the nearest
  prototype, in units of the 95th-percentile distance, and half cosine
  similarity to the most similar prototype. The game keeps both halves and
  squares the distance, which makes it additive. The attributions sum to that
  squared-distance form of the score, which orders samples almost exactly as
  the rank does (Spearman 0.99 to 1.00 on the paper's three datasets).
- **Stereotypical.** Unchanged: the sample's distance from the target, as a
  weight, times the mean magnitude of its values over the coalition.

**Since v0.8.2 all three are exact by default** (`explanation_method='exact'`):
each has a closed form, verified against exhaustive coalition enumeration, so
the explanations cost O(d) per sample and need no permutations.

That matters most for wide data. On a corpus of 100 documents and 1,277 terms,
sampled explanations at the default 100 permutations took about twenty-five
minutes per significance type. The exact ones take milliseconds per
significance type, and the whole Shapley fit under a second.

```python
from datatypical import (exact_explanations_archetypal_fitted,
                         exact_explanations_prototypical_fitted,
                         exact_explanations_stereotypical)

exact_explanations_archetypal_fitted(scaled_X, dt.H_, dt.nmf_model_)
exact_explanations_prototypical_fitted(scaled_X, dt.prototype_features_)
exact_explanations_stereotypical(scaled_X, target_values, target='max')
```

Through v0.8.2 the archetypal and prototypical games were defined feature by
feature, without the archetypes or prototypes, and their values did not follow
the ranks: on the paper's datasets the prototypical one, the variance of a
sample's own values, correlated with `prototypical_rank` at -0.18 to -0.81.
`explanation_game='legacy'` keeps them, and `exact_explanations_archetypal` and
`exact_explanations_prototypical` compute them, to reproduce an earlier result.

Pass `explanation_method='monte_carlo'` with `explanation_game='legacy'` to
reproduce a result computed before v0.8.2. With the 0.8.2 games the sampled
values estimate the same quantities, but at 100 permutations they differ from
the exact values by a median of 15 to 27 per cent, and name a different most
important feature for 11 to 26 per cent of samples (measured on 300 rows of the
MAGIC telescope data). The fitted prototypical game is additive, so sampling
it returns its exact values.

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

When you only need one significance type, set `selected_significance` to skip the others entirely, saving substantial compute time:

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

All three serve tabular, text and graph fits: a text fit is labelled by its
vocabulary, a graph fit by its node features and topology columns.

**Sizing.** Each plot is built around a fixed plot area, so long labels enlarge
the figure rather than squeezing the plot. Choose the area with `plot_size`:

```python
significance_plot(results, significance='archetypal', plot_size=(6, 6))
```

Text, markers and colour bars scale with it, so a larger plot keeps its
proportions. `figsize` still fixes the whole canvas instead, as it did before
v0.8.2. When there are more features or samples than can be labelled legibly,
the plot shows as many as fit and says so; `top_features` and `top_n` choose
another number. Long sample labels are shortened in the middle beyond
`max_label_length` characters, and `sample_labels` replaces them outright.

**Style.** Viridis throughout, with two categories in viridis purple and green;
one font size per role (12 points for ticks and legends, 14 for axis labels and
titles); nothing bold; legends outside the plot and without a frame. Importing
`datatypical_viz` leaves matplotlib's global settings alone.

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
- [docs/START_HERE.md](docs/START_HERE.md): friendly introduction and first steps
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md): daily reference for parameters and workflows
- [docs/EXAMPLES.md](docs/EXAMPLES.md): complete worked examples across domains

**Visualization**:
- [docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md): comprehensive guide to plots and interpretation

**Advanced**:
- [docs/INTERPRETATION_GUIDE.md](docs/INTERPRETATION_GUIDE.md): interpreting complex patterns
- [docs/COMPUTATION_GUIDE.md](docs/COMPUTATION_GUIDE.md): implementation details and algorithms

**Benchmarks**:
- [benchmarks/](benchmarks/): structural-significance benchmarks (model-free comparison, bounded metrics, synthetic ground truth) and the formative-Shapley scaling study, re-run with v0.8.2. See its `README.md`, and run `Benchmarks.ipynb` for the same experiments with figures shown inline.

---

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- SciPy ≥ 1.7
- scikit-learn ≥ 1.0
- Matplotlib ≥ 3.6
- Seaborn ≥ 0.11
- Numba ≥ 0.55 (compiled kernels; without it a NumPy fallback runs, with a warning)
- NetworkX ≥ 2.6 (graph topology features)
- py_pcha ≥ 0.1.3 (required for `archetypal_method='aa'`)
- joblib ≥ 1.0 and threadpoolctl ≥ 2.0 (parallelism and thread control)

All eleven are installed by `pip install datatypical`. The installed version is
`datatypical.__version__`.

---

## Citation

If you use DataTypical in your research, please cite:

```bibtex
@software{datatypical2026,
  author = {Barnard, Amanda S.},
  title = {DataTypical: Scientific Data Significance Rankings with Shapley Explanations},
  year = {2026},
  url = {https://github.com/amaxiom/DataTypical},
  version = {0.8.3}
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

**Current Version**: 0.8.3

**Recent Updates (v0.8.3)**, see [CHANGELOG.md](CHANGELOG.md):
- **The archetypal and prototypical explanations now decompose the ranks they
  explain.** Their games are the significance scores with the fitted
  archetypes and prototypes held fixed (new `explanation_game`, default
  `'fitted'`): the archetypal attributions sum to `archetypal_rank`, and the
  prototypical ones to the prototypical score with its distance squared. The
  0.8.2 games did not follow the ranks, and the prototypical one ran against
  `prototypical_rank`. This **changes every archetypal and prototypical
  explanation**; `explanation_game='legacy'` reproduces them. New public
  functions `exact_explanations_archetypal_fitted` and
  `exact_explanations_prototypical_fitted`. See
  [Feature-level explanations](#feature-level-explanations).
- **`transform()` on new rows no longer reports other rows' formative ranks.**
  It used to align them by index label alone, so a new row got those of the
  training row sharing its label; a row that is not a training sample now gets
  NaN, with a warning. Text transforms now scale new documents by the training
  corpus and take `text_metadata=` for them. Fixes from a pre-release audit,
  none of which changes a `fit_transform()` result; see the CHANGELOG.
- pytest suite: 981 tests, 100% statement coverage of both modules, measured
  with `NUMBA_DISABLE_JIT=1` because compiled kernels are invisible to the line
  tracer. `tests/conftest.py` sets that whenever coverage is measuring, so the
  figure cannot be taken the other way by accident; with the JIT active the
  same suite reports 96.1%.

**v0.8.2**, see [CHANGELOG.md](CHANGELOG.md):
- **Fixed an import that could hang.** From a folder the user cannot write to,
  such as a system-wide Anaconda install on Windows, Numba's cache check
  retried up to 2**31 - 1 times. The folder is now tested with one real write,
  and caching is turned off where it would fail.
- **Fixed a Numba fallback that did not fall back.** Without a working Numba
  the compiled kernels ran as interpreted loops, thousands of times slower,
  and nothing said so. Vectorised NumPy versions now run instead, with a
  warning, and `NUMBA_AVAILABLE` reports which path is in use.
- **Added `explanation_method`, defaulting to `'exact'`**: closed forms for
  all three feature-level explanation games, so explanations no longer depend
  on the seed and cost O(d) per sample. New public functions
  `exact_explanations_archetypal`, `exact_explanations_prototypical` and
  `exact_explanations_stereotypical`. This **changes the explanations** any
  earlier version produced; `explanation_method='monte_carlo'` reproduces them.
  See [Feature-level explanations](#feature-level-explanations).
- **The stereotypical formative game now measures the same distance as the
  stereotypical rank** for `stereotype_target='max'` and `'min'`, which
  **changes those formative values**. Numeric targets are unchanged.
- **The visualisations serve text and graph fits**, take a `plot_size`, scale
  their text and markers with it, and no longer render blank or illegible at
  large feature counts. Importing `datatypical_viz` no longer changes
  matplotlib's global settings. See [Visualization](#visualization).
- Fixed a Shapley fit with an ordered categorical `stereotype_column`, a
  `formative_method` error message that named the wrong default, and
  stereotypical explanations drawn as zeros where none existed.
- Added `__version__`, the one place the version is written down.
- Declared `joblib` and `threadpoolctl`, which were imported but never declared.

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
- Added a pytest suite, 738 tests at release.

**Recent Updates (v0.7.7)**:
- Streaming formative-Shapley computation: each Monte Carlo permutation now updates the value functions incrementally along the growing coalition instead of recomputing them from scratch at every step. Per-fit complexity drops from O(M·n²) to O(M·n) for archetypal and stereotypical significance, and from O(M·n³) to O(M·n²) for prototypical. Rankings are numerically identical to v0.7.6; only runtime changes.
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

MIT License. See [LICENSE](LICENSE) for details.

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
