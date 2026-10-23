# DataTypical Package Summary

**Version 0.8.2** | Comprehensive technical reference

---

## Overview

DataTypical is a Python library for explainable instance significance discovery in scientific datasets. It analyzes data through three complementary lenses (archetypal, prototypical, stereotypical) with optional Shapley value explanations revealing why instances are significant and which ones create dataset structure.

**Key Innovation**: Dual-perspective analysis distinguishing samples that ARE significant (actual) from samples that CREATE structure (formative).

**Supported Data Types**: Tabular, text, and graph networks through unified API with automatic detection.

---

## What's New in v0.8.2

- **Exact feature-level explanations**: `explanation_method='exact'` (the default) computes all three explanation games in closed form, so explanations cost O(d) per sample and no longer depend on the seed. New public functions `exact_explanations_archetypal`, `exact_explanations_prototypical` and `exact_explanations_stereotypical`. `explanation_method='monte_carlo'` reproduces earlier results.
- **The stereotypical formative game measures the same distance as the stereotypical rank** for `stereotype_target='max'` and `'min'` (the distance from the column maximum or minimum, not the excess over the median). Numeric targets are unchanged.
- **Import no longer hangs** where Numba cannot write its cache, and **a real NumPy fallback** runs when Numba is missing or broken; `NUMBA_AVAILABLE` says which path is in use.
- **Visualisations serve text and graph fits**, take a `plot_size`, scale text and markers with it, and stay legible at large feature counts. Importing `datatypical_viz` no longer changes matplotlib's global settings.
- **`__version__`**, the single source of the version.
- `joblib` and `threadpoolctl` are now declared dependencies.

## What's New in v0.8.1

- **Archetypal analysis works on NumPy 2**: `py_pcha` calls `np.mat`, removed in NumPy 2.0; the alias is restored before import.
- **`stereotypical_rank` is scale-free**: a column with a small spread no longer collapses every rank to 1.0.
- **`exact_formative_prototypical` refuses NaN** instead of returning confident values.

## What's New in v0.8.0

- **Exact formative values by default** (`formative_method='exact'`): closed forms for all three formative games, with no sampling and no seed dependence.
- **`archetypal_method='aa'` never substitutes another method**: it raises when PCHA is unavailable. `'auto'` keeps the old cascade with warnings, and `archetypal_backend_` records what ran.
- **The working `dtype` no longer changes results**: scaling happens in float64 first.
- Nineteen fixes in all; see the CHANGELOG.

## What's New in v0.7.7

- **Streaming formative-Shapley computation**: Each Monte Carlo permutation now updates the value functions incrementally along the growing coalition instead of re-evaluating them from scratch at every prefix. Per-fit complexity drops from O(M·n²) to O(M·n) for archetypal and stereotypical significance, and from O(M·n³) to O(M·n²) for prototypical significance. The formative step at n = 10,000 falls from hours to seconds. Output is numerically identical to v0.7.6; rankings do not change.
- **ASCII-only console output**: Verbose logs and the bundled test suites no longer emit Unicode glyphs, so they run cleanly under any terminal encoding (including Windows cp1252).

## What's New in v0.7.6

- **`selected_significance` parameter**: Compute only one significance type (`'archetypal'`, `'prototypical'`, or `'stereotypical'`) and skip the others, substantially reducing compute time when only one perspective is needed. Uncomputed ranks are returned as NaN.
- **Fixed prototype transform correctness**: `transform()` on new data now uses stored training prototype feature vectors (`prototype_features_`, `prototype_features_l2_`) rather than incorrectly indexing the new data by the training prototype positions.
- **Text Shapley fully supported**: Both formative instance discovery and feature-level Shapley explanations now run on text data paths, matching the tabular behaviour. Previously text paths silently skipped Shapley analysis.
- **Iterator exhaustion fix**: All text fit/transform methods now materialize iterables to lists at entry, preventing silent failures when a generator-type corpus was passed.
- **Stereotypical explanations index fix**: Corrected a local/global index mismatch in `_fit_shapley_explanations` where `explain_stereotypical_features` received local indices (0..len(subset)-1) but was indexing the full-length target array. `target_values` is now correctly sliced to match each computation tier.
- **Improved error messages**: `_score_with_fitted` now gives specific guidance when a significance type was not fitted due to `selected_significance` mismatch.
- **FAISS removed**: The FAISS import was dead code (only referenced inside a removed internal method) and has been cleaned up.

---

## Installation
```bash
pip install datatypical
```

**Requirements**:
- Python ≥ 3.8
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- SciPy ≥ 1.7
- scikit-learn ≥ 1.0
- Matplotlib ≥ 3.3
- Seaborn ≥ 0.11
- Numba ≥ 0.55 (compiled kernels; a NumPy fallback runs without it, with a warning)
- NetworkX ≥ 2.6
- py_pcha ≥ 0.1.3 (required for `archetypal_method='aa'`)
- joblib ≥ 1.0
- threadpoolctl ≥ 2.0

All are installed by `pip install datatypical`. The installed version is
`datatypical.__version__`.

---

## Core API

### DataTypical Class
```python
from datatypical import DataTypical

dt = DataTypical(                  # the defaults
    shapley_mode=False,
    fast_mode=False,
    nmf_rank=8,                    # number of archetypes
    n_prototypes=20,
    stereotype_column=None,
    stereotype_target='max',       # 'max', 'min', or numeric value
    selected_significance=None,    # 'archetypal', 'prototypical', 'stereotypical', or None (all)
    archetypal_method=None,        # None follows fast_mode: 'aa' or 'nmf'
    formative_method='exact',      # or 'monte_carlo'
    explanation_method='exact',    # or 'monte_carlo'
    shapley_n_permutations=100,    # used only by 'monte_carlo'
    shapley_top_n=None,
    random_state=42,
    verbose=False,
    max_memory_mb=2048
)
```

### Main Methods

#### `fit(X, **kwargs)`
Fit the model to data. The data type is detected from `X` and the keywords.

**Parameters**:
- `X`: Data to fit
  - `pd.DataFrame`: Tabular data (most common)
  - `List[str]`: Text data (auto-detected)
  - `np.ndarray`: Numeric array
- `edges` (or `edge_index`): edge list for graph data, as `(n_edges, 2)` or `(2, n_edges)`; its presence selects the graph path
- `text_metadata`: a DataFrame of per-document columns for text data, needed when `stereotype_column` names one of them
- `vectorizer`: `'tfidf'` (the default) for text data

Columns to carry through to the results without fitting on them, such as
labels, are named with the constructor's `label_columns`.

**Returns**: `self`

#### `transform(X, **kwargs)`
Transform new data using the fitted model.

**Parameters**:
- `X`: Data to transform (same format and features as the fit)
- The same keywords as `fit()`, for example `edges` for graph data

**Returns**: `pd.DataFrame` with rank columns

#### `fit_transform(X, return_ranks_only=None, **kwargs)`
Fit the model and transform the same data in one step.

**Parameters**: as `fit()`, plus `return_ranks_only=True` to return only the
rank columns.

**Returns**: `pd.DataFrame` with rank columns

#### `get_shapley_explanations(sample_idx)`
Get feature-level Shapley explanations for one sample.

**Parameters**:
- `sample_idx`: Index of sample (from original DataFrame)

**Returns**: `Dict[str, np.ndarray]` with keys:
- `'archetypal'`: Feature attributions for archetypal significance
- `'prototypical'`: Feature attributions for prototypical significance
- `'stereotypical'`: Feature attributions for stereotypical significance

**Requires**: `shapley_mode=True`

#### `get_formative_attributions(sample_idx)`
Get sample-level formative attributions.

**Parameters**:
- `sample_idx`: Index of sample

**Returns**: `Dict[str, float]` with keys:
- `'archetypal'`: Sample's contribution to archetypal structure
- `'prototypical'`: Sample's contribution to prototypical structure
- `'stereotypical'`: Sample's contribution to stereotypical structure

**Requires**: `shapley_mode=True`, `fast_mode=False`

---

## Parameters Reference

### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shapley_mode` | bool | `False` | Enable Shapley explanations and formative analysis |
| `fast_mode` | bool | `False` | True for exploration: NMF archetypes, no formative values |
| `stereotype_column` | str | `None` | Column name for stereotypical target (required for stereotypical analysis) |

### Significance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nmf_rank` | int | `8` | Number of archetypal "corners" to find |
| `n_prototypes` | int | `20` | Number of prototypical representatives |
| `archetypal_method` | str | `None` | `'aa'` (PCHA, raises if unavailable), `'auto'` (cascade with warnings), `'nmf'`, or `None` to follow `fast_mode` |

### Shapley Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formative_method` | str | `'exact'` | `'exact'` closed forms, or `'monte_carlo'` permutation sampling |
| `explanation_method` | str | `'exact'` | `'exact'` closed forms, or `'monte_carlo'` permutation sampling |
| `shapley_n_permutations` | int | `100` | Permutations for `'monte_carlo'` (30 in fast_mode) |
| `shapley_top_n` | int/float | `None` | Limit explanations to top N samples (int) or fraction (float) |

### System Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int | `42` | Random seed for reproducibility |
| `verbose` | bool | `False` | Print progress messages |
| `max_memory_mb` | int | `2048` | Memory limit for distance computations |

---

## Output Format

### Without Shapley (`shapley_mode=False`)
```python
results.columns
# ['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']
```

**All ranks are 0-1 normalized**:
- 1.0 = most significant
- 0.0 = least significant

### With Shapley (`shapley_mode=True`)
```python
results.columns
# ['archetypal_rank', 'prototypical_rank', 'stereotypical_rank',
#  'archetypal_shapley_rank', 'prototypical_shapley_rank', 'stereotypical_shapley_rank']
```

**Actual ranks** (`*_rank`): How significant samples ARE  
**Formative ranks** (`*_shapley_rank`): How much samples CREATE structure

---

## Visualization Module

### Import
```python
from datatypical_viz import (
    significance_plot,
    heatmap,
    profile_plot,
    get_top_sample
)
```

### significance_plot()

**Purpose**: Dual-perspective scatter showing actual vs formative significance.
```python
significance_plot(
    results,                          # DataFrame from fit_transform()
    significance='archetypal',        # 'archetypal', 'prototypical', or 'stereotypical'
    color_by=None,                    # Optional column to color points
    size_by=None,                     # Optional column to size points
    labels=None,                      # Optional pd.Series of labels
    label_top=0,                      # Number of top points to label
    quadrant_lines=True,              # Draw quadrant division lines
    quadrant_threshold=(0.5, 0.5),   # (x, y) thresholds for quadrants
    figsize=None,                     # Whole canvas; None builds it around plot_size
    cmap='viridis',                   # Colormap
    title=None,                       # Optional title
    ax=None,                          # Optional existing axes
    plot_size=None                    # Plot area in inches, default (4.1, 4.0)
)
```

**Returns**: `matplotlib.axes.Axes`

**Automatically determines**:
- X-axis: `{significance}_rank` (actual)
- Y-axis: `{significance}_shapley_rank` (formative)

**Interpretation**:
- Top-right: Critical (keep all)
- Top-left: Gap fillers
- Bottom-right: Replaceable (keep one per cluster)
- Bottom-left: Redundant

### heatmap()

**Purpose**: Feature attribution heatmap showing Shapley explanations.
```python
heatmap(
    dt_fitted,                        # Fitted DataTypical instance
    results,                          # DataFrame from fit_transform()
    significance='archetypal',        # 'archetypal', 'prototypical', or 'stereotypical'
    order='actual',                   # 'actual' or 'formative' (instance ordering)
    top_n=20,                         # Number of samples to display
    top_features=None,                # Optional limit on features
    figsize=None,                     # Whole canvas; None builds it around plot_size
    cmap='viridis',                   # Colormap
    center=None,                      # Optional value to centre the colour scale on
    title=None,                       # Optional title
    ax=None,                          # Optional existing axes
    plot_size=None,                   # Plot area in inches, default (4.0, 3.9)
    sample_labels=None,               # Row labels: mapping, Series or function of the index
    max_label_length=24               # Longer row labels lose their middle
)
```

**Returns**: `matplotlib.axes.Axes`

**Key Design**:
- **Always plots explanations** (feature attributions)
- **Features always ordered by global importance** (left to right)
- **Instances ordered by** `order` parameter:
  - `'actual'`: Order by how significant samples ARE
  - `'formative'`: Order by how much samples CREATE structure

**Boxes**: Heatmap and colorbar have black borders for publication quality.

**Warning**: When `order='formative'`, may show zero Shapley values if formative samples aren't in top-N for explanations.

### profile_plot()

**Purpose**: Feature importance profile for individual sample.
```python
profile_plot(
    dt_fitted,                        # Fitted DataTypical instance
    sample_idx,                       # Index of sample to profile
    significance='archetypal',        # 'archetypal', 'prototypical', or 'stereotypical'
    order='local',                    # 'local' or 'global' (feature ordering)
    top_features=None,                # Optional limit on features
    figsize=None,                     # Whole canvas; None builds it around plot_size
    cmap='viridis',                   # Colormap for bar colors
    title=None,                       # Optional title
    ax=None,                          # Optional existing axes
    plot_size=None,                   # Plot area in inches, default (8.7, 3.9)
    max_label_length=24               # Longer feature names lose their middle
)
```

**Returns**: `matplotlib.axes.Axes`

**Key Design**:
- **Bar heights**: Shapley values (positive = increases significance, negative = decreases)
- **Bar colors**: Normalized feature values (yellow = high, purple = low)
- **Feature ordering**:
  - `'local'`: Order by THIS sample's Shapley values
  - `'global'`: Order by average importance across ALL samples

**Use cases**:
- `order='local'`: "What makes THIS sample significant?"
- `order='global'`: "How does this sample compare to typical patterns?"


### get_top_sample()

**Purpose**: Helper to safely get top sample(s) from results.
```python
sample_idx = get_top_sample(
    results,                          # DataFrame from fit_transform()
    rank_column='archetypal_rank',    # Column to rank by
    n=1,                              # Number of samples to return
    mode='max'                        # 'max' or 'min'
)
```

**Returns**: 
- `int` if `n=1`
- `List[int]` if `n>1`
- `None` if column not available

**Handles**: Missing formative data gracefully with informative messages.

All three plots serve tabular, text and graph fits, and share one style:
viridis throughout (two categories in viridis purple and green), 12-point ticks
and legends, 14-point axis labels and titles, nothing bold, and frameless
legends outside the plot. See the Visualization Guide for sizing.

---

## Data Type Support

### Tabular Data (Default)
```python
df = pd.DataFrame(...)
dt = DataTypical()
results = dt.fit_transform(df)
```

**Auto-detection**: Any DataFrame or 2D array

**Features**: All numeric columns (categorical columns ignored)

**Labels**: Preserved in output via index matching

### Text Data
```python
texts = ["document 1", "document 2", ...]
dt = DataTypical()
results = dt.fit_transform(texts)
```

**Auto-detection**: List of strings

**Vectorization**: TF-IDF with configurable parameters

**Features**: TF-IDF term weights

### Graph Data
```python
node_features = pd.DataFrame(...)
edges = [(0, 1), (1, 2), (2, 3), ...]
dt = DataTypical()
results = dt.fit_transform(node_features, edges=edges)
```

**Detection**: Via `edges` parameter

**Features**: Node features + graph topology features (degree, clustering, centrality)

**Applications**: Protein networks, molecular graphs, social networks

---

## Performance Characteristics

### Speed

Measured on an Intel Core Ultra 9 285K (24 threads, 64 GB) under Windows 11, Python 3.13.9, NumPy 2.3.5 and Numba 0.62.1, on 2026-10-22, ten features, best of two fits after a warm-up:

| Samples | Ranks only | Ranks, `fast_mode` | Shapley | Shapley, `fast_mode` |
| --- | --- | --- | --- | --- |
| 1,000 | 0.30 s | 0.13 s | 0.47 s | 0.21 s |
| 5,000 | 1.65 s | 0.85 s | 5.2 s | 1.0 s |
| 10,000 | 4.0 s | 3.0 s | 19.6 s | 2.8 s |

"Shapley" is `shapley_mode=True` with a `stereotype_column`, so formative
values and explanations for all three significance types. With both computed
by permutation sampling instead (`'monte_carlo'`, 100 permutations, as through
v0.7.7), the same 1,000-sample Shapley fit took 445 seconds, and 103 with
`fast_mode`.

Every Shapley game has a closed form, used by default, so these fits sample
nothing. The prototypical formative values are the one O(n^2) step.

### Memory

Distance computations are chunked to stay within `max_memory_mb`, and the
n by n similarity matrix of the prototypical formative game is never formed.
Intermediate arrays are released during transform.

### fast_mode Effects

| Operation | fast_mode=True | fast_mode=False |
|-----------|----------------|-----------------|
| Archetypal method | NMF | PCHA |
| Formative values | Skipped | Computed (exact) |
| Explanations | Top half of the samples | All samples |
| Shapley permutations (`'monte_carlo'` only) | 30 | 100 |
| Speed | See the table above | |

**Recommendation**: Use `fast_mode=True` for exploration, `fast_mode=False` for publication.

---

## Common Workflows

### Workflow 1: Quick Exploration
```python
# No Shapley - just ranks (seconds)
dt = DataTypical()
results = dt.fit_transform(data)

# Find top samples
top_archetypes = results.nlargest(10, 'archetypal_rank')
top_prototypes = results.nlargest(10, 'prototypical_rank')
```

**Use case**: Initial exploration, large datasets, rapid iteration

**Time**: Seconds to minutes

### Workflow 2: Detailed Analysis
```python
# With Shapley - explanations + formative (seconds to minutes)
dt = DataTypical(shapley_mode=True, fast_mode=False)
results = dt.fit_transform(data)

# Visualize
significance_plot(results, significance='archetypal')
heatmap(dt, results, significance='archetypal', top_n=20)

# Profile
top_idx = results['archetypal_rank'].idxmax()
profile_plot(dt, top_idx, significance='archetypal')
```

**Use case**: Understanding mechanisms, publication figures

**Time**: Seconds to minutes: about 20 s at 10,000 samples, and under ten
minutes for the whole CASP example notebook (45,730 rows)

### Workflow 3: Two-Phase Analysis
```python
# Phase 1: Fast exploration
dt_fast = DataTypical(fast_mode=True)
results_fast = dt_fast.fit_transform(data)

# Identify interesting samples
interesting = results_fast[results_fast['archetypal_rank'] > 0.8].index

# Phase 2: Detailed analysis on subset
subset = data.loc[interesting]
dt_detailed = DataTypical(shapley_mode=True, fast_mode=False)
results_detailed = dt_detailed.fit_transform(subset)

# Create publication figures
heatmap(dt_detailed, results_detailed, significance='archetypal')
```

**Use case**: Large datasets requiring detailed understanding

**Time**: Optimal (fast exploration + targeted analysis)

### Workflow 4: Transform New Data
```python
# Fit on training set
dt = DataTypical(shapley_mode=True)
results_train = dt.fit_transform(train_data)

# Transform test set
results_test = dt.transform(test_data)

# Compare distributions
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(results_train['archetypal_rank'], alpha=0.5, label='Train', color='#440154')
ax.hist(results_test['archetypal_rank'], alpha=0.5, label='Test', color='#32b67a')
ax.legend(frameon=False)
```

**Use case**: Quality control, anomaly detection, model validation

---

## Quick Interpretation Guide

### Archetypal Rank (Extremeness)

**High (>0.7)**:
- Boundary samples
- Unusual feature combinations
- Outliers
- **Use for**: Edge case discovery, range understanding

**Low (<0.3)**:
- Central samples
- Typical combinations
- Interior of distribution
- **Use for**: Understanding "normal"

### Prototypical Rank (Representativeness)

**High (>0.7)**:
- Central, typical samples
- Close to many neighbors
- Cluster centers
- **Use for**: Dataset summarization, representatives

**Low (<0.3)**:
- Outliers
- Far from others
- Unique samples
- **Use for**: Anomaly detection

### Stereotypical Rank (Target Similarity)

**High (>0.8)**:
- Very similar to target pattern
- Match desired properties
- Optimal samples
- **Use for**: Target-driven selection

**Requires**: `stereotype_column` parameter set

### Formative Ranks (Structure Creation)

**High formative + High actual**: Critical - irreplaceable  
**High formative + Low actual**: Gap fillers - structurally important  
**Low formative + High actual**: Replaceable - many similar samples  
**Low formative + Low actual**: Redundant - safe to remove

---

## Feature Attribution Interpretation

### Shapley Values (from get_shapley_explanations)

**Positive values**: Feature increases significance  
**Negative values**: Feature decreases significance  
**Magnitude**: Strength of contribution  
**Sum**: Total significance score

### In Heatmaps

**Horizontal patterns**: Features consistently important across samples  
**Vertical patterns**: Samples with similar feature profiles  
**Bright colors**: Strong contributions (positive or negative)  
**Neutral colors**: Features don't matter for these samples

### In Profile Plots

**Tall bars**: Strong feature contributions  
**Bar direction**: Positive vs negative contribution  
**Bar color**: Feature value (yellow = high, purple = low)  
**Combination**: Tall + yellow = high value of important feature

---

## Optimization Tips

### For Large Datasets (>10,000 samples)
```python
dt = DataTypical(
    fast_mode=True,              # NMF archetypes, no formative values
    shapley_mode=False,          # Skip Shapley initially
    nmf_rank=6,                  # Fewer archetypes
    n_prototypes=6,              # Fewer prototypes
    max_memory_mb=16000          # Increase if available
)
```

### For Shapley at Scale
```python
dt = DataTypical(
    shapley_mode=True,
    fast_mode=True,              # skips the O(n^2) prototypical formative step
    shapley_top_n=500            # explain only the top 500 samples
)
```

### For Publication
```python
dt = DataTypical(
    shapley_mode=True,
    fast_mode=False,
    archetypal_method='aa'       # PCHA archetypes; raises rather than substitute
)
```

The formative values and explanations are exact by default, so there is no
permutation count to raise.

---

## Reproducibility

### Setting Random Seed
```python
dt = DataTypical(random_state=42)
```

**Deterministic**:
- Archetypal analysis (both NMF and PCHA)
- Prototypical selection
- Stereotypical ranking

**Exact, so independent of random_state**:
- Formative values and explanations (the defaults, `'exact'`)

**Stochastic** (affected by random_state):
- NMF initialization (when fast_mode=True)
- Shapley permutation order, only with `formative_method='monte_carlo'` or `explanation_method='monte_carlo'`

**Note**: Results may vary slightly across different NumPy/SciPy versions due to internal algorithm changes.

---

## Troubleshooting

### "No explanations available for top N instances"

**Cause**: `shapley_top_n` limits which samples have explanations computed.

**Solution**:
- Increase `shapley_top_n`
- Use `fast_mode=False` for all samples
- Select samples within the top-N for the significance type

### Zero Shapley values in formative ordering

**Cause**: Sample is formative but not in top-N for actual significance.

**Meaning**: Expected behavior - sample creates structure but isn't individually extreme.

**Action**: This is informative, not an error.

### "Feature dimension mismatch"

**Cause**: Transform data has different features than fit data.

**Solution**: Ensure same features in same order for fit and transform.

### Memory errors

**Cause**: Distance computations for large datasets.

**Solution**:
- Increase `max_memory_mb`
- Reduce `shapley_top_n`
- Use `fast_mode=True`
- Process in batches

### Slow performance

**Cause**: The prototypical formative values (O(n^2)) at large n, or permutation sampling if `'monte_carlo'` was asked for.

**Solution**:
- Use `fast_mode=True`, which skips the formative values
- Use the exact defaults rather than `'monte_carlo'`
- Set `shapley_top_n=500`
- Two-phase workflow (fast exploration → detailed subset)

---

## Version History

### v0.8.2 (Current)

See [What's New in v0.8.2](#whats-new-in-v082) above, and the CHANGELOG for
every release.

### v0.7

**Changes**:
- Simplified visualization API (removed mode confusion)
- Always-global feature ordering in heatmaps
- Cleaned output (only rank columns)
- Publication-ready boxed heatmaps
- Improved memory management
- Enhanced error messages

**Stability**: Production-ready for research use

---

## Citation
```bibtex
@software{datatypical2026,
  author = {Barnard, Amanda S.},
  title = {DataTypical: Scientific Data Significance Rankings with Shapley Explanations},
  year = {2026},
  url = {https://github.com/amaxiom/DataTypical},
  version = {0.8.2}
}
```

---

## License

MIT License. See [LICENSE](../LICENSE) for full text.

Copyright (c) 2026 Amanda S. Barnard

---

## Support

**Documentation**: 
- [START_HERE.md](START_HERE.md) - Getting started
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Daily reference
- [EXAMPLES.md](EXAMPLES.md) - Complete examples
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Visualization details

**Issues**: [GitHub Issues](https://github.com/amaxiom/datatypical/issues)

**Questions**: [GitHub Discussions](https://github.com/amaxiom/datatypical/discussions)

---

## Quick Reference Card

### Import
```python
from datatypical import DataTypical
from datatypical_viz import significance_plot, heatmap, profile_plot
```

### Fit and Transform
```python
dt = DataTypical(shapley_mode=True)
results = dt.fit_transform(data)
```

### Get Explanations
```python
explanations = dt.get_shapley_explanations(sample_idx)
```

### Visualize
```python
significance_plot(results, significance='archetypal')
heatmap(dt, results, significance='archetypal', order='actual', top_n=20)
profile_plot(dt, sample_idx, significance='archetypal', order='local')
```

### Find Top Samples
```python
top_idx = results['archetypal_rank'].idxmax()
top_formative = results['archetypal_shapley_rank'].idxmax()
```

---

**For detailed examples, see [EXAMPLES.md](EXAMPLES.md)**

**For interpretation guidance, see [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)**
