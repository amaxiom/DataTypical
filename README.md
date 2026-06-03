# DataTypical

**Scientific Data Significance Rankings with Shapley Explanations**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/datatypical.svg)](https://pypi.org/project/datatypical/)

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

# Three significance perspectives (0-1 normalized ranks)
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

---

## Citation

If you use DataTypical in your research, please cite:

```bibtex
@software{datatypical2026,
  author = {Barnard, Amanda S.},
  title = {DataTypical: Scientific Data Significance Rankings with Shapley Explanations},
  year = {2026},
  url = {https://github.com/amaxiom/DataTypical},
  version = {0.7.7}
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

**Current Version**: 0.7.7

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
