# DataTypical

**Scientific Data Significance Rankings with Shapley Explanations**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DataTypical analyzes datasets through three complementary lenses: archetypal (extreme), prototypical (representative), and stereotypical (target-like), with Shapley value explanations revealing why instances matter and which ones create your dataset's structure.

---

## Key Features

**Three Significance Types**: Archetypal, prototypical, stereotypical (all computed simultaneously)
**Shapley Explanations**: Feature-level attributions for why samples are significant  
**Formative Discovery**: Distinguish samples that ARE significant from those that CREATE structure  
**Publication Visualizations**: Dual-perspective scatter plots, heatmaps, and profile plots  
**Multi-Modal Support**: Tabular data, text, and graph networks through unified API  
**Performance Optimized**: Fast exploration mode and efficient Shapley computation  

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

### Speed Benchmarks

| Dataset Size | Without Shapley | With Shapley |
|--------------|-----------------|--------------|
| 1,000 samples | ~5 seconds | ~5 minutes |
| 10,000 samples | ~30 seconds | ~60 minutes |

### Optimization Strategy

**Phase 1**: Fast exploration (`fast_mode=True`, no Shapley)  
↓ Identify interesting samples  
**Phase 2**: Detailed analysis (`shapley_mode=True`, subset to interesting samples)  
↓ Generate explanations and publication figures

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
    
    # Shapley optimization
    shapley_top_n=500,            # Limit explanations to top N
    shapley_n_permutations=100,   # Number of permutations (30 in fast_mode)
    
    # Reproducibility
    random_state=None,            # Set for reproducible results
    
    # Memory management
    max_memory_mb=8000            # Memory limit for operations
)
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

See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for detailed interpretation.

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
- [START_HERE.md](START_HERE.md) - Friendly introduction and first steps
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Daily reference for parameters and workflows
- [EXAMPLES.md](EXAMPLES.md) - Complete worked examples across domains

**Visualization**:
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Comprehensive guide to plots and interpretation

**Advanced**:
- [FORMATIVE_GUIDE.md](FORMATIVE_GUIDE.md) - Deep dive into formative instances
- [INTERPRETATION_GUIDE.md](INTERPRETATION_GUIDE.md) - Interpreting complex patterns
- [COMPUTATIONAL_GUIDE.md](COMPUTATIONAL_GUIDE.md) - Implementation details

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
  url = {https://github.com/amaxiom/datatypical},
  version = {0.7}
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

**Current Version**: 0.7

**Recent Updates**:
- Simplified visualization API (removed mode confusion)
- Always-global feature ordering in heatmaps
- Cleaned output (only rank columns)
- Publication-ready boxed heatmaps
- Improved memory management

**Stability**: Production-ready for research use

---

## License

MIT License - See [LICENSE](LICENSE) for details.

Copyright (c) 2025 Amanda S. Barnard

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

---

## Support

- **Documentation**: See docs/ folder or links above
- **Issues**: Report bugs via [GitHub Issues](https://github.com/amaxiom/datatypical/issues)
- **Questions**: Open a [GitHub Discussion](https://github.com/amaxiom/datatypical/discussions)

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

[Documentation](https://github.com/amaxiom/datatypical/docs)  
[Quick Start](#quick-start)  
[Examples](EXAMPLES.md)  
[Visualization Guide](VISUALIZATION_GUIDE.md)  
[Report Issues](https://github.com/amaxiom/datatypical/issues)  
[Discussions](https://github.com/amaxiom/datatypical/discussions)

---

**Ready to explore your data?**

```bash
pip install datatypical
```

Then see [START_HERE.md](START_HERE.md) for your first analysis!
```
