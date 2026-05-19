# START HERE: Welcome to DataTypical

**DataTypical** is a Python library for discovering which instances in your dataset matter most, and understanding why they matter.

---

## What DataTypical Does

Have you ever asked:
- "Which samples in my dataset are most unusual?"
- "Which samples are most representative?"
- "Which samples best match my target?"
- "Why is this particular sample significant?"
- "Which samples define my dataset's structure?"

**DataTypical answers all of these questions simultaneously.**

### The Innovation

Unlike traditional methods that just rank samples, DataTypical:

1. **Explores through three complementary lenses** - archetypal (extreme), prototypical (representative), stereotypical (target-like)
2. **Explains why samples are significant** - via Shapley values showing feature contributions
3. **Discovers formative instances** - samples that CREATE your dataset's structure vs samples that ARE significant

This dual perspective - what IS significant vs what CREATES significance - is a genuinely novel contribution that emerges from the Shapley mechanism.

---

## Quick Example
```python
from datatypical import DataTypical
from datatypical_viz import significance_plot, heatmap, profile_plot

# Load your data
import pandas as pd
data = pd.read_csv('your_data.csv')

# Analyze (with explanations)
dt = DataTypical(shapley_mode=True)
results = dt.fit_transform(data)

# Explore the three perspectives
print(results[['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']].head())

# Visualize dual perspective (actual vs formative)
significance_plot(results, significance='archetypal')

# See which features matter
heatmap(dt, results, significance='archetypal', top_n=20)

# Understand one sample in detail
top_idx = results['archetypal_rank'].idxmax()
profile_plot(dt, top_idx, significance='archetypal')
```

**That's it.** You now have:
- Rankings for all samples across three significance types
- Understanding of which samples are critical vs replaceable
- Feature-level explanations for why samples are significant

---

## Three Complementary Lenses

DataTypical analyzes your data through three simultaneous perspectives:

### 1. Archetypal (Extremeness)
**Question**: Which samples are at the boundaries?  
**Finds**: Outliers, corner cases, unusual combinations  
**Use for**: Discovering edge cases, understanding data range, quality control

**Example**: In drug discovery, archetypal compounds explore extreme regions of chemical space

### 2. Prototypical (Representativeness)  
**Question**: Which samples are most typical?  
**Finds**: Central examples, cluster centers, common patterns  
**Use for**: Dataset summarization, selecting representatives, understanding "normal"

**Example**: In materials science, prototypical samples represent common structural classes

### 3. Stereotypical (Target Similarity)
**Question**: Which samples match my target pattern?  
**Finds**: Samples similar to desired properties or outcomes  
**Use for**: Optimization, target-driven selection, goal-oriented analysis

**Example**: In biomarker discovery, stereotypical samples resemble disease phenotype

**The Power**: You don't choose just one - you get all three perspectives and see how they relate.

---

## Dual Perspective: Actual vs Formative

When you enable `shapley_mode=True`, DataTypical reveals two complementary views:

### Actual Significance (`*_rank`)
**Samples that ARE significant**
- High archetypal rank = this sample IS extreme
- High prototypical rank = this sample IS representative
- High stereotypical rank = this sample IS target-like

### Formative Significance (`*_shapley_rank`)
**Samples that CREATE the structure**
- High formative archetypal = removing this sample would collapse boundaries
- High formative prototypical = removing this sample would create coverage gaps
- High formative stereotypical = removing this sample would change target distribution

### Why Both Matter

**Four Quadrants in significance plot**:
```
     Formative High
          │
  Gap     │  Critical
  Fillers │  (keep all)
──────────┼──────────── Actual High
  Redundant│ Replaceable
          │  (keep one)
          |
     Formative Low
```

- **Critical** (top-right): Both significant AND structure-defining → irreplaceable
- **Replaceable** (bottom-right): Significant but redundant → keep one per cluster  
- **Gap Fillers** (top-left): Not extreme but fill structural holes → often overlooked
- **Redundant** (bottom-left): Neither significant nor necessary → safe to remove

---

## What You Can Discover

### Scientific Discovery
- **Alternative mechanisms**: High-formative samples reveal different pathways to same outcome
- **Boundary definition**: Which samples define the limits of your system
- **Coverage gaps**: Regions where more sampling needed
- **Redundancy**: Clusters of similar samples where one representative suffices

### Dataset Curation
- **Quality control**: Detect truly novel vs known variation
- **Size reduction**: Remove redundant samples while preserving diversity
- **Representative selection**: Choose samples that span the full space
- **Gap identification**: Find undersampled regions

### Model Understanding
- **Feature importance**: Which features drive significance (globally)
- **Individual explanations**: Why THIS sample is significant (locally)
- **Pattern recognition**: Horizontal/vertical patterns in heatmaps reveal structure
- **Mechanism clustering**: Samples may achieve significance via different features

---

## Installation
```bash
pip install datatypical
```

**Requirements**:
- Python ≥ 3.8
- NumPy, Pandas, SciPy, scikit-learn
- Matplotlib, Seaborn (for visualization)
- Numba (for performance)

---

## Your First Analysis: Step by Step

### Step 1: Import and Load Data
```python
from datatypical import DataTypical
import pandas as pd

# Your data as DataFrame
data = pd.read_csv('my_data.csv')
print(f"Dataset: {data.shape[0]} samples × {data.shape[1]} features")
```

### Step 2: Quick Exploration (Fast)
```python
# Without Shapley - just get ranks (seconds)
dt = DataTypical()
results = dt.fit_transform(data)

# See the three perspectives
print(results[['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']])

# Find top samples
top_archetypes = results.nlargest(10, 'archetypal_rank')
print("Top 10 most extreme samples:")
print(top_archetypes)
```

### Step 3: Understand Why (Moderate Speed)
```python
# With Shapley - get explanations (minutes)
dt = DataTypical(shapley_mode=True)
results = dt.fit_transform(data)

# Now you also have formative ranks
print(results[['archetypal_rank', 'archetypal_shapley_rank']])

# Visualize
from datatypical_viz import significance_plot
significance_plot(results, significance='archetypal')
```

### Step 4: Deep Dive into Features
```python
from datatypical_viz import heatmap, profile_plot

# Feature patterns across top samples
heatmap(dt, results, significance='archetypal', top_n=20)

# Individual sample explanation
top_idx = results['archetypal_rank'].idxmax()
profile_plot(dt, top_idx, significance='archetypal')
```

---

## Performance Expectations

### Without Shapley (Exploration)
| Dataset Size | Time |
|--------------|------|
| 1,000 samples | ~5 seconds |
| 10,000 samples | ~30 seconds |
| 100,000 samples | ~5 minutes |

### With Shapley (Explanation + Discovery)
| Dataset Size | Time |
|--------------|------|
| 100 samples | ~10 seconds |
| 1,000 samples | ~5 minutes |
| 10,000 samples | ~60 minutes |

**Strategy**: Start fast (no Shapley), then add explanations for interesting subsets.

---

## Two-Phase Workflow

### Phase 1: Exploration (Fast)
```python
# Quick iteration, large datasets
dt = DataTypical(fast_mode=True)  # No Shapley
results = dt.fit_transform(data)

# Identify interesting regions
interesting_samples = results[results['archetypal_rank'] > 0.8].index
```

### Phase 2: Understanding (Moderate)
```python
# Subset to interesting samples
subset = data.loc[interesting_samples]

# Detailed analysis with explanations
dt = DataTypical(shapley_mode=True, fast_mode=False)
results_detailed = dt.fit_transform(subset)

# Create publication figures
from datatypical_viz import *
significance_plot(results_detailed, significance='archetypal')
heatmap(dt, results_detailed, significance='archetypal', top_n=20)
```

---

## Data Types Supported

DataTypical automatically detects and handles:

**Tabular data** (most common):
```python
df = pd.DataFrame(...)
dt = DataTypical()
results = dt.fit_transform(df)
```

**Text data**:
```python
texts = ["document 1", "document 2", ...]
dt = DataTypical()
results = dt.fit_transform(texts)
```

**Graph data** (networks):
```python
node_features = pd.DataFrame(...)
edges = [(0, 1), (1, 2), ...]
dt = DataTypical()
results = dt.fit_transform(node_features, edges=edges)
```

---

## Key Concepts

### Ranks (0-1 normalized)
- **1.0** = most significant for this type
- **0.0** = least significant for this type
- All samples get a rank for each significance type

### Shapley Values (feature attributions)
- **Positive** = feature increases significance
- **Negative** = feature decreases significance  
- **Magnitude** = strength of contribution
- Sum to the sample's total significance

### Formative vs Actual
- **Actual** = descriptive (what samples ARE)
- **Formative** = causal (what samples CREATE)
- Both perspectives needed for complete understanding

### Global vs Local Ordering
- **Global** = across all samples (typical importance)
- **Local** = for one sample (individual importance)
- Heatmaps always global, profiles can be either

---

## Documentation Roadmap

**Start here** (you are here!)
↓

**QUICK_REFERENCE.md**  
↳ Daily reference, parameters, common workflows

**README.md**  
↳ Package overview, installation, basic usage

**EXAMPLES.md**  
↳ Complete worked examples across domains

**VISUALIZATION_GUIDE.md**  
↳ Detailed guide to plots and interpretation

**Advanced (for researchers)**:
- **INTERPRETATION_GUIDE.md**: Deep dive into formative instances, and how to interpret complex patterns
- **COMPUTATION_GUIDE.md**: Details on implementation for advanced users

---

## Common Questions

**Q: How is this different from PCA or clustering?**  
A: PCA finds linear components, clustering finds groups. DataTypical finds instance-level significance through multiple lenses simultaneously, with explanations for why samples matter.

**Q: Do I need to choose between archetypal/prototypical/stereotypical?**  
A: No! You get all three at once. But if you only need one, set `selected_significance='archetypal'` (or `'prototypical'`/`'stereotypical'`) to skip computing the others and save time.

**Q: Why is Shapley mode slower?**  
A: Shapley values require computing contributions across many sample coalitions. The depth of explanation requires computational work.

**Q: Can I use this for feature selection?**  
A: Indirectly yes - Shapley values reveal which features drive significance, but DataTypical is primarily for instance analysis.

**Q: What if my data has missing values?**  
A: DataTypical requires complete data. Impute missing values before analysis.

**Q: How do I choose n_archetypes and n_prototypes?**  
A: Defaults (8 each) work well for most datasets. Increase for more complex structure, decrease for simpler datasets.

---

## Getting Help

**Documentation**: Read QUICK_REFERENCE.md for daily reference  
**Examples**: See EXAMPLES.md for complete worked examples  
**Issues**: Report bugs and request features on GitHub  
**Questions**: Open a discussion on GitHub for usage questions

---

## What Makes DataTypical Different

### From Other Tools

**Traditional outlier detection**: Only finds extremes, no explanations  
**DataTypical**: Finds extremes AND explains why via features

**Clustering**: Groups samples, picks centroids  
**DataTypical**: Finds representatives that maximize coverage

**Supervised learning**: Optimizes for prediction  
**DataTypical**: Explores significance without predefined targets

**Dimensionality reduction**: Projects to low dimensions  
**DataTypical**: Maintains interpretability in original feature space

### The Unique Contribution

**Formative instances are genuinely novel**. The distinction between samples that ARE significant vs samples that CREATE structure emerges from the Shapley mechanism and doesn't exist in prior methods. This enables:

- Detecting redundancy even among significant samples
- Finding structurally important samples that aren't extreme
- Understanding which samples are irreplaceable vs interchangeable
- Quality control based on structural contribution

---

## Ready to Start?

1. **Install**: `pip install datatypical`
2. **Try Quick Example** (at top of this file)
3. **Read QUICK_REFERENCE.md** for parameters
4. **See EXAMPLES.md** for your domain
5. **Explore your data!**

The best way to understand DataTypical is to use it on your own data. Start with fast exploration, then add Shapley explanations when you find something interesting.

---

**Next recommended reading**: QUICK_REFERENCE.md for daily reference

**For complete examples**: See EXAMPLES.md

**For visualization details**: See VISUALIZATION_GUIDE.md