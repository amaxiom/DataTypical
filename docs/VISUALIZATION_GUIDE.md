# DataTypical Visualization Guide

**Version 0.7.6**

## Overview

The `datatypical_viz.py` module provides publication-quality visualizations for exploring instance significance through Shapley value explanations. The module contains three complementary visualization functions that reveal **why** instances are significant and **how** they relate to dataset structure, plus one helper utility.

**Key Principle**: All visualizations display **explanations** (feature attributions showing why samples ARE significant), with flexible ordering options to explore different perspectives.

**Note**: All three visualization functions require a **tabular** DataTypical fit. Text and graph fits do not produce `feature_columns_` or `_df_original_fit`, so calling these functions on a text/graph model raises a `RuntimeError` with a clear message.

---

## The Four Functions

### 1. `significance_plot()` - Dual-Perspective Scatter (Hero Visualization)

**Purpose**: Reveals the relationship between actual significance (samples that ARE significant) and formative significance (samples that CREATE the structure).

**What It Shows**: 
- X-axis: Actual rank (how significant each sample IS)
- Y-axis: Formative rank (how much each sample CREATES structure)
- Each point represents one sample

**When to Use**:
- Discovering which samples are critical vs replaceable
- Identifying gap-filling instances
- Understanding redundancy in your dataset
- Quality control for dataset composition

**Key Parameters**:
- `significance`: Which type to visualize ('archetypal', 'prototypical', 'stereotypical')
  - Automatically determines x-axis: `{significance}_rank` and y-axis: `{significance}_shapley_rank`
- `color_by`: Column to color points by (e.g., property values)
- `label_top`: Number of top samples to label

**Example**:
```python
ax = significance_plot(
    results,
    significance='archetypal',
    color_by='property_column',
    label_top=5
)
```

**Interpretation - The Four Quadrants**:
```
     High Formative
          │
  Gap     │  Critical
  Fillers │  (keep all)
──────────┼──────────── High Actual
 Redundant│ Replaceable
          │  (keep one)
          |
     Low Formative
```

**Quadrant Meanings**:

1. **Top-Right (Critical)**: High actual AND high formative
   - These samples ARE significant AND CREATE the structure
   - Keep all of these - they're irreplaceable
   - Example: Unique boundary-defining samples

2. **Top-Left (Gap Fillers)**: Low actual but high formative
   - Not particularly extreme themselves, but fill important structural gaps
   - Often overlooked by standard methods
   - Removing them creates holes in coverage
   - Example: Samples that bridge between clusters

3. **Bottom-Right (Replaceable)**: High actual but low formative
   - Very significant, but many similar samples exist
   - Keep one representative per cluster
   - Example: Multiple samples at same extreme

4. **Bottom-Left (Redundant)**: Low actual AND low formative
   - Neither significant nor structurally important
   - Safe to remove without loss
   - Example: Average samples with many neighbors

---

### 2. `heatmap()` - Feature Attribution Heatmap

**Purpose**: Shows which features contribute to sample significance through Shapley value explanations.

**What It Shows**:
- Rows: Samples (instances)
- Columns: Features (always ordered by global importance)
- Colors: Shapley values (feature contributions to significance)
- Box outlines around heatmap and colorbar for publication quality

**Key Parameters**:
- `significance`: Which type to visualize ('archetypal', 'prototypical', 'stereotypical')
- `order`: How to order samples/rows ('actual' or 'formative')
- `top_n`: Number of samples to display
- `top_features`: Optional limit on number of features to show

**Example**:
```python
ax = heatmap(
    dt,
    results=results,
    significance='archetypal',
    order='actual',          # Order samples by actual ranks
    top_n=20
)
```

**Sample Ordering Options (`order` parameter)**:

**`order='actual'`** (default):
- Orders samples by how significant they ARE
- Top row = most archetypal/prototypical/stereotypical sample
- Shows feature patterns in your most significant samples
- Use when: Exploring what makes top samples significant

**`order='formative'`**:
- Orders samples by how much they CREATE structure
- Top row = most structure-defining sample
- Shows feature patterns in samples that drive the analysis
- Use when: Understanding which samples shape your dataset
- **Note**: May show zero Shapley values if formative samples aren't in top-N explanations
  - These samples CREATE structure globally but aren't themselves highly significant
  - Determined by `shapley_top_n` parameter

**Feature Ordering**:
- Features are ALWAYS ordered by global importance (left to right)
- Global importance = average |Shapley value| across ALL samples
- Left-most features matter most for this significance type overall
- Ensures consistent, interpretable column ordering across all rows

**Color Interpretation**:
- **Positive values** (warm colors): Features that INCREASE significance
- **Negative values** (cool colors): Features that DECREASE significance
- **High absolute values** (bright): Strong feature contributions
- **Near zero** (neutral): Feature doesn't matter much for this sample

**Pattern Recognition**:
- **Horizontal patterns** (same color across rows): 
  - Features consistently important across samples
  - Defines the "signature" of this significance type
  
- **Vertical patterns** (same color down columns): 
  - Samples with similar feature attribution profiles
  - May indicate clusters or mechanisms

- **Checkerboard patterns**: 
  - Different samples achieve significance through different features
  - Suggests multiple pathways or mechanisms

- **Dominant columns**: 
  - A few features drive most of the significance
  - Simple, interpretable structure

- **Distributed colors**: 
  - Many features contribute
  - Complex, multifactorial significance

**Warning Messages**:

When using `order='formative'`, you may see:
```
⚠ Warning: X/Y top formative instances have zero Shapley values
  This occurs when a formative instance is not in the top instances
  (determined by shapley_top_n parameter)
  These instances CREATE structure but are not themselves highly significant
```

**What this means**: The sample is formative (important for structure) but wasn't in the top-N for actual significance, so detailed feature explanations weren't computed. This is expected behavior when using `shapley_top_n` optimization.

---

### 3. `profile_plot()` - Individual Sample Profile

**Purpose**: Deep-dive into a single sample to understand exactly which features make it significant.

**What It Shows**:
- X-axis: Features (ordered by importance)
- Y-axis: Shapley values for this sample
- Bar heights: Magnitude of feature contribution
- Bar colors: Normalized feature values (yellow = high value, purple = low value)
- Bars above zero: Features increasing significance
- Bars below zero: Features decreasing significance

**Key Parameters**:
- `sample_idx`: Which sample to profile (use DataFrame index from results)
- `significance`: Which type to analyze ('archetypal', 'prototypical', 'stereotypical')
- `order`: Feature ordering method ('local' or 'global')
- `top_features`: Optional — show only the N most important features

**Example**:
```python
# Profile the most archetypal sample
top_idx = results['archetypal_rank'].idxmax()
ax = profile_plot(
    dt,
    sample_idx=top_idx,
    significance='archetypal',
    order='local'
)

# Profile a formative sample
top_formative = results['archetypal_shapley_rank'].idxmax()
ax = profile_plot(
    dt,
    sample_idx=top_formative,
    significance='archetypal',
    order='global'
)
```

**Feature Ordering Options (`order` parameter)**:

**`order='local'`** (default):
- Orders features by importance FOR THIS SAMPLE ONLY
- Left-most features have largest |Shapley values| for this sample
- Shows this sample's unique feature importance profile
- Use when asking: "What makes THIS specific sample significant?"

**`order='global'`**:
- Orders features by average importance across ALL samples
- Left-most features are most important across the dataset
- Shows how this sample's profile compares to typical patterns
- Use when asking: "How does this sample compare to global importance?"
- Based on mean(|explanations|) across all samples
- Raises `RuntimeError` if the Phi matrix for the requested significance is not available (e.g., `selected_significance` excluded it)

**Reading the Plot**:

**Bar Height** (Shapley value):
- Tall positive bars: Features strongly INCREASE significance
- Tall negative bars: Features strongly DECREASE significance
- Short bars: Features don't matter much for this sample

**Bar Color** (normalized feature value):
- Yellow/bright: HIGH value for this feature (relative to dataset)
- Purple/dark: LOW value for this feature (relative to dataset)

**Combined Interpretation**:
- **Tall + Yellow**: High value of important feature → increases significance
- **Tall + Purple**: Low value of important feature → different from typical
- **Positive + Yellow**: High value makes sample MORE significant
- **Negative + Purple**: Low value makes sample LESS significant

**Common Patterns**:

1. **Few Dominant Features**:
   - A handful of tall bars, rest are short
   - Significance driven by specific features
   - Simple, interpretable profile
   - Example: Extreme in just 2-3 dimensions

2. **Many Contributing Features**:
   - Many moderately tall bars
   - Significance from combination of many features
   - Complex, multifactorial profile
   - Example: Unusual across many dimensions

3. **Consistent Colors**:
   - All bars similar color (all yellow or all purple)
   - Monotonic relationship with significance
   - Either uniformly high or uniformly low values
   - Example: Sample scaled consistently across features

4. **Mixed Colors**:
   - Yellow and purple bars scattered
   - Complex interaction of high/low values
   - Non-monotonic patterns
   - Example: High in some features, low in others

5. **Negative Bars**:
   - Bars extending below zero line
   - Features that DECREASE significance
   - May represent "opposite" of target pattern
   - Example: Low values where high values are typical

---

### 4. `get_top_sample()` - Safe Rank Helper

**Purpose**: Safely retrieve top sample index/indices from results, handling missing formative data gracefully (prints a helpful message instead of raising an error when the column is all NaN).

**Key Parameters**:
- `results`: Results DataFrame from `fit_transform()`
- `rank_column`: Column to rank by (e.g., `'archetypal_rank'`, `'archetypal_shapley_rank'`)
- `n`: Number of top samples to return (default: 1)
- `mode`: `'max'` for highest values, `'min'` for lowest

**Returns**: Single `int` if `n=1`, `list` if `n>1`, or `None` if data not available.

**Example**:
```python
# Safe retrieval - won't crash if formative data missing
top_idx = get_top_sample(results, 'archetypal_rank')
top_formative = get_top_sample(results, 'archetypal_shapley_rank')  # None if not computed

if top_formative is not None:
    profile_plot(dt, top_formative, significance='archetypal', order='global')

# Get top 5
top_5 = get_top_sample(results, 'prototypical_rank', n=5)
```

---

## Impact of `selected_significance` on Visualizations

When fitting with `selected_significance='archetypal'` (or `'prototypical'`/`'stereotypical'`), only that significance type is computed. The Phi matrices for the other types remain `None`. This affects visualizations:

- **`significance_plot()`**: Will show "Formative data not available" if the requested type was not computed
- **`heatmap()`**: Will show "Explanations not available" message if the requested Phi is None
- **`profile_plot()`**: Raises `RuntimeError` if the requested significance type has no explanations, or if `order='global'` is requested for a Phi that is None

```python
# Only archetypal computed
dt = DataTypical(shapley_mode=True, selected_significance='archetypal')
results = dt.fit_transform(data)

profile_plot(dt, top_idx, significance='archetypal')   # works
profile_plot(dt, top_idx, significance='prototypical') # RuntimeError: no explanations for 'prototypical'
```

---

## Understanding Feature Importance: Local vs Global

### In Profile Plots

**Local Ordering** (`order='local'`):
- Specific to ONE sample
- Shows what matters most for THIS sample
- Different samples can have different feature orders
- Use case: Understanding individual samples in detail

**Global Ordering** (`order='global'`):
- Same for ALL samples
- Shows what matters most on average across dataset
- Consistent ordering enables comparison across samples
- Matches the heatmaps
- Use case: Comparing samples to typical importance patterns

### In Heatmaps

**Always Global**:
- All rows share the SAME column ordering
- Based on average importance across ALL samples
- Ensures interpretability and comparability
- Left-most columns are most important overall

**Why not local for heatmaps?**
- Each row can't have its own column order (impossible to display)
- "Local to subset" would be arbitrary and inconsistent
- Global ordering provides clear, comparable structure

---

## Complete Workflow Examples

### Example 1: Discovering Archetypal Patterns

**Goal**: Understand which samples are extreme and why
```python
from datatypical import DataTypical
from datatypical_viz import significance_plot, heatmap, profile_plot

# Fit model
dt_model = DataTypical(shapley_mode=True, fast_mode=False)
results = dt_model.fit_transform(data)

# Step 1: Overview - Which samples are critical vs replaceable?
ax1 = significance_plot(results, significance='archetypal')
# Look for: Clusters in different quadrants, gaps vs redundancy

# Step 2: Feature patterns - What makes top samples archetypal?
ax2 = heatmap(
    dt_model, results,
    significance='archetypal',
    order='actual',
    top_n=15
)
# Look for: Horizontal patterns (common features), vertical clusters (similar samples)

# Step 3: Deep dive - Why is the #1 sample archetypal?
top_idx = results['archetypal_rank'].idxmax()
ax3 = profile_plot(dt_model, top_idx, significance='archetypal', order='local')
# Look for: Which specific features make this sample extreme
```

**Interpretation Flow**:
1. Significance plot shows 3 clusters in top-right (critical extremes)
2. Heatmap reveals they use different features (3 vertical patterns)
3. Profile plots show each uses a different "pathway" to extremeness

**Discovery**: Multiple mechanisms for achieving archetypal status!

---

### Example 2: Understanding Formative Instances

**Goal**: Find samples that define dataset structure, even if not extreme
```python
# Step 1: Identify formative vs actual significance
ax1 = significance_plot(results, significance='prototypical')
# Look for: Top-left quadrant (gap fillers), bottom-right (replaceable)

# Step 2: Examine formative samples
ax2 = heatmap(
    dt_model, results,
    significance='prototypical',
    order='formative',    # Order by structure-creation
    top_n=20
)
# Note: May see zeros - formative samples not in top-N for prototypical

# Step 3: Profile a gap-filler (high formative, low actual)
gap_fillers = results[
    (results['prototypical_shapley_rank'] > 0.8) &
    (results['prototypical_rank'] < 0.3)
]
if len(gap_fillers) > 0:
    idx = gap_fillers.index[0]
    ax3 = profile_plot(dt_model, idx, significance='prototypical', order='global')
    # Look for: Moderate values across many features (fills space between clusters)
```

**Interpretation Flow**:
1. Significance plot reveals 5 samples in top-left (gap fillers)
2. These samples aren't particularly representative themselves
3. But they fill structural gaps that would otherwise exist

**Discovery**: Removing these 5 would create coverage holes!

---

### Example 3: Comparing Multiple Perspectives

**Goal**: See how the same sample looks across all three significance types
```python
from datatypical_viz import profile_plot

# Deep dive: Same sample across all three significance types
top_idx = results['archetypal_rank'].idxmax()

fig, axes = plt.subplots(1, 3, figsize=(30, 5))
profile_plot(dt_model, top_idx, significance='archetypal', order='global', ax=axes[0])
profile_plot(dt_model, top_idx, significance='prototypical', order='global', ax=axes[1])
profile_plot(dt_model, top_idx, significance='stereotypical', order='global', ax=axes[2])
plt.tight_layout()

# Look for: Different feature importance patterns across significance types
```

**Interpretation**:
- Same sample may be extreme (archetypal) due to features A, B
- But representative (prototypical) due to features C, D
- And optimal (stereotypical) due to features E, F

**Discovery**: Different significance types reveal different aspects of the same sample!

---

### Example 4: Dataset Curation

**Goal**: Decide which samples to keep/remove for a curated subset
```python
# Identify redundant samples (bottom-right quadrant)
redundant = results[
    (results['archetypal_rank'] > 0.7) &
    (results['archetypal_shapley_rank'] < 0.3)
]

print(f"Found {len(redundant)} replaceable samples")

# Cluster them to keep one per cluster
from sklearn.cluster import KMeans
feature_cols = [c for c in data.columns if c != 'label']
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(data.loc[redundant.index, feature_cols])

# Keep highest-ranked sample from each cluster
keepers = redundant.groupby(clusters).apply(
    lambda x: x.nlargest(1, 'archetypal_rank')
)

print(f"Reduced from {len(redundant)} to {len(keepers)} samples")

# Visualize remaining diversity
curated_data = data.drop(redundant.index.difference(keepers.index))
```

**Interpretation**:
1. Found 50 high-archetypal, low-formative samples
2. They clustered into 5 groups (5 different "types" of extremes)
3. Kept 1 per group, removed 45 redundant samples
4. Maintained diversity while reducing size 90%

**Discovery**: Can dramatically reduce dataset size while preserving structure!

---

### Example 5: Mechanism Discovery in Drug Design

**Goal**: Find alternative mechanisms for achieving target property
```python
# Focus on stereotypical (target-optimal) samples
dt_model = DataTypical(
    shapley_mode=True, 
    stereotype_column='activity',
    fast_mode=False
)
results = dt_model.fit_transform(compounds)

# Significance plot colored by activity
ax = significance_plot(
    results,
    significance='stereotypical',
    color_by='activity',
    label_top=10
)

# Find high-formative stereotypical samples (unique mechanisms)
unique_mechanisms = results[
    (results['stereotypical_shapley_rank'] > 0.8)
].nlargest(5, 'stereotypical_rank')

# Profile each to understand mechanism
fig, axes = plt.subplots(len(unique_mechanisms), 1, figsize=(12, 4*len(unique_mechanisms)))
for i, idx in enumerate(unique_mechanisms.index):
    profile_plot(dt_model, idx, significance='stereotypical', order='local', ax=axes[i])
```

**Interpretation**:
1. Found 5 high-activity compounds with high formative ranks
2. Each profile shows different feature patterns
3. Compound A: High potency via features [X, Y]
4. Compound B: High potency via features [Z, W]
5. Different structural pathways to same activity!

**Discovery**: Multiple scaffolds for drug design!

---

## Troubleshooting

### "No explanations available for top N instances"

**Cause**: When using `shapley_top_n`, explanations are computed only for the union of top-N samples per metric. Requesting samples outside this set yields no explanations.

**Solution**: 
- Increase `shapley_top_n` when fitting
- Or use `fast_mode=False` to compute explanations for all samples
- Or select samples that are in the top-N for the significance type you're visualizing

### Zero Shapley values in formative ordering

**Cause**: Sample is formative (creates structure) but not in top-N for actual significance, so detailed explanations weren't computed.

**What it means**: This is expected! The sample is important globally but not individually extreme.

**Solution**: Not an error — it reveals the distinction between formative and actual significance. Increase `shapley_top_n` to raise the chance all formative instances have explanations.

### "No explanations available for '{significance}'"

**Cause**: `profile_plot()` was called for a significance type excluded by `selected_significance`.

**Solution**: Refit with `selected_significance=None` (compute all three) or set `selected_significance` to include the type you need.

### "profile_plot()/heatmap() requires a tabular DataTypical fit"

**Cause**: Model was fit on text or graph data, which does not store `feature_columns_` or `_df_original_fit`.

**Solution**: These visualizations only work with tabular (DataFrame) input.

### "Global ordering requires {significance} explanations, which are not available"

**Cause**: `profile_plot(..., order='global')` called but the Phi matrix for that significance type is None (either not computed or excluded by `selected_significance`).

**Solution**: Use `order='local'` instead, or refit ensuring the significance type is included.

### Features don't match between heatmap and profile plot

**Cause**: Heatmap always uses global ordering; profile plot uses local or global based on `order` parameter.

**Solution**: Use `order='global'` in profile plot to match heatmap column order.

---

### Figure Sizing

**Single plot**:
- Significance plot: `figsize=(6, 5)`
- Heatmap: `figsize=(8, 6)` or wider if many features
- Profile plot: `figsize=(12, 5)`

**Multi-panel**:
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
significance_plot(results, significance='archetypal', ax=axes[0])
heatmap(dt_model, results, significance='archetypal', ax=axes[1])
profile_plot(dt_model, top_idx, significance='archetypal', ax=axes[2])
```

### Saving
```python
ax.figure.savefig('figure.png', dpi=300, bbox_inches='tight')
ax.figure.savefig('figure.pdf', bbox_inches='tight')  # Vector format
```

---

## Summary

### When to Use Each Plot

| Question | Use This Plot |
|----------|---------------|
| Which samples are critical? | `significance_plot()` |
| Which samples are redundant? | `significance_plot()` (bottom-right quadrant) |
| What features drive significance overall? | `heatmap()` with `order='actual'` |
| What features define structure? | `heatmap()` with `order='formative'` |
| Why is THIS sample significant? | `profile_plot()` with `order='local'` |
| How does this compare to typical? | `profile_plot()` with `order='global'` |
| Are there clusters/mechanisms? | `heatmap()` looking for vertical patterns |
| What's the most important feature? | `heatmap()` left-most column |
| Get top sample index safely | `get_top_sample()` |

### Key Principles

1. **Features are always explanations**: All plots show feature attributions (why samples ARE significant)
2. **Global feature ordering**: In heatmaps, features ordered by average importance for interpretability
3. **Flexible sample ordering**: Order by actual (what IS) or formative (what CREATES) based on question
4. **Local vs global**: Profile plots can show individual (local) or typical (global) importance patterns
5. **Complementary views**: Use all three plots together for complete understanding
6. **Tabular only**: Visualization functions require a tabular fit; text/graph fits are not supported

### The Workflow

1. **Explore** with `significance_plot()` → identify patterns
2. **Understand** with `heatmap()` → see feature patterns
3. **Explain** with `profile_plot()` → understand individuals

This three-step workflow enables discovery, interpretation, and communication of instance significance in your scientific datasets.
