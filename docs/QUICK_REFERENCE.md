# DataTypical Quick Reference

**Version 0.7.6** | Daily reference for exploring instance significance

---

## 30-Second Start
```python
from datatypical import DataTypical
from datatypical_viz import significance_plot, heatmap, profile_plot

# Basic exploration (fast)
dt = DataTypical()
results = dt.fit_transform(data)

# With explanations (moderate speed)
dt = DataTypical(shapley_mode=True)
results = dt.fit_transform(data)

# Visualize
significance_plot(results, significance='archetypal')
heatmap(dt, results, significance='archetypal', top_n=20)
```

---

## Three Questions DataTypical Answers

| Question | Significance Type | What It Finds |
|----------|------------------|---------------|
| **Which are extreme?** | Archetypal | Boundary samples, outliers, corners |
| **Which are representative?** | Prototypical | Central samples, typical examples |
| **Which match my target?** | Stereotypical | Samples similar to target pattern |

**All three work simultaneously** - you get all perspectives in one analysis.

---

## Results DataFrame

### Without Shapley (fast)
```python
results.columns
# ['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']
```

### With Shapley (adds explanations + formative)
```python
results.columns
# ['archetypal_rank', 'prototypical_rank', 'stereotypical_rank',
#  'archetypal_shapley_rank', 'prototypical_shapley_rank', 'stereotypical_shapley_rank']
```

**Ranks are 0-1 normalized**: 
- 1.0 = most significant
- 0.0 = least significant

---

## Dual Perspective (Shapley Mode)

### Actual Ranks (`*_rank`)
**What samples ARE significant**
- Use for: Selection, prioritization
- Example: Top 10 archetypal samples

### Formative Ranks (`*_shapley_rank`)
**What samples CREATE the structure**
- Use for: Understanding, quality control
- Example: Which samples define boundaries?

### The Key Insight
High actual + Low formative = Replaceable (many similar samples)  
Low actual + High formative = Gap filler (removes coverage holes)

---

## Key Parameters

### Speed vs Insight
```python
# Fast exploration (seconds)
dt = DataTypical(fast_mode=True)

# Publication quality (minutes)
dt = DataTypical(fast_mode=False)

# With explanations (adds time)
dt = DataTypical(shapley_mode=True)
```

**Rules of thumb**:
- 1,000 samples: ~5 sec (no Shapley) → ~5 min (with Shapley)
- 10,000 samples: ~30 sec (no Shapley) → ~60 min (with Shapley)

### Essential Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `shapley_mode` | `False` | Enable explanations & formative analysis |
| `fast_mode` | `True` | Speed vs accuracy trade-off |
| `n_archetypes` | `8` | Number of extreme "corners" to find |
| `n_prototypes` | `8` | Number of representative samples |
| `stereotype_column` | `None` | Column name for target pattern |
| `selected_significance` | `None` | Compute only one type: `'archetypal'`, `'prototypical'`, or `'stereotypical'` (None = all three) |
| `shapley_top_n` | `500` | Limit explanations to top N samples |

### For Stereotypical Analysis
```python
# Must specify target column
dt = DataTypical(
    shapley_mode=True,
    stereotype_column='activity'  # Your target property
)
```

---

## Visualization Quick Reference

### 1. Significance Plot (Overview)
```python
significance_plot(results, significance='archetypal')
```
**Shows**: Actual vs Formative scatter  
**Use for**: Identifying critical/replaceable/gap-filling samples

### 2. Heatmap (Feature Patterns)
```python
heatmap(dt, results, 
        significance='archetypal',
        order='actual',  # or 'formative'
        top_n=20)
```
**Shows**: Which features make samples significant  
**Use for**: Understanding feature importance patterns

### 3. Profile Plot (Individual Sample)
```python
top_idx = results['archetypal_rank'].idxmax()
profile_plot(dt, top_idx, 
             significance='archetypal',
             order='local')  # or 'global'
```
**Shows**: Feature contributions for one sample  
**Use for**: Deep-dive into specific samples

---

## Common Workflows

### Workflow 1: Quick Exploration (No Shapley)
```python
# Fit and get ranks
dt = DataTypical()
results = dt.fit_transform(data)

# Find top samples
top_archetypes = results.nlargest(10, 'archetypal_rank')
top_prototypes = results.nlargest(10, 'prototypical_rank')
```
**Time**: Seconds  
**When**: Initial exploration, large datasets, iteration

### Workflow 2: Understand Why (With Shapley)
```python
# Enable explanations
dt = DataTypical(shapley_mode=True, fast_mode=False)
results = dt.fit_transform(data)

# Visualize
significance_plot(results, significance='archetypal')
heatmap(dt, results, significance='archetypal', top_n=15)

# Deep dive
top_idx = results['archetypal_rank'].idxmax()
profile_plot(dt, top_idx, significance='archetypal')
```
**Time**: Minutes  
**When**: Understanding mechanisms, publication figures

### Workflow 3: Dataset Curation
```python
# Find redundant samples
dt = DataTypical(shapley_mode=True)
results = dt.fit_transform(data)

# High actual, low formative = replaceable
redundant = results[
    (results['archetypal_rank'] > 0.7) &
    (results['archetypal_shapley_rank'] < 0.3)
]

# Keep one per cluster
from sklearn.cluster import KMeans
# ... cluster and keep representatives
```
**When**: Building curated subsets, reducing dataset size

---

## Interpreting Ranks

### Archetypal Rank (Extremeness)
- **High (>0.8)**: Boundary samples, unusual combinations
- **Medium (0.4-0.8)**: Moderately extreme
- **Low (<0.4)**: Central, typical samples

### Prototypical Rank (Representativeness)
- **High (>0.8)**: Very representative, close to many samples
- **Medium (0.4-0.8)**: Somewhat representative
- **Low (<0.4)**: Outliers, dissimilar to others

### Stereotypical Rank (Target Similarity)
- **High (>0.8)**: Very similar to target pattern
- **Medium (0.4-0.8)**: Moderately similar
- **Low (<0.4)**: Dissimilar to target

---

## Feature Ordering: Local vs Global

### In Profile Plots

**`order='local'`** (default):
- Orders by THIS sample's Shapley values
- Different for each sample
- Answers: "What makes THIS sample significant?"

**`order='global'`**:
- Orders by average importance across ALL samples
- Same order for all samples
- Answers: "How does this compare to typical patterns?"

### In Heatmaps

**Always global ordering**:
- Features ordered by average importance
- Consistent, interpretable columns
- X-axis shows features by global importance

---

## Troubleshooting

### "No explanations available"
**Problem**: Requesting samples without computed explanations  
**Solution**: Increase `shapley_top_n` or use `fast_mode=False`

### Zero Shapley values in heatmap
**Problem**: Formative sample not in top-N for actual significance  
**Solution**: This is expected! Shows distinction between formative and actual

### Very slow with Shapley
**Problem**: Too many samples or permutations  
**Solutions**:
- Use `fast_mode=True` (fewer permutations)
- Set `shapley_top_n=500` (limit to top samples)
- Reduce `shapley_n_permutations=30` (less accurate but faster)

### Stereotypical all zeros
**Problem**: Forgot to set `stereotype_column`  
**Solution**: `DataTypical(stereotype_column='your_column')`

### Feature dimension mismatch
**Problem**: Transform data has different features than fit data  
**Solution**: Ensure same features in same order for fit and transform

---

## Performance Guidelines

### Dataset Size Recommendations

| Samples | Shapley | fast_mode | shapley_top_n | Expected Time |
|---------|---------|-----------|---------------|---------------|
| < 500  | Yes | False | None | ~30 sec |
| 500-2,000 | Yes | True | 500 | ~2 min |
| 2,000-5,000 | Yes | True | 500 | ~10 min |
| 5,000-10,000 | Yes | True | 500 | ~30 min |
| > 10,000 | No | True | - | ~1 min |

### Optimization Strategy

**Phase 1: Exploration** (fast_mode=True, no Shapley)
- Find interesting samples quickly
- Iterate on parameters
- Screen large datasets

**Phase 2: Understanding** (fast_mode=False, shapley_mode=True)
- Subset to interesting samples
- Generate explanations
- Create publication figures

---

## Data Types Supported

### Tabular Data (default)
```python
dt = DataTypical()
results = dt.fit_transform(dataframe)
```

### Text Data (auto-detected)
```python
texts = ["sample 1 text", "sample 2 text", ...]
dt = DataTypical()
results = dt.fit_transform(texts)
```

### Graph Data (requires edges parameter)
```python
dt = DataTypical()
results = dt.fit_transform(node_features, edges=edge_list)
```

---

## Getting Explanations

### For One Sample
```python
explanations = dt.get_shapley_explanations(sample_idx)

# Returns dict with three keys
explanations['archetypal']     # Feature contributions to archetypal
explanations['prototypical']   # Feature contributions to prototypical  
explanations['stereotypical']  # Feature contributions to stereotypical
```

### For Formative Attribution
```python
attributions = dt.get_formative_attributions(sample_idx)

# Returns dict with three keys (sample-level, not feature-level)
attributions['archetypal']     # How sample creates archetypal structure
attributions['prototypical']   # How sample creates prototypical structure
attributions['stereotypical']  # How sample creates stereotypical structure
```

---

## Memory Management

### For Large Datasets
```python
dt = DataTypical(
    shapley_mode=True,
    fast_mode=True,
    shapley_top_n=500,        # Limit explanations
    max_memory_mb=8000        # Limit memory usage
)
```

### Cleanup After Transform
DataTypical automatically cleans up large intermediate arrays during transform to minimize memory usage.

---

## Reproducibility

### Set Random Seed
```python
dt = DataTypical(random_state=42)
```

**Deterministic operations**:
- Archetypal analysis (PCHA)
- Prototypical selection (facility location)
- Stereotypical ranking

**Stochastic operations** (require random_state):
- NMF initialization (when fast_mode=True)
- Shapley permutations

---

## Quick Tips

✓ **Start without Shapley** - Get familiar with ranks first  
✓ **Use fast_mode=True** for iteration - Switch to False for finals  
✓ **Set shapley_top_n** for large data - Limits computational cost  
✓ **Visualize all three** - Different perspectives reveal different patterns  
✓ **Check formative ranks** - Reveals structural importance  
✓ **Profile interesting samples** - Understand individual mechanisms  
✓ **Compare local vs global** - See individual vs typical patterns  

✗ **Don't forget stereotype_column** - Required for stereotypical analysis  
✗ **Don't expect instant results** - Shapley takes time but worth it  
✗ **Don't ignore warnings** - They guide optimization  
✗ **Don't mix feature sets** - fit and transform must match  

---

## Next Steps

- **README.md**: Overview and installation
- **EXAMPLES.md**: Complete worked examples
- **VISUALIZATION_GUIDE.md**: Detailed visualization documentation
- **INTERPRETATION_GUIDE.md**: Deep dive into formative instances, and how to interpret results
- **COMPUTATION_GUIDE.md**: Details on implementation for advanced users
