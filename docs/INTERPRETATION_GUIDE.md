# DataTypical v0.7.6 - Interpretation Guide

## Table of Contents

1. [Understanding Instance Significance](#understanding-instance-significance)
2. [The Three Significance Types](#the-three-significance-types)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Shapley Value Explanations](#shapley-value-explanations)
5. [Formative Instance Discovery](#formative-instance-discovery)
6. [Dual-Perspective Analysis](#dual-perspective-analysis)
7. [Interpreting Results](#interpreting-results)
8. [Visualization Guide](#visualization-guide)
9. [Decision Making Framework](#decision-making-framework)
10. [Assumptions and Limitations](#assumptions-and-limitations)
11. [Use Cases and Examples](#use-cases-and-examples)

---

## Understanding Instance Significance

### The Fundamental Question

When analyzing a dataset, we often ask: **"Which instances are most important?"**

This seemingly simple question has no single answer. Importance depends on perspective, context, and purpose. DataTypical provides three distinct but complementary answers, each capturing a different notion of significance:

| Significance Type | Philosophical Basis | Question Answered |
|-------------------|---------------------|-------------------|
| **Archetypal** | Objective | Which instances occupy extreme positions? |
| **Prototypical** | Representative | Which instances best represent the population? |
| **Stereotypical** | Subjective | Which instances match my target criterion? |

These are not competing definitions but complementary perspectives that together provide a complete picture of dataset structure.

### Why Multiple Perspectives Matter

Consider a drug discovery dataset with molecular descriptors and activity measurements:

**Archetypal compounds** have extreme combinations of properties:
- Example: Highest lipophilicity combined with lowest polarity
- Value: Identify boundary cases, understand limits of chemical space
- Use: Novelty detection, coverage testing, outlier characterization

**Prototypical compounds** are representative of the dataset:
- Example: Average across all properties, common structural motifs
- Value: Select diverse panel for screening, understand typical behavior
- Use: Dataset summarization, active learning, library curation

**Stereotypical compounds** match a specific target:
- Example: Lowest IC50 (most potent)
- Value: Direct optimization, find best candidates
- Use: Hit identification, constraint satisfaction, target-based selection

**All three are "significant" but for different reasons.** The choice depends on your scientific question, not on which metric produces the highest scores.

### The v0.6+ Innovation: Dual Perspectives

Traditional significance metrics tell you WHERE samples are positioned in feature space. DataTypical v0.6+ adds a second dimension: which samples CREATE the significance structure itself.

This distinction between "what samples ARE" versus "what samples CREATE" is the core innovation enabling formative instance discovery.

---

## The Three Significance Types

### 1. Archetypal Significance (Objective)

**Definition**: Instances that occupy extreme positions in feature space, representing the "corners" or boundary cases of the data distribution.

**Conceptual Basis**: Archetypal significance is objective because it is determined entirely by the geometric structure of the data, independent of user preferences or domain knowledge. A sample is archetypal because of its position relative to other samples, not because of any external criterion.

**Geometric Interpretation**:

Imagine plotting all samples in n-dimensional feature space:
- The data forms a cloud or distribution
- Some samples sit at the periphery, at the vertices of the convex hull
- These boundary samples represent the most extreme combinations of features
- All other samples can be expressed as convex combinations of these extremes

**What High Archetypal Rank Means**:

A sample with high archetypal rank:
- Occupies an extreme position compared to the population
- Combines features in unusual ways
- Lies far from the center of the distribution
- Can be expressed with high weight on a single basis vector
- May be an outlier, but structured (not random noise)

**What Low Archetypal Rank Means**:

A sample with low archetypal rank:
- Is central or average across features
- Well-explained by combinations of multiple basis vectors
- Has no single extreme characteristic
- Is typical of the bulk population

**When to Use Archetypal Analysis**:

| Use Case | Rationale |
|----------|-----------|
| Boundary exploration | Understanding limits of feature space |
| Novelty detection | Finding samples with unusual properties |
| Coverage testing | Ensuring test sets span the full range |
| Outlier characterization | Distinguishing structured extremes from noise |
| Diversity assessment | Identifying samples that extend the distribution |

**Example Applications**:

- Materials science: Compounds with extreme hardness combined with low density
- Drug discovery: Molecules with unusual property combinations
- Quality control: Products at specification limits
- Ecology: Species with extreme environmental tolerances
- Finance: Portfolios with extreme risk/return profiles

---

### 2. Prototypical Significance (Representative)

**Definition**: Instances that best represent the dataset, collectively providing maximum coverage with minimum redundancy.

**Conceptual Basis**: Prototypical significance is representative because these instances capture the essential characteristics of the population. They are the "best examples" that someone unfamiliar with the dataset should see first. Unlike archetypes (which are extreme), prototypes are typical.

**Algorithmic Interpretation**:

The prototype selection problem is formulated as facility location optimization:
- Given n samples, select k "facilities" (prototypes)
- Each sample is "covered" by its nearest prototype
- Objective: maximize total coverage (sum of similarities to nearest prototypes)
- Constraint: no more than k prototypes

The key insight is that prototypes collectively minimize redundancy while maximizing representation.

**What High Prototypical Rank Means**:

A sample with high prototypical rank:
- Is representative of a cluster or region of the data
- Has high similarity to many other samples
- Would be "missed" if removed (coverage gap)
- Is typical of some subpopulation
- Was selected early in the greedy selection process

**What Low Prototypical Rank Means**:

A sample with low prototypical rank:
- Is redundant (well-represented by existing prototypes)
- Has high similarity to already-selected prototypes
- Can be easily reconstructed from neighboring samples
- Adds little new information to the prototype set

**When to Use Prototypical Analysis**:

| Use Case | Rationale |
|----------|-----------|
| Dataset summarization | Select k representatives from n samples |
| Active learning | Choose informative examples for labeling |
| Library curation | Build diverse but representative collections |
| Teaching sets | Show typical examples to learners |
| Cluster representatives | One prototype per cluster |
| Stratified sampling | Ensure all regions are represented |

**Example Applications**:

- Drug discovery: Diverse compound libraries for high-throughput screening
- Image datasets: Representative images for human review or annotation
- Customer segmentation: Archetypal customers per segment for personas
- Species selection: Representative samples for genetic or ecological studies
- Model distillation: Training examples that capture the full distribution

---

### 3. Stereotypical Significance (Subjective)

**Definition**: Instances that most closely match a user-specified target criterion, representing the "ideal" according to a defined objective.

**Conceptual Basis**: Stereotypical significance is subjective because it depends entirely on user-defined goals. The same sample may be highly stereotypical for one target but not another. This makes stereotypical analysis fundamentally different from archetypal and prototypical analysis, which are intrinsic to the data.

**Target Types**:

DataTypical supports three target specifications:

| Target | Interpretation | Score Computation |
|--------|----------------|-------------------|
| `'max'` | Maximize the feature | Distance above median |
| `'min'` | Minimize the feature | Distance below median |
| `float` | Match specific value | Negative absolute distance |

**What High Stereotypical Rank Means**:

A sample with high stereotypical rank:
- Closely matches the specified target
- Is extreme in the desired direction
- Represents the "best" according to the criterion
- Would be prioritized in optimization workflows

**What Low Stereotypical Rank Means**:

A sample with low stereotypical rank:
- Is far from the target value
- May be extreme in the opposite direction
- Represents the "worst" according to the criterion

**When to Use Stereotypical Analysis**:

| Use Case | Rationale |
|----------|-----------|
| Optimization | Find best instances for a specific goal |
| Hit identification | Compounds meeting activity threshold |
| Target-based selection | Samples closest to a reference profile |
| Constraint satisfaction | Instances meeting specifications |
| Outlier identification | Samples far from expected values |

**Example Applications**:

- Drug discovery: Lowest IC50 (stereotypical potent compounds)
- Materials science: Highest conductivity (stereotypical conductors)
- Manufacturing: Closest to target dimensions (quality control)
- Agriculture: Highest yield varieties
- Healthcare: Patients closest to treatment response profile

---

## Mathematical Foundations

### Archetypal Analysis via NMF

**Non-negative Matrix Factorization (NMF)** approximates the data matrix X as a product of two non-negative matrices:

```
X approx W * H
```

where:
- X is the (n_samples, n_features) data matrix
- W is the (n_samples, k) coefficient matrix
- H is the (k, n_features) basis matrix
- k is the number of archetypes (nmf_rank parameter)

**Objective Function**:

NMF minimizes the Frobenius norm of the reconstruction error:

```
min_{W,H >= 0} (1/2) ||X - WH||_F^2
```

**Interpretation**:

Each row of H represents an "archetype" in feature space. Each row of W represents the coefficients expressing that sample as a combination of archetypes. Samples with high weight on a single archetype (sparse W row) are archetypal.

**Archetypal Score Computation**:

```
1. Normalize W rows: W_norm = W / sum(W, axis=1)
2. Maximum coefficient: arch_wmax = max(W_norm, axis=1)
3. Corner score: distance to nearest hypercube corner
4. Combined: archetypal_score = 0.7 * arch_wmax + 0.3 * corner_score
```

**Alternative: PCHA (Principal Convex Hull Analysis)**:

In publication mode (`fast_mode=False`), DataTypical uses PCHA when available:

```
X approx X * C * S
```

where:
- C: Convex combination matrix (archetypes are combinations of data points)
- S: Membership matrix (data points are combinations of archetypes)

This ensures archetypes lie within the convex hull of the data.

**Method Selection in v0.7**:

| Mode | Method | Fallback Chain |
|------|--------|----------------|
| `fast_mode=True` | NMF | None (NMF only) |
| `fast_mode=False` | PCHA | ConvexHull (if n_features <= 20) then NMF |

### Prototypical Selection via Facility Location

**Facility Location Problem**:

Given n samples with pairwise similarities s(i,j), select k prototypes P to maximize coverage:

```
max_{|P| <= k} sum_{i=1}^{n} w_i * max_{p in P} s(i, p)
```

where w_i are optional sample weights.

**Submodular Maximization**:

The coverage function is submodular, meaning adding a prototype yields diminishing returns. This property enables efficient greedy optimization with guaranteed approximation ratio.

**Approximation Guarantee**:

The greedy CELF algorithm achieves:

```
coverage_greedy >= (1 - 1/e) * coverage_optimal approx 0.632 * coverage_optimal
```

**Similarity Metric**:

DataTypical uses cosine similarity on L2-normalized features:

```
s(i, j) = (x_i dot x_j) / (||x_i|| * ||x_j||)
```

**Prototypical Score Computation**:

```
1. Euclidean distance to nearest prototype: d_euc
2. Cosine similarity to nearest prototype: cos_sim
3. Normalize distance: d_norm = clip(d_euc / percentile_95, 0, 1)
4. Combined: prototypical_score = 0.5 * (1 - d_norm) + 0.5 * cos_sim
```

### Stereotypical Scoring

**Distance-Based Targeting**:

For a target column with values v and specified target t:

**Maximize** (`stereotype_target='max'`):
```
score_i = max(0, v_i - median(v))
```

**Minimize** (`stereotype_target='min'`):
```
score_i = max(0, median(v) - v_i)
```

**Match Value** (`stereotype_target=float`):
```
score_i = -|v_i - t|
```

All scores are normalized to [0, 1] range.

---

## Shapley Value Explanations

### The Explanation Problem

After identifying that a sample is significant (high archetypal/prototypical/stereotypical rank), the natural question is: **"WHY is this sample significant?"**

Shapley values provide a principled answer by attributing the significance score to individual features.

### Shapley Value Theory

**Game-Theoretic Foundation**:

Shapley values originate from cooperative game theory. For a game with n players and value function v, the Shapley value of player i is:

```
phi_i(v) = sum_{S subset N\{i}} [|S|!(n-|S|-1)!/n!] * [v(S union {i}) - v(S)]
```

This formula computes the average marginal contribution of player i across all possible orderings.

**Axiomatic Properties**:

Shapley values are the unique allocation satisfying:

1. **Efficiency**: Sum of all phi_i equals v(N) - v(empty set)
   (Total attributions sum to total value)

2. **Symmetry**: If two features contribute equally, they receive equal attribution

3. **Null Player**: Non-contributors receive zero attribution

4. **Linearity**: Attributions add across games

### Permutation-Based Approximation

Exact Shapley computation requires O(2^n) value function evaluations. DataTypical uses permutation sampling for efficient approximation:

```
For each permutation pi:
    For each feature f in pi:
        S = features before f in pi
        marginal[f] += v(S union {f}) - v(S)

phi[f] = marginal[f] / n_permutations
```

**Convergence**: By the Central Limit Theorem, approximation error decreases as O(1/sqrt(M)) where M is the number of permutations.

**Early Stopping**: DataTypical monitors convergence and can terminate early when Shapley values stabilize (controlled by `shapley_early_stopping_patience` and `shapley_early_stopping_tolerance`).

### Value Functions for Explanations

**Archetypal Explanations**:

Measures how much each feature contributes to the sample's extremeness:

```python
def explain_archetypal_features(X_subset, indices, context):
    # Distance to boundary (0 or 1) for each feature
    dist_to_boundary = np.minimum(X_subset, 1.0 - X_subset)
    # Closer to boundary = more archetypal
    archetypal_contribution = np.mean(1.0 - 2.0 * dist_to_boundary, axis=1)
    return np.mean(archetypal_contribution)
```

**Interpretation**: Features with values near 0 or 1 (extremes) contribute positively. Features with values near 0.5 (central) contribute negatively.

**Prototypical Explanations**:

Measures how much each feature contributes to representativeness:

```python
def explain_prototypical_features(X_subset, indices, context):
    # Higher variance = more representative of data spread
    return np.mean(np.var(X_subset, axis=1))
```

**Interpretation**: Features where the sample has values consistent with the population variance contribute positively.

**Stereotypical Explanations**:

Measures how much each feature contributes to matching the target:

```python
def explain_stereotypical_features(X_subset, indices, context):
    target_value = context['target_values'][sample_idx]
    distance = compute_target_distance(target_value, context)
    feature_contrib = np.mean(np.abs(X_subset))
    return distance * feature_contrib
```

**Interpretation**: Features that align the sample toward the target contribute positively. Features that pull away contribute negatively.

### Interpreting Shapley Explanations

**Positive Shapley Value**:
- Feature INCREASES the sample's significance
- For archetypal: Feature is extreme (near 0 or 1)
- For prototypical: Feature makes sample representative
- For stereotypical: Feature pushes toward target

**Negative Shapley Value**:
- Feature DECREASES the sample's significance
- For archetypal: Feature is central (near 0.5)
- For prototypical: Feature makes sample atypical
- For stereotypical: Feature pulls away from target

**Zero/Small Shapley Value**:
- Feature has minimal impact on this sample's significance
- Other features dominate the score

**Magnitude**:
- Large |phi|: Feature is a dominant contributor
- Small |phi|: Feature has minor influence

---

## Formative Instance Discovery

### The Discovery

When `shapley_mode=True`, DataTypical doesn't just explain WHY instances are significant---it discovers a fundamentally new type of instance:

**Formative instances**: Samples that CREATE the significance structure of your dataset.

This is the core innovation in v0.6+, distinguishing between:
- **Actual significance**: Samples that ARE archetypal/prototypical/stereotypical
- **Formative significance**: Samples that CREATE the archetypal/prototypical/stereotypical structure

### The Core Distinction

| Perspective | Question | Measurement | Meaning |
|-------------|----------|-------------|---------|
| **Actual** | Which samples ARE significant? | Direct evaluation (scores) | Positional property |
| **Formative** | Which samples CREATE structure? | Shapley values (contributions) | Causal property |

**Example**:

A compound with very high logP and very low TPSA may be:
- **High actual archetypal**: It occupies an extreme position
- **Low formative archetypal**: Many similar extremes exist; removing it wouldn't change boundaries

Conversely, a moderately extreme compound might be:
- **Moderate actual archetypal**: Not the most extreme
- **High formative archetypal**: Unique boundary-defining position; removal would shrink the convex hull

### Why Formative Matters

**The Problem**:

Traditional significance metrics tell you WHERE samples are, not how IMPORTANT they are to the structure.

Scenario: You find 50 highly archetypal (extreme) compounds.
- Are they 50 unique boundary-defining samples? Keep all.
- Are they 10 clusters of similar extremes? Keep one per cluster.

**You cannot distinguish these cases from archetypal rank alone.**

**The Solution**:

Formative ranks reveal structural importance:

```python
from datatypical_viz import significance_plot

# Compare perspectives
significance_plot(
    results,
    significance='archetypal'  # Automatically plots actual vs formative
)
```

The plot reveals:
- **Critical**: High both -> unique boundary definers
- **Replaceable extremes**: High actual, low formative -> cluster of similar extremes
- **Structure creators**: Low actual, high formative -> fill important gaps
- **Redundant**: Low both -> safe to remove

### Formative Value Functions

**Archetypal Formative (Convex Hull)**:

Measures how much each sample expands the convex hull volume:

```python
def formative_archetypal_convex_hull(X_subset, indices, context):
    if n_features > 20:
        # Safe fallback for high dimensions
        ranges = X_subset.max(axis=0) - X_subset.min(axis=0)
        return np.prod(ranges + 1e-10)
    else:
        hull = ConvexHull(X_subset)
        return hull.volume
```

**Interpretation**: Samples on the convex hull boundary have high formative values. Samples inside the hull have low formative values.

**Prototypical Formative (Coverage)**:

Measures how much each sample contributes to coverage:

```python
def formative_prototypical_coverage(X_subset, indices, context):
    X_l2 = X_subset / np.linalg.norm(X_subset, axis=1, keepdims=True)
    similarities = X_l2 @ X_l2.T
    np.fill_diagonal(similarities, 0)
    max_sims = np.max(similarities, axis=1)
    return np.mean(max_sims)
```

**Interpretation**: Samples that fill coverage gaps have high formative values. Samples with many similar neighbors have low formative values.

**Stereotypical Formative (Extremeness)**:

Measures how much each sample contributes to the target distribution:

```python
def formative_stereotypical_extremeness(X_subset, indices, context):
    subset_vals = context['target_values'][indices]
    
    if target == 'max':
        return np.mean(np.maximum(subset_vals - median, 0))
    elif target == 'min':
        return np.mean(np.maximum(median - subset_vals, 0))
    else:
        return median_dist - np.mean(np.abs(subset_vals - target))
```

**Interpretation**: Samples that pull the distribution toward the target have high formative values.

### The Four-Quadrant Discovery

When you plot actual vs formative significance, instances cluster into four regions:

```
         Formative
            High
             |
    Gap      | Critical
   Fillers   | Instances
             |
    ---------+---------  Actual
             |
  Redundant  | Replaceable
  Instances  | Extremes
             |
            Low
```

**Quadrant 1: Critical Instances (Top-Right)**

- High actual, High formative
- Extreme/representative AND structure-defining
- **Discovery**: These are irreplaceable instances
- **Example**: Unique boundary vertex with no similar samples
- **Action**: Always keep, highest priority

**Quadrant 2: Replaceable Extremes (Bottom-Right)**

- High actual, Low formative
- Extreme/representative but NOT unique
- **Discovery**: These cluster together; pick one per cluster
- **Example**: One of 10 samples with similar extreme profiles
- **Action**: Select one representative, others are redundant

**Quadrant 3: Gap Fillers (Top-Left)**

- Low actual, High formative
- Not extreme/representative but structurally important
- **Discovery**: Hidden important instances often overlooked
- **Example**: Sample connecting regions or preventing coverage holes
- **Action**: Keep for completeness, underappreciated samples

**Quadrant 4: Redundant Instances (Bottom-Left)**

- Low actual, Low formative
- Neither extreme nor structure-defining
- **Discovery**: Safe to remove
- **Example**: Average samples with many similar neighbors
- **Action**: Low priority, can be excluded

### Practical Discoveries

**Discovery 1: Hidden Critical Instances**

```python
# Find high-formative, low-actual (gap fillers)
gap_fillers = results[
    (results['prototypical_rank'] < 0.5) &
    (results['prototypical_shapley_rank'] > 0.8)
]
print(f"Found {len(gap_fillers)} hidden structure creators")
```

These are instances you would have ignored (low actual rank) that are actually critical for complete coverage.

**Discovery 2: Redundant Extremes**

```python
# Find high-actual, low-formative (replaceable)
replaceable = results[
    (results['archetypal_rank'] > 0.8) &
    (results['archetypal_shapley_rank'] < 0.3)
]
print(f"Found {len(replaceable)} replaceable extremes")

# Cluster them to keep one per cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(replaceable[feature_columns])

# Keep one representative per cluster
keepers = replaceable.groupby(clusters).apply(
    lambda x: x.nlargest(1, 'archetypal_rank')
)
```

You thought you had 50 unique extremes; you actually have 5 clusters. Keep 5, not 50.

**Discovery 3: Alternative Mechanisms**

```python
from datatypical_viz import profile_plot

# Compare high-formative stereotypical instances
unique_performers = results.nlargest(10, 'stereotypical_shapley_rank')

for idx in unique_performers.index:
    profile_plot(dt, idx, significance='stereotypical')
```

Different pathways to the same outcome. Each high-formative sample may use different features to achieve the target.

---

## Dual-Perspective Analysis

### Combining Actual and Formative

The power of DataTypical lies in combining both perspectives:

| Actual High, Formative High | Actual High, Formative Low |
|----------------------------|---------------------------|
| **"Critical"** | **"Exemplars"** |
| Extreme AND structure-creating | Extreme but replaceable |
| Example: Boundary vertex | Example: One of many extremes |
| Keep always | Optional, similar samples exist |

| Actual Low, Formative High | Actual Low, Formative Low |
|---------------------------|--------------------------|
| **"Structure Creators"** | **"Redundant"** |
| Central but necessary | Neither extreme nor necessary |
| Example: Gap-filling keystone | Example: Average with neighbors |
| Keep for completeness | Safe to remove |

### Practical Example: Drug Discovery

Consider a dataset of 1000 compounds:

**Compound A**:
- Actual archetypal rank: 0.98 (extremely high)
- Formative archetypal rank: 0.12 (low)
- **Interpretation**: Very extreme combination of properties, BUT many other compounds are similarly extreme. Removing A wouldn't change the boundaries much.
- **Action**: Interesting but not essential.

**Compound B**:
- Actual archetypal rank: 0.87 (high)
- Formative archetypal rank: 0.94 (very high)
- **Interpretation**: Extreme properties AND unique boundary-defining position. Removing B would shrink the convex hull significantly.
- **Action**: Critical for understanding chemical space.

**Compound C**:
- Actual archetypal rank: 0.23 (low)
- Formative archetypal rank: 0.88 (high)
- **Interpretation**: Not particularly extreme itself, BUT occupies a key position that fills a gap. Removing C would create a hole in coverage.
- **Action**: Important for completeness.

### Correlation Patterns

**Strong Positive Correlation** (actual ~ formative):
- Extreme samples ARE the structure-creators
- Clean, well-separated boundaries
- Example: Well-defined material classes

**Weak/No Correlation**:
- Extremeness and structural importance are independent
- Complex, overlapping distributions
- Example: Continuous chemical space with no clear boundaries

**Points Far From Diagonal**:
- High actual, low formative: Many similar extreme samples (redundancy)
- Low actual, high formative: Hidden structural keystones (gap fillers)

---

## Interpreting Results

### Output Columns

DataTypical returns a DataFrame with the following columns:

**Actual Significance Ranks** (always computed):

| Column | Range | Interpretation |
|--------|-------|----------------|
| `archetypal_rank` | [0, 1] | Higher = more extreme position |
| `prototypical_rank` | [0, 1] | Higher = more representative |
| `stereotypical_rank` | [0, 1] | Higher = closer to target |

**Formative Significance Ranks** (when `shapley_mode=True`, `fast_mode=False`):

| Column | Range | Interpretation |
|--------|-------|----------------|
| `archetypal_shapley_rank` | [0, 1] | Higher = more boundary-defining |
| `prototypical_shapley_rank` | [0, 1] | Higher = more coverage-creating |
| `stereotypical_shapley_rank` | [0, 1] | Higher = more target-distribution-defining |

**Note**: With `fast_mode=True`, formative columns contain `None` values (formative computation skipped for speed).

### Accessing Explanations

```python
# Get feature-level explanations for a specific sample
explanations = dt.get_shapley_explanations(sample_idx)

# Returns dict with keys: 'archetypal', 'prototypical', 'stereotypical'
# Each value is a numpy array of shape (n_features,)

arch_shap = explanations['archetypal']
print(f"Most important features for archetypal significance:")
top_idx = np.argsort(np.abs(arch_shap))[::-1][:5]
for i in top_idx:
    print(f"  {feature_names[i]}: {arch_shap[i]:.4f}")
```

### Accessing Formative Attributions

```python
# Get formative attributions (requires fast_mode=False)
try:
    attributions = dt.get_formative_attributions(sample_idx)
    # Returns dict with keys: 'archetypal', 'prototypical', 'stereotypical'
except RuntimeError as e:
    print("Formative data not available (fast_mode=True)")
```

### Interpreting Rank Values

**High Rank (> 0.8)**:
- Sample is in the top 20% for this significance type
- Strong candidate for selection/prioritization
- Warrants detailed examination via profile plots

**Moderate Rank (0.4 - 0.8)**:
- Sample has some significance but not exceptional
- May be included depending on selection criteria
- Useful for understanding typical behavior

**Low Rank (< 0.4)**:
- Sample has minimal significance for this type
- Likely redundant or average
- May be excluded unless formative rank is high

### Rank Correlations

Examine correlations between rank types:

```python
import seaborn as sns

rank_cols = ['archetypal_rank', 'prototypical_rank', 'stereotypical_rank']
corr = results[rank_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
```

**Typical Patterns**:

- Archetypal vs Prototypical: Usually negative (extremes vs central)
- Archetypal vs Stereotypical: Varies (depends on target)
- Prototypical vs Stereotypical: Often positive (representative samples meet targets)

---

## Visualization Guide

DataTypical provides three core visualizations through `datatypical_viz.py`:

### 1. Significance Plot (Dual-Perspective Scatter)

**Purpose**: Reveal the relationship between actual and formative significance.

**Usage**:

```python
from datatypical_viz import significance_plot

# Basic plot
ax = significance_plot(results, significance='archetypal')

# With color coding
ax = significance_plot(
    results,
    significance='prototypical',
    color_by='target_property',  # Auto-detects discrete vs continuous
    label_top=5
)
```

**Elements**:
- X-axis: Actual rank (0-1)
- Y-axis: Formative rank (0-1)
- Quadrant lines: Default at (0.5, 0.5)
- Color: Optional property for additional insight

**Color Mode Detection**:

| Unique Values | Treatment |
|---------------|-----------|
| 2 | Binary: purple (low), green (high) |
| 3-5 | Discrete: viridis palette with legend |
| 6-12 | Discrete: different colors and markers |
| > 12 | Continuous: viridis colormap with colorbar |

**Interpretation**:

- **Top-right**: Critical instances (keep always)
- **Bottom-right**: Replaceable extremes (deduplicate)
- **Top-left**: Gap fillers (hidden importance)
- **Bottom-left**: Redundant (safe to remove)

### 2. Heatmap (Feature Attribution Matrix)

**Purpose**: Show which features contribute to significance across multiple samples.

**Usage**:

```python
from datatypical_viz import heatmap

# Top samples by actual rank
ax = heatmap(
    dt, results,
    significance='stereotypical',
    order='actual',  # Order rows by actual rank
    top_n=25
)

# Top samples by formative rank
ax = heatmap(
    dt, results,
    significance='archetypal',
    order='formative',  # Order rows by formative rank
    top_n=20
)
```

**Elements**:
- Rows: Samples (ordered by actual or formative rank)
- Columns: Features (ordered by global importance)
- Color: Shapley value (contribution to significance)

**Interpretation**:

- **Bright (positive)**: Feature increases significance
- **Dark (negative)**: Feature decreases significance
- **Neutral**: Feature has minimal impact

**Pattern Recognition**:

- **Vertical clusters**: Groups of samples with similar feature patterns
- **Horizontal bands**: Features that consistently matter
- **Mixed patterns**: Multiple mechanisms achieving significance

### 3. Profile Plot (Individual Feature Importance)

**Purpose**: Detailed feature-level analysis for a single sample.

**Usage**:

```python
from datatypical_viz import profile_plot, get_top_sample

# Get top sample safely
top_idx = get_top_sample(results, 'archetypal_rank')

# Profile with local ordering (by this sample's importance)
ax = profile_plot(dt, top_idx, significance='archetypal', order='local')

# Profile with global ordering (by average importance)
ax = profile_plot(dt, top_idx, significance='archetypal', order='global')
```

**Elements**:
- X-axis: Features (ordered by local or global importance)
- Y-axis: Shapley value for this sample
- Bar color: Normalized feature value (viridis)
- Zero line: Reference for positive/negative contributions

**Interpretation**:

| Bar Color | Bar Direction | Meaning |
|-----------|---------------|---------|
| Yellow (high value) | Positive | High value HELPS |
| Yellow (high value) | Negative | High value HURTS |
| Purple (low value) | Positive | Low value HELPS |
| Purple (low value) | Negative | Low value HURTS |

**Magnitude Guide**:
- |phi| > 0.5: Dominant feature for this sample
- 0.2 < |phi| < 0.5: Important contributor
- |phi| < 0.2: Minor factor

### Helper Functions

```python
from datatypical_viz import get_top_sample

# Safely get top sample (handles NaN columns gracefully)
top_arch = get_top_sample(results, 'archetypal_rank')
top_formative = get_top_sample(results, 'archetypal_shapley_rank')

# Get multiple top samples
top_5 = get_top_sample(results, 'prototypical_rank', n=5)

# Get bottom samples
bottom_idx = get_top_sample(results, 'stereotypical_rank', mode='min')
```

---

## Decision Making Framework

### Selection Decisions

**Goal: Select k representative samples**

Strategy:
1. Use prototypical ranks for diversity
2. Check formative ranks to ensure coverage
3. Verify no critical boundary samples excluded

```python
# Select top prototypes
top_proto = results.nlargest(k, 'prototypical_rank')

# Check formative coverage
formative_covered = top_proto['prototypical_shapley_rank'].mean()
if formative_covered < 0.5:
    # Add some high-formative samples
    additional = results.nlargest(k//10, 'prototypical_shapley_rank')
    selection = pd.concat([top_proto, additional]).drop_duplicates()
```

**Goal: Find extreme/novel samples**

Strategy:
1. Use archetypal ranks
2. Filter by formative ranks to avoid redundancy
3. Profile plots to understand what makes them extreme

```python
# Find extreme samples
extreme = results[results['archetypal_rank'] > 0.8]

# Keep only structure-defining ones
critical_extreme = extreme[extreme['archetypal_shapley_rank'] > 0.5]

# Examine with profile plots
for idx in critical_extreme.index:
    profile_plot(dt, idx, significance='archetypal')
```

**Goal: Optimize toward target**

Strategy:
1. Use stereotypical ranks
2. Profile plots show what to modify
3. Formative ranks identify unique vs common high-performers

```python
# Top performers
best = results.nlargest(20, 'stereotypical_rank')

# Understand what makes them good
for idx in best.index[:5]:
    profile_plot(dt, idx, significance='stereotypical')
```

### Quality Control Decisions

**Identifying outliers vs extremes**:

**Outliers** (error or noise):
- High archetypal rank + very high formative rank + isolated
- Profile plot shows one feature dominates (likely error)

**Extremes** (real boundary cases):
- High archetypal rank + moderate formative rank
- Profile plot shows multiple features contribute
- Similar samples exist (check formative rank)

**Dataset completeness**:

**Well-covered space**:
- Prototypical actual ~ prototypical formative (correlation > 0.7)
- Few samples in top-left quadrant of significance plot

**Gaps in coverage**:
- Many samples in top-left quadrant (structure creators)
- Low correlation between actual and formative
- Action: Collect more samples in gap regions

### Research Decisions

**Mechanism understanding**:

Use profile plots to identify:
- **Which features drive outcomes**: Look at stereotypical explanations
- **Feature interactions**: Samples with multi-feature patterns
- **Trade-offs**: Negative Shapley values indicate constraints

**Hypothesis generation**:

Heatmaps reveal:
- **Clusters of mechanism**: Similar patterns = similar mechanisms
- **Alternative strategies**: Different patterns achieving same outcome
- **Necessary vs sufficient features**: Consistent positives vs variable

---

## Assumptions and Limitations

### General Assumptions

**1. Numeric Features**

All computational features must be numeric. Categorical features should be encoded before processing. Text and graph data require feature extraction via the appropriate data type pathway.

**2. Feature Scaling**

DataTypical applies MinMax scaling to [0, 1]. This assumes:
- Features are comparable after scaling
- Outliers don't dominate (robust to some outliers via clipping)
- Linear scaling is appropriate

**3. Missing Data**

Median imputation is applied. This assumes:
- Missingness is random (MCAR or MAR)
- Median is a reasonable proxy
- Large amounts of missingness (>30%) may affect results

**4. Sample Independence**

Samples are assumed independent. Temporal or spatial correlations are not explicitly modeled. Duplicates should be removed beforehand.

### Method-Specific Limitations

**Archetypal (NMF)**:

- **Non-negativity**: Requires features >= 0 or translation (some patterns may be lost)
- **Linear assumption**: NMF assumes linear combinations; non-linear archetypes may be missed
- **Convergence**: May not find global optimum; multiple runs recommended
- **Rank selection**: No automatic nmf_rank selection; requires domain knowledge

**Prototypical (Facility Location)**:

- **Greedy approximation**: Not guaranteed optimal (theoretical ratio ~ 0.632)
- **Cosine similarity**: Assumes angular distance is meaningful; may not suit all features
- **No cluster assumption**: Doesn't assume clusters exist; may select multiple from same region

**Stereotypical (Distance to Target)**:

- **Single feature**: Only one target at a time; multi-objective requires multiple runs
- **Distance metric**: Absolute difference may not suit all targets
- **Median reference**: Assumes median is meaningful reference point

**Shapley Values**:

- **Computational cost**: Exponential exact computation requires permutation approximation
- **Convergence**: May require many permutations; early stopping helps
- **Value function design**: Simplified for single samples; may not perfectly reflect computation
- **Additivity**: Assumes value function properties; non-monotonic functions violate assumptions

### Interpretability Caveats

**Feature Attribution**:

Shapley explanations show correlation, not causation. A feature may have high attribution because it correlates with the true driver, not because it causes significance.

**Formative vs Actual**:

A sample can be critical for structure (high formative) without being an obvious extreme (moderate actual). Don't equate structural importance with significance.

**Scale Sensitivity**:

Rankings depend on scaling. Different scaling methods may produce different rankings, especially for archetypal analysis.

### When Results May Be Unreliable

- **Very small datasets** (n < 20): Insufficient samples for meaningful Shapley computation
- **Very high dimensions** (d > n): Overfitting risk; consider dimensionality reduction
- **Highly correlated features**: Feature attributions may be unstable; consider PCA first
- **Non-numeric targets**: Stereotypical analysis requires numeric targets
- **Extreme missingness**: >50% missing in any feature reduces reliability

---

## Use Cases and Examples

### Use Case 1: Drug Discovery - Alternative Mechanisms

**Goal**: Find compounds with high activity and understand if they achieve activity through different mechanisms.

```python
from datatypical import DataTypical
from datatypical_viz import significance_plot, heatmap, profile_plot

# Detailed analysis with explanations
dt = DataTypical(
    shapley_mode=True,
    stereotype_column='activity',
    fast_mode=False
)
results = dt.fit_transform(compounds[feature_cols])

# Are high-activity compounds redundant or diverse?
significance_plot(
    results, 
    significance='stereotypical',
    color_by='IC50',
    label_top=5
)

# Feature patterns across active compounds
heatmap(
    dt, results,
    significance='stereotypical',
    order='actual',
    top_n=25
)

# Find unique mechanism compounds (high formative)
unique_mechanisms = results[
    (results['stereotypical_shapley_rank'] > 0.8)
].nlargest(3, 'stereotypical_rank')

# Profile each mechanism
for idx in unique_mechanisms.index:
    profile_plot(dt, idx, significance='stereotypical', order='local')
```

**Discovery**: Three distinct mechanisms for achieving high activity:
- Mechanism 1: Dominated by high logP (lipophilic pathway)
- Mechanism 2: High aromatic rings with moderate logP (pi-stacking pathway)
- Mechanism 3: High HBD and HBA with low logP (polar interaction pathway)

### Use Case 2: Materials Science - Boundary Extremes

**Goal**: Identify materials with extreme property combinations.

```python
dt = DataTypical(
    shapley_mode=True,
    fast_mode=False,
    nmf_rank=10
)
results = dt.fit_transform(materials[property_cols])

# Which extremes are unique vs clustered?
significance_plot(
    results,
    significance='archetypal',
    color_by='performance_metric'
)

# Find critical boundary materials
critical = results[
    (results['archetypal_rank'] > 0.9) &
    (results['archetypal_shapley_rank'] > 0.7)
]
```

**Discovery**: 5 critical boundary materials defining extreme property space, 15 replaceable extremes that cluster into 3 groups.

### Use Case 3: Dataset Curation - Coverage Gaps

**Goal**: Select 500 diverse molecules from 50,000.

```python
dt = DataTypical(
    shapley_mode=True,
    fast_mode=True,
    shapley_top_n=1000
)
results = dt.fit_transform(molecules[descriptor_cols])

# Critical representatives (top-right quadrant)
critical_reps = results[
    (results['prototypical_rank'] > 0.7) &
    (results['prototypical_shapley_rank'] > 0.7)
]

# Boundary definers (archetypal top-right)
boundary_definers = results[
    (results['archetypal_rank'] > 0.9) &
    (results['archetypal_shapley_rank'] > 0.7)
]

# Gap fillers (prototypical top-left)
gap_fillers = results[
    (results['prototypical_rank'] < 0.4) &
    (results['prototypical_shapley_rank'] > 0.6)
]

# Combine for curated set
curated = pd.concat([critical_reps, boundary_definers, gap_fillers]).drop_duplicates()
```

**Discovery**: 99.4% coverage of property space with 1% of molecules, with better boundary coverage than random sampling.

### Use Case 4: Quality Control - Novel vs Known Variation

**Goal**: Detect if a new batch represents unprecedented or known variation.

```python
# Fit on historical data
dt = DataTypical(shapley_mode=True, fast_mode=False)
results = dt.fit_transform(historical_batches)

# Transform new batch
new_results = dt.transform(new_batch)

# Check archetypal rank (unprecedented = very high)
arch_rank = new_results['archetypal_rank'].values[0]

if arch_rank > 0.95:
    print("UNPRECEDENTED: Novel variation detected")
    profile_plot(dt, new_results.index[0], significance='archetypal')
elif arch_rank > 0.8:
    # Check if it matches known extremes
    form_rank = new_results['archetypal_shapley_rank'].values[0]
    if form_rank > 0.7:
        print("NOVEL EXTREME: Unique boundary case")
    else:
        print("KNOWN EXTREME: Similar to existing extremes")
else:
    print("NORMAL: Within expected variation")
```

---

## Summary

### Three Significance Types

| Type | Basis | Finds | Use When |
|------|-------|-------|----------|
| Archetypal | Objective | Extremes | Exploring boundaries |
| Prototypical | Representative | Typical cases | Building diverse sets |
| Stereotypical | Subjective | Target matches | Optimizing for goals |

### Dual Perspectives

| Perspective | Measures | Use For |
|-------------|----------|---------|
| Actual | What samples ARE | Selection, characterization |
| Formative | What samples CREATE | Structure understanding, QC |

### Four-Quadrant Discovery

| Actual | Formative | Name | Action |
|--------|-----------|------|--------|
| High | High | Critical | Keep always |
| High | Low | Replaceable | Deduplicate |
| Low | High | Gap Filler | Keep for coverage |
| Low | Low | Redundant | Safe to remove |

### Visualization Guide

| Plot | Shows | Key Insight |
|------|-------|-------------|
| Significance | Actual vs Formative | Which samples are critical |
| Heatmap | Feature contributions | Why samples are significant |
| Profile | Individual breakdown | How to modify/optimize |

### Decision Framework

1. **Define goal** -> Choose significance type
2. **Examine distribution** -> Use significance plot
3. **Understand mechanism** -> Use heatmap/profile
4. **Select samples** -> Use dual perspectives
5. **Validate** -> Check formative coverage

**Remember**: All three significance types are valid. Choose based on your scientific question, not the "best" metric. The dual perspective reveals insights impossible from ranks alone.

---

*DataTypical v0.7 - Interpretation Guide*
*For understanding and applying instance significance analysis*