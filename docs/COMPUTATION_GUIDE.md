# DataTypical v0.7 - Computation Guide

## Table of Contents

1. [Overview](#overview)
2. [Dependencies and Optional Packages](#dependencies-and-optional-packages)
3. [Data Format Auto-Detection](#data-format-auto-detection)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Archetypal Computation](#archetypal-computation)
6. [Prototypical Computation](#prototypical-computation)
7. [Stereotypical Computation](#stereotypical-computation)
8. [Shapley Value Computation](#shapley-value-computation)
9. [Parallelization and Threading](#parallelization-and-threading)
10. [Determinism and Reproducibility](#determinism-and-reproducibility)
11. [Memory Management](#memory-management)
12. [Performance Optimization](#performance-optimization)
13. [Parameters Reference](#parameters-reference)
14. [Thresholds and Constants](#thresholds-and-constants)
15. [Troubleshooting](#troubleshooting)
16. [Mathematical Foundations](#mathematical-foundations)
17. [References](#references)

---

## Overview

DataTypical v0.7 implements a dual-perspective significance analysis framework that identifies important instances in scientific datasets through three complementary lenses:

| Significance Type | Perspective | Mathematical Basis |
|-------------------|-------------|-------------------|
| **Archetypal** | Objective | Geometric extremeness via NMF or PCHA |
| **Prototypical** | Representative | Coverage optimization via Facility Location |
| **Stereotypical** | Subjective | Distance-based targeting |

The key innovation is the Shapley-based dual perspectives introduced in v0.6:

- **Actual significance**: Samples that ARE archetypal/prototypical/stereotypical (geometric properties)
- **Formative significance**: Samples that CREATE the archetypal/prototypical/stereotypical structure (influence-based)
- **Explanations**: Feature-level attributions explaining WHY each sample is significant

Version 0.7 introduced `fast_mode` for rapid exploration with 30x speedup versus publication-quality analysis.

Version 0.7.2 replaces the ConvexHull-based archetypal formative value function with a cached archetype geometry approach. The previous ConvexHull implementation scaled as O(n^(d/2)) per permutation call, making it intractable for datasets with more than 8 features. The cached approach uses fixed archetypes from the fitted model (H_), reducing computation to O(n_archetypes x n_subset x n_features) distance arithmetic while strengthening the scientific consistency of the dual-perspective scatter plot.

---

## Dependencies and Optional Packages

### Required Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.20 | Core numerical operations |
| `pandas` | >= 1.3 | Data handling and indexing |
| `scikit-learn` | >= 1.0 | NMF, MinMaxScaler, TfidfVectorizer |
| `joblib` | >= 1.0 | Parallel processing |
| `threadpoolctl` | >= 3.0 | Thread control for determinism |

### Optional Dependencies (Performance)

| Package | Version | Purpose | Speedup |
|---------|---------|---------|---------|
| `numba` | >= 0.56 | JIT compilation of distance functions | 2-5x |
| `faiss` | >= 1.7 | Fast similarity search for large datasets | 10-50x |
| `scipy` | >= 1.7 | Sparse matrices, ConvexHull | Required for text |
| `py_pcha` | >= 0.1 | Principal Convex Hull Analysis | Stable high-D archetypes |
| `networkx` | >= 2.6 | Graph topology features | Required for graph mode |

### Detection at Runtime

```python
# Check availability
from datatypical import DataTypical

dt = DataTypical(verbose=True)
# Output shows which optimizations are available:
#   NUMBA_AVAILABLE: True/False
#   FAISS_AVAILABLE: True/False
#   PCHA: True/False
```

DataTypical automatically uses available accelerators when present, with graceful fallbacks when absent.

---

## Data Format Auto-Detection

### Detection Algorithm

DataTypical automatically determines input format using the following priority order:

```
Priority 1: Graph Data
    Condition: 'edges' or 'edge_index' parameter present
    Result: data_type = 'graph'

Priority 2: Text Data
    Condition: X is list/tuple AND X[0] is str
    Result: data_type = 'text'

Priority 3: Tabular Data
    Condition: X is DataFrame or ndarray
    Result: data_type = 'tabular'

Priority 4: Error
    Condition: None of the above
    Result: ValueError with guidance
```

### Implementation Details

The detection is performed in `_auto_detect_data_type()`:

```python
def _auto_detect_data_type(self, X, **kwargs) -> str:
    # Priority 1: Graph (edges parameter indicates graph data)
    if 'edges' in kwargs or 'edge_index' in kwargs:
        return 'graph'
    
    # Priority 2: Text (list/tuple of strings)
    if isinstance(X, (list, tuple)):
        if len(X) > 0 and isinstance(X[0], str):
            return 'text'
    
    # Priority 3: Tabular (DataFrame or array)
    if isinstance(X, (pd.DataFrame, np.ndarray)):
        return 'tabular'
    
    # Cannot determine - raise with helpful message
    raise ValueError(...)
```

### Detection Overhead

Format detection adds negligible overhead (< 1ms) as it performs only type checking without data inspection.

### Manual Override

Detection can be bypassed:

```python
dt = DataTypical(data_type='tabular')  # Force tabular processing
```

---

## Preprocessing Pipeline

### Tabular Data Pipeline

The preprocessing sequence for tabular data follows strict ordering to ensure reproducibility:

**Step 1: Label Column Separation**

If `label_columns` is specified, these columns are preserved but excluded from numerical processing. They are rejoined to results after analysis.

**Step 2: Numeric Feature Selection**

Automatic selection of numeric columns with exclusion heuristics:

- Columns named `id`, `*_id`, `id_*` are excluded
- Columns with >= 80% unique values are excluded (likely identifiers)
- Strictly monotonic columns are excluded (likely indices)

```python
# Uniqueness threshold
if nunique >= 0.8 * n:
    to_drop.add(col)
```

**Step 3: Missing Value Handling**

DataTypical uses **median imputation** for missing values:

```python
med = np.nanmedian(X, axis=0)      # Per-feature median
inds = np.where(np.isnan(X))       # Find NaN positions
X[inds] = np.take(med, inds[1])    # Replace with column median
```

**Rationale for Median Imputation**:

- Robust to outliers (unlike mean)
- Preserves data scale
- Deterministic (no random sampling)
- Works with any distribution shape

The `max_missing_frac` parameter (default: 1.0) allows rejection of datasets with excessive missingness:

```python
if worst_missingness > self.max_missing_frac:
    raise DataTypicalError("Missingness too high")
```

**Step 4: Feature Scaling**

MinMax scaling to [0, 1] range with clipping:

```python
self.scaler_ = MinMaxScaler(copy=True, clip=True)
X_scaled = self.scaler_.fit_transform(X)
```

The `clip=True` ensures transform-time data stays within [0, 1] even if it exceeds training bounds.

**Mathematical formulation**:

For feature j with training minimum m_j and maximum M_j:

$$x_{ij}^{scaled} = \text{clip}\left(\frac{x_{ij} - m_j}{M_j - m_j}, 0, 1\right)$$

**Step 5: Constant Feature Removal**

Features with zero variance after scaling are dropped:

```python
var = X_scaled.var(axis=0)
keep_mask = var > 0.0
X_scaled = X_scaled[:, keep_mask]
```

Dropped columns are recorded in `dropped_columns_` for transparency.

**Step 6: Optional Feature Weighting**

If `feature_weights` is provided, features are multiplied by weights:

```python
if self.feature_weights is not None:
    X_scaled = X_scaled * weights[keep_mask]
```

**Step 7: L2 Normalization**

A separate L2-normalized copy is created for cosine similarity operations:

```python
def _l2_normalize_rows_dense(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms
```

### Text Data Pipeline

**Step 1: TF-IDF Vectorization**

```python
self.vectorizer_ = TfidfVectorizer()
X_tfidf = self.vectorizer_.fit_transform(corpus_list)
```

Default TfidfVectorizer settings are used (can be customized in future versions).

**Step 2: Sparse MinMax Scaling**

Custom sparse-aware scaling preserves memory efficiency:

```python
def _sparse_minmax_0_1_nonneg(M: sp.spmatrix) -> sp.spmatrix:
    A = M.tocsc(copy=False)
    col_max = A.max(axis=0).toarray().ravel()
    col_max[col_max == 0.0] = 1.0
    return (A @ sp.diags(1.0 / col_max)).tocsr()
```

**Step 3: Sparse L2 Normalization**

```python
def _sparse_l2_normalize_rows(X: sp.spmatrix) -> sp.spmatrix:
    X = X.tocsr(copy=False)
    sq = X.multiply(X).sum(axis=1)
    norms = np.sqrt(np.maximum(np.asarray(sq).ravel(), 0.0))
    norms[norms == 0.0] = 1.0
    D = sp.diags(1.0 / norms)
    return D @ X
```

### Graph Data Pipeline

**Step 1: Topology Feature Computation**

If `graph_topology_features` is specified, NetworkX computes structural metrics:

| Feature | Complexity | Description |
|---------|------------|-------------|
| `degree` | O(\|E\|) | Node degree |
| `clustering` | O(\|V\| × k²) | Local clustering coefficient |
| `pagerank` | O((\|V\| + \|E\|) × iter) | PageRank centrality |
| `triangles` | O(\|V\| × k²) | Triangle count |
| `betweenness` | O(\|V\| × \|E\|) | Betweenness centrality |
| `closeness` | O(\|V\| × \|E\|) | Closeness centrality |
| `eigenvector` | O(\|V\|² × iter) | Eigenvector centrality |

**Step 2: Feature Merging**

Topology features are appended to node features, then processed as tabular data.

---

## Archetypal Computation

### Method Selection (v0.7)

DataTypical v0.7 introduces `archetypal_method` parameter:

| Method | When Used | Stability | Speed |
|--------|-----------|-----------|-------|
| `'aa'` | Publication mode (default when `fast_mode=False`) | High | Medium |
| `'nmf'` | Fast mode (default when `fast_mode=True`) | Medium | Fast |

The `'aa'` method attempts PCHA first, then ConvexHull, with NMF as final fallback.

### Method 1: PCHA (Principal Convex Hull Analysis)

**Algorithm**: Iterative optimization to find archetypal points that lie on the convex hull of the data.

**Implementation**:

```python
from py_pcha import PCHA

# Transpose data (PCHA expects features × samples)
X_T = X_scaled.T.copy()

# Ensure non-negative
if X_T.min() < 0:
    X_T = X_T - X_T.min() + 1e-10

# Compute archetypes
XC, S, C, SSE, varexpl = PCHA(X_T, noc=k, delta=0.0)

# S: (k, n_samples) - archetype membership weights
# XC: (n_features, k) - archetype locations in feature space
```

**Output**:

- `W = S.T`: Sample weights matrix (n_samples, k)
- `H = XC.T`: Archetype feature matrix (k, n_features)

**Stability**: PCHA is numerically stable in high dimensions (tested up to 100+ features).

### Method 2: ConvexHull (Low-Dimensional Fallback)

**Condition**: Used when PCHA fails AND n_features <= 20.

**Algorithm**:

1. Compute convex hull vertices using scipy
2. Identify boundary sample indices
3. Compute distance-based weights to boundary points

```python
hull = ConvexHull(X_scaled)
boundary_indices = np.unique(hull.simplices.ravel())

# Weights via inverse distance
distances = cdist(point, X_scaled[boundary_indices])
weights = 1.0 / (distances + 1e-6)
W[i, :] = weights / weights.sum()
```

**Critical Limitation**: ConvexHull can segfault in high dimensions (> 20). This is a known scipy/Qhull issue. DataTypical enforces a strict dimension check:

```python
if n_features > 20 or ConvexHull is None:
    # Skip ConvexHull, use range-based fallback
```

### Method 3: NMF (Non-negative Matrix Factorization)

**Algorithm**: Approximate X as W × H where both matrices are non-negative.

**Mathematical formulation**:

Given X ∈ ℝ^{n×d}, find W ∈ ℝ^{n×k} and H ∈ ℝ^{k×d} minimizing:

$$\min_{W,H \geq 0} \|X - WH\|_F^2$$

**Implementation**:

```python
nmf = NMF(
    n_components=k_eff,
    init='nndsvd',           # Deterministic initialization
    max_iter=self.max_iter_nmf,  # Default: 400
    tol=self.tol_nmf,        # Default: 1e-4
    random_state=self.random_state
)
W = nmf.fit_transform(X_nonneg)
H = nmf.components_
```

**Initialization**: `nndsvd` (Non-negative Double Singular Value Decomposition) provides deterministic initialization for reproducibility.

### Archetypal Score Computation

The archetypal score combines two components:

**Component 1: NMF Weight Maximum (70%)**

```python
W_row_sum = W.sum(axis=1, keepdims=True)
W_row_sum[W_row_sum == 0.0] = 1.0
W_norm = W / W_row_sum
arch_wmax = W_norm.max(axis=1)
```

**Intuition**: Samples with high weight on a single basis vector have extreme profiles.

**Component 2: Corner Score (30%)**

```python
# Select 2 highest-variance features that span [0, 1]
m = np.minimum(X2, 1.0 - X2)
dmin = np.sqrt(np.sum(m * m, axis=1))
corner_score = 1.0 - np.clip(dmin / sqrt(n_features), 0.0, 1.0)
```

**Intuition**: Distance from the nearest corner of the unit hypercube.

**Combined Score**:

$$\text{archetypal\_score} = 0.7 \times \text{arch\_wmax} + 0.3 \times \text{corner\_score}$$

---

## Prototypical Computation

### Algorithm: CELF (Cost-Effective Lazy Forward Selection)

Prototypes are selected via the facility location problem with lazy evaluation:

**Objective**: Select k prototypes P to maximize coverage:

$$\max_P \sum_{i=1}^{n} w_i \cdot \max_{p \in P} \text{sim}(x_i, x_p)$$

where sim is cosine similarity and w_i are optional weights.

### Implementation Details

**Similarity Matrix**:

For L2-normalized data, cosine similarity equals the dot product:

```python
S = X_l2 @ X_l2.T  # (n, n) similarity matrix
np.maximum(S, 0.0, out=S)  # Clip negative similarities
```

**CELF Optimization**:

```python
# Initial gains: g0[c] = Σ_i w_i * max(0, <x_i, x_c>)
g0 = np.zeros(n)
for batch in batches:
    S_batch = X[batch] @ X.T
    g0 += (w[batch, None] * S_batch).sum(axis=0)

# Heap with lazy evaluation
heap = [(-gain, key, idx) for idx in candidates]
heapq.heapify(heap)

while len(selected) < k and heap:
    neg_g_est, _, c = heapq.heappop(heap)
    
    if last_eval[c] == iteration:
        # Accept this candidate
        selected.append(c)
        np.maximum(best, S[c, :], out=best)
    else:
        # Recompute exact gain
        improv = S[c, :] - best
        improv[improv < 0.0] = 0.0
        g_exact = (w * improv).sum()
        heapq.heappush(heap, (-g_exact, key, c))
```

### Deterministic Tie-Breaking

For reproducibility, ties are broken using content-based hashing:

```python
def row_key(i: int) -> int:
    h = hashlib.blake2b(X[i].tobytes(), digest_size=8)
    return int.from_bytes(h.digest(), "big", signed=False)
```

This ensures identical results regardless of input ordering.

### Optional FAISS Acceleration

For datasets with n > 1,000, FAISS provides massive speedups:

```python
if FAISS_AVAILABLE and n > 1000 and not self.speed_mode:
    return self._select_with_faiss(X, weights, forbidden)
```

### Prototypical Score Computation

**Component 1: Distance Score (50%)**

```python
best_euc = _euclidean_min_to_set_dense(X_euc, P_mat)
norm95 = np.percentile(best_euc, 95) or 1.0
proto_d_norm95 = np.clip(best_euc / norm95, 0.0, 1.0)
distance_score = 1.0 - proto_d_norm95
```

**Component 2: Cosine Score (50%)**

```python
best_cos, proto_label = self._assignments_cosine(X_l2, P_idx)
```

**Combined Score**:

$$\text{prototypical\_score} = 0.5 \times (1 - \text{dist\_norm}) + 0.5 \times \text{cos\_sim}$$

### Automatic Prototype Count (Kneedle)

If `auto_n_prototypes='kneedle'`:

```python
U = np.cumsum(gains)
U_norm = U / U[-1]
x = np.linspace(1/k, 1.0, k)
diff = U_norm - x
knee = np.argmax(diff) + 1
```

---

## Stereotypical Computation

### Target Types

| Target | Interpretation | Score Computation |
|--------|----------------|-------------------|
| `'max'` | Maximize feature | `max(0, value - median)` |
| `'min'` | Minimize feature | `max(0, median - value)` |
| `float` | Match specific value | `-|value - target|` |

### Algorithm

```python
def _compute_stereotypical_rank(self, X_scaled, index, stereotype_source):
    # Case 1: Column-based stereotype
    if self.stereotype_column is not None and stereotype_source is not None:
        values = stereotype_source.values
        
        if self.stereotype_target == 'max':
            return values / (values.max() or 1.0)
        elif self.stereotype_target == 'min':
            return 1.0 - (values / (values.max() or 1.0))
        else:
            target = float(self.stereotype_target)
            distances = np.abs(values - target)
            return 1.0 - (distances / (distances.max() or 1.0))
    
    # Case 2: Default (feature extremeness)
    X_arr = np.asarray(X_scaled)
    s = np.max(np.abs(X_arr - 0.5), axis=1) * 2.0
    return s / (s.max() or 1.0)
```

### Keyword-Based Stereotypes (Text Mode)

For text data with `stereotype_keywords`:

```python
keyword_indices = [vocab[kw] for kw in keywords if kw in vocab]
X_keywords = X_tfidf[:, keyword_indices]
scores = X_keywords.sum(axis=1)  # TF-IDF sum for keywords
```

---

## Shapley Value Computation

### Overview

DataTypical computes two types of Shapley values:

1. **Explanations**: Feature-level attributions (why is this sample significant?)
2. **Formative**: Sample-level attributions (which samples create structure?)

### Mathematical Foundation

Shapley values decompose a value function V among players (features or samples):

$$\phi_i = \frac{1}{n!} \sum_{\pi \in \Pi} \left[ V(S_i^\pi \cup \{i\}) - V(S_i^\pi) \right]$$

where S_i^π is the set of players before i in permutation π.

### Permutation-Based Approximation

Exact Shapley computation is O(2^n), so we use permutation sampling:

```python
for perm in permutations:
    for j, sample_idx in enumerate(perm):
        S = perm[:j]  # Coalition before sample
        v_with = value_function(X[S ∪ {sample_idx}])
        v_without = value_function(X[S])
        marginal[sample_idx] += v_with - v_without

Phi = marginal / n_permutations
```

### Value Functions

**Archetypal Formative Value Function (v0.7.2)**:

Measures how well a sample subset supports the pre-computed archetypal geometry. Uses archetypes cached from the initial PCHA or NMF fit (stored in H_) as a fixed geometric reference.

```python
def formative_archetypal_pcha_cached(X_subset, indices, context):
    archetypes = context['archetypes']  # H_ from fitted model (n_archetypes, n_features)

    # Pairwise distances: each archetype to each subset member
    diffs = archetypes[:, np.newaxis, :] - X_subset[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))  # (n_archetypes, n_subset)

    # Minimum distance from each archetype to its nearest subset member
    min_dists = dists.min(axis=1)  # (n_archetypes,)

    # Negative mean distance: higher = better coverage of archetypal corners
    return -np.mean(min_dists)
```

**Scientific rationale**: Both axes of the dual-perspective scatter plot now reference the same geometric model. The actual significance axis (archetypal_rank) measures how extreme each sample is relative to H_. The formative significance axis (archetypal_shapley_rank) measures how much each sample contributes to covering H_. Because both axes derive from the same fitted archetypes, the scatter plot is a true dual perspective on a single geometric model.

**Computational improvement**: The previous `formative_archetypal_convex_hull` function scaled as O(n^(d/2)) per value function call due to ConvexHull computation. With 100 permutations and 312 samples, this required approximately 31,200 ConvexHull calls. In 13 dimensions, each call scales as O(n^6.5), making the total computation intractable. The cached approach replaces this with O(n_archetypes x n_subset x n_features) distance arithmetic, which is fast regardless of dimensionality.

| Method | Per-call complexity | 312 samples, 13 features |
|--------|--------------------|-----------------------------|
| ConvexHull (v0.7) | O(n^6.5) | Days |
| Cached archetypes (v0.7.2) | O(n_arch x n_sub x d) | Minutes |

**Prototypical Formative Value Function**:

Measures pairwise similarity coverage.

```python
def formative_prototypical_coverage(X_subset, indices, context):
    X_l2 = X_subset / np.linalg.norm(X_subset, axis=1, keepdims=True)
    similarities = X_l2 @ X_l2.T
    np.fill_diagonal(similarities, 0)
    max_sims = np.max(similarities, axis=1)
    return np.mean(max_sims)
```

**Stereotypical Formative Value Function**:

Measures contribution toward target.

```python
def formative_stereotypical_extremeness(X_subset, indices, context):
    subset_vals = context['target_values'][indices]
    
    if context['target'] == 'max':
        return np.mean(np.maximum(subset_vals - median, 0))
    elif context['target'] == 'min':
        return np.mean(np.maximum(median - subset_vals, 0))
    else:
        target_val = float(context['target'])
        return median_dist - np.mean(np.abs(subset_vals - target_val))
```

### Early Stopping

Shapley computation uses convergence detection:

```python
class ShapleyEarlyStopping:
    def update(self, shapley_estimates, n_perms):
        if n_perms < 20:  # Minimum permutations
            return False
        
        rel_change = np.abs(new - old) / (np.abs(old) + 1e-12)
        mean_rel_change = np.mean(rel_change)
        
        if mean_rel_change < self.tolerance:  # Default: 0.01
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        return self.stable_count >= self.patience  # Default: 10
```

**Parameters**:

- `shapley_early_stopping_patience`: Consecutive stable iterations (default: 10)
- `shapley_early_stopping_tolerance`: Relative change threshold (default: 0.01)

### Subsampling (fast_mode)

When `shapley_top_n` is set (automatically 0.5 in fast_mode):

```python
if shapley_top_n is not None:
    # Compute preliminary ranks
    temp_results = self._score_with_fitted(X_scaled, X_l2, index)
    
    # Get top N from each metric
    top_arch = temp_results.nlargest(n_subsample, 'archetypal_rank').index
    top_proto = temp_results.nlargest(n_subsample, 'prototypical_rank').index
    top_stereo = temp_results.nlargest(n_subsample, 'stereotypical_rank').index
    
    # Union ensures no empty Shapley explanations
    subsample_indices = top_arch | top_proto | top_stereo
```

### JIT-Compiled Helpers

Performance-critical loops use Numba JIT compilation:

```python
@jit(nopython=True, cache=True, fastmath=True)
def _compute_marginals_jit(perm, values, n_samples, n_features):
    shapley_contrib = np.zeros((n_samples, n_features))
    for j in range(n_samples):
        sample_idx = perm[j]
        marginal = values[j+1] - values[j]
        for f in range(n_features):
            shapley_contrib[sample_idx, f] = marginal / n_features
    return shapley_contrib
```

---

## Parallelization and Threading

### Parallelization Strategy

DataTypical uses joblib with the **threading backend** for Shapley computation:

```python
batch_results = Parallel(
    n_jobs=self.n_jobs,
    backend='threading',  # Shared memory, no pickle overhead
    verbose=0
)(delayed(self._process_single_permutation)(...) for perm in batch_perms)
```

**Why Threading Over Multiprocessing**:

- Shared memory avoids data copying
- NumPy operations release GIL
- Lower overhead for small tasks
- Better for memory-constrained systems

### Thread Control for Determinism

BLAS libraries use internal threading that can cause non-determinism:

```python
class _ThreadControl:
    def __enter__(self):
        if self.deterministic:
            self._ctx = threadpool_limits(limits=1)  # Single-thread BLAS
        else:
            self._ctx = threadpool_limits(limits=None)  # Default
```

**Effect**: When `deterministic=True`, internal BLAS threads are limited to 1, ensuring identical results across runs.

### n_jobs Parameter

| Value | Behavior |
|-------|----------|
| `1` | Serial execution (no parallelism) |
| `-1` | Use all available cores |
| `N` | Use exactly N cores |

### Parallelization Decision Logic

```python
# Shapley sample-level
use_parallel = self.n_jobs != 1 and n_samples >= 20

# Shapley feature-level  
use_parallel = self.n_jobs != 1 and n_features >= 10
```

Small problems use serial computation to avoid parallelization overhead.

---

## Determinism and Reproducibility

### Sources of Non-Determinism

1. **Random initialization**: NMF, permutation sampling
2. **BLAS threading**: Multi-threaded operations may vary
3. **Floating-point accumulation**: Sum order affects precision
4. **Hash collisions**: Tie-breaking in prototype selection

### Ensuring Reproducibility

**Step 1: Set random_state**

```python
dt = DataTypical(random_state=42)
```

This seeds all random operations:

```python
def _seed_everything(seed: int) -> None:
    np.random.seed(seed)

# In ShapleySignificanceEngine
self.rng = np.random.RandomState(random_state)
```

**Step 2: Enable deterministic mode**

```python
dt = DataTypical(deterministic=True)  # Default
```

This:
- Limits BLAS threads to 1
- Uses `nndsvd` NMF initialization (deterministic)
- Uses content-based tie-breaking

**Step 3: Control speed_mode**

```python
dt = DataTypical(speed_mode=False)  # Stricter determinism
```

`speed_mode=True` may use approximations that vary slightly.

### Verification

```python
# Run twice, compare results
dt1 = DataTypical(random_state=42, deterministic=True)
results1 = dt1.fit_transform(data)

dt2 = DataTypical(random_state=42, deterministic=True)
results2 = dt2.fit_transform(data)

assert results1.equals(results2)  # Should pass
```

---

## Memory Management

### Explicit Cleanup Function

Large temporaries are explicitly deleted:

```python
def _cleanup_memory(*arrays, force_gc: bool = False) -> None:
    for arr in arrays:
        if arr is not None:
            del arr
    if force_gc:
        gc.collect()
```

**Usage Points**:

- After similarity matrix computation
- After Shapley batch processing
- After transform operations

### Memory-Optimized Patterns

**Chunked Distance Computation**:

```python
def _chunk_len(n_left, n_right, bytes_per, max_memory_mb):
    max_bytes = max_memory_mb * 1024 * 1024
    return max(1, min(n_right, int(max_bytes // (n_left * bytes_per))))
```

**dtype Selection**:

```python
# Default to float32 (4 bytes vs 8 bytes)
self.dtype = "float32"

# Preserve float64 when input requires it
if input_dtype == np.float64:
    target_dtype = np.float64
```

### max_memory_mb Parameter

Controls maximum memory for chunked operations:

```python
dt = DataTypical(max_memory_mb=2048)  # Default: 2GB
```

Used in:
- Euclidean distance computation
- Similarity matrix computation
- CELF prototype selection

### Memory Usage Estimates

| Operation | Memory Complexity |
|-----------|------------------|
| Similarity matrix | O(n² × 8 bytes) |
| NMF | O(n × d × k × 8 bytes) |
| Shapley (sample) | O(n × d × 8 bytes) per permutation |
| Shapley (feature) | O(n × d × 8 bytes) |

**Example**: 10,000 samples
- Similarity matrix: ~800 MB
- With float32: ~400 MB

---

## Performance Optimization

### fast_mode Preset

`fast_mode=True` applies exploration-optimized defaults:

| Parameter | fast_mode=True | fast_mode=False |
|-----------|----------------|-----------------|
| `archetypal_method` | `'nmf'` | `'aa'` |
| `shapley_n_permutations` | 30 | 100 |
| `shapley_top_n` | 0.5 (50%) | None (all) |
| `shapley_compute_formative` | False | True |

**Speedup**: Approximately 30x faster than publication mode.

### Numba JIT Acceleration

JIT-compiled functions provide 2-5x speedup:

```python
@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_min_jit(X, Y):
    for i in prange(n):  # Parallel loop
        min_dist = np.inf
        for j in range(m):
            dist_sq = 0.0
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                dist_sq += diff * diff
            if dist_sq < min_dist:
                min_dist = dist_sq
        min_dists[i] = np.sqrt(min_dist)
    return min_dists
```

**Flags**:
- `nopython=True`: Pure machine code, no Python fallback
- `parallel=True`: Enable `prange` parallelization
- `cache=True`: Cache compiled function between runs
- `fastmath=True`: Allow floating-point optimizations

### FAISS Integration

For large datasets (n > 1,000):

```python
if FAISS_AVAILABLE and n > 1000:
    # Use FAISS for approximate nearest neighbor
    index = faiss.IndexFlatIP(d)  # Inner product (cosine after L2 norm)
    index.add(X_l2.astype(np.float32))
    D, I = index.search(query, k)
```

**Speedup**: 10-50x for similarity search on large datasets.

### Recommended Workflows

**Quick Exploration (< 1 minute)**:

```python
dt = DataTypical(
    fast_mode=True,
    shapley_mode=True,
    verbose=True
)
```

**Detailed Analysis (5-30 minutes)**:

```python
dt = DataTypical(
    fast_mode=False,
    shapley_mode=True,
    shapley_n_permutations=50,
    verbose=True
)
```

**Publication Quality (30+ minutes)**:

```python
dt = DataTypical(
    fast_mode=False,
    shapley_mode=True,
    shapley_n_permutations=100,
    deterministic=True,
    random_state=42,
    verbose=True
)
```

### Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Preprocessing | O(n × d) | O(n × d) |
| NMF | O(n × d × k × iter) | O(n × k + k × d) |
| PCHA | O(n × d × k × iter) | O(n × k + k × d) |
| Prototype selection | O(k × n²) | O(n²) |
| Shapley (formative) | O(perm × n² × V) | O(n × d) |
| Shapley (explanations) | O(n × perm × d × V) | O(n × d) |

where V is value function cost.

---

## Parameters Reference

### Core Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nmf_rank` | int | 8 | Number of archetypal basis vectors |
| `n_prototypes` | int | 20 | Number of prototypes to select |
| `scale` | str | "minmax" | Scaling method |
| `distance_metric` | str | "euclidean" | Distance metric for scores |
| `similarity_metric` | str | "cosine" | Similarity metric for prototypes |

### Computational Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deterministic` | bool | True | Enforce reproducibility |
| `n_jobs` | int | -1 | Parallel workers (-1 = all cores) |
| `max_iter_nmf` | int | 400 | NMF maximum iterations |
| `tol_nmf` | float | 1e-4 | NMF convergence tolerance |
| `dtype` | str | "float32" | Internal data type |
| `random_state` | int | 42 | Random seed |
| `max_memory_mb` | int | 2048 | Memory budget (MB) |

### Stereotype Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stereotype_column` | str | None | Target column name |
| `stereotype_target` | str/float | "max" | Target value ('min', 'max', or float) |
| `stereotype_keywords` | list | None | Keywords for text mode |
| `label_columns` | list | None | Columns to preserve as labels |

### Shapley Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shapley_mode` | bool | False | Enable Shapley analysis |
| `shapley_n_permutations` | int | 100 | Permutations for Shapley |
| `shapley_top_n` | int/float | None | Subsample for explanations |
| `shapley_early_stopping_patience` | int | 10 | Convergence patience |
| `shapley_early_stopping_tolerance` | float | 0.01 | Convergence threshold |
| `shapley_compute_formative` | bool | None | Compute formative instances |

### Performance Mode (v0.7)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fast_mode` | bool | False | Enable exploration mode |
| `archetypal_method` | str | None | 'nmf' or 'aa' (None = auto) |
| `speed_mode` | bool | False | Relax determinism for speed |

### Data Type Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_type` | str | None | Force data type ('tabular', 'text', 'graph') |
| `graph_topology_features` | list | None | Topology features to compute |
| `max_missing_frac` | float | 1.0 | Maximum allowed missingness |

---

## Thresholds and Constants

### Numeric Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| Epsilon (scaling) | 1e-12 | _score_with_fitted | Avoid division by zero |
| Epsilon (distance) | 1e-6 | ConvexHull weights | Numerical stability |
| Epsilon (PCHA shift) | 1e-10 | _fit_archetypal_aa | Ensure positivity |
| L2 norm floor | 1.0 | _l2_normalize_rows | Prevent zero division |

### Dimensional Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| ConvexHull max dim | 20 | Avoid segfaults in high-D (archetypal fitting fallback only; not used in formative computation from v0.7.2) |
| FAISS activation | n > 1000 | Use approximate search |
| Parallel threshold (samples) | n >= 20 | Avoid parallelization overhead |
| Parallel threshold (features) | d >= 10 | Avoid parallelization overhead |

### Algorithmic Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| ID-like uniqueness | 80% | Exclude likely identifiers |
| Percentile normalization | 95th | Robust distance scaling |
| Minimum permutations | 20 | Before early stopping check |
| Early stop minimum | 50 | Ensure adequate sampling |

### Memory Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Default memory budget | 2048 MB | Chunking limit |
| Similarity batch target | 256 MB | CELF scratch space |
| Small problem threshold | n × m < 100,000 | Direct vs chunked computation |

---

## Troubleshooting

### Common Issues

**Issue: "All Shapley values are zero"**

Cause: Insufficient samples for coalition evaluation.

Solution: Ensure n >= 20 samples.

**Issue: "Shapley values not converging"**

Cause: High variance in value function or insufficient permutations.

Solutions:
- Increase `shapley_n_permutations` (try 200)
- Increase `shapley_early_stopping_tolerance` (try 0.05)
- Check for NaN values in data

**Issue: "NMF not converging"**

Cause: Poorly conditioned data or high nmf_rank.

Solutions:
- Reduce `nmf_rank`
- Increase `max_iter_nmf`
- Check for constant features

**Issue: "ConvexHull segfault"**

Cause: High-dimensional data (> 20 features) passed to scipy ConvexHull.

Solution: From v0.7.2, ConvexHull is no longer used in the formative Shapley computation. It is used only as an optional fallback in archetypal fitting when PCHA is unavailable and n_features <= 20. DataTypical enforces a strict dimension check before any ConvexHull call, so this should not occur in normal use.

**Issue: "Memory error"**

Cause: Similarity matrix too large.

Solutions:
- Reduce dataset size
- Set `max_memory_mb` lower
- Disable `shapley_mode` during exploration
- Use `fast_mode=True`

**Issue: "Results not reproducible"**

Cause: Non-deterministic operations.

Solutions:
- Set `random_state`
- Set `deterministic=True`
- Set `speed_mode=False`
- Check for external threading (set OMP_NUM_THREADS=1)

### Performance Issues

**Slow fitting (> 1 hour)**

- Reduce `shapley_n_permutations` (30-50)
- Reduce `nmf_rank` (5-8)
- Reduce `n_prototypes` (10-20)
- Set `shapley_mode=False` for exploration
- Use `fast_mode=True`
- Verify `n_jobs=-1` is using all cores

**High memory usage**

- Reduce dataset size via sampling
- Reduce features via PCA
- Disable `shapley_mode`
- Process in batches
- Set `dtype='float32'`

### Diagnostic Information

Enable verbose output:

```python
dt = DataTypical(verbose=True)
dt.fit(data)

# Output includes:
# - Data type detection
# - Preprocessing steps
# - NMF/PCHA convergence
# - Prototype selection progress
# - Shapley convergence
# - Memory usage hints
```

Access settings after fit:

```python
print(dt.settings_)
# {'deterministic': True, 'speed_mode': False, 
#  'thread_limit': 1, 'random_state': 42, ...}
```

---

## Mathematical Foundations

### Non-negative Matrix Factorization

Given data matrix X ∈ ℝ^{n×d} with X ≥ 0, NMF finds:

$$\min_{W,H \geq 0} \frac{1}{2}\|X - WH\|_F^2$$

where W ∈ ℝ^{n×k} and H ∈ ℝ^{k×d}.

The Frobenius norm squared is:

$$\|X - WH\|_F^2 = \sum_{i,j} (X_{ij} - (WH)_{ij})^2$$

**Interpretation**: Each row w_i of W represents sample i's coefficients for the k basis vectors (rows of H).

### Principal Convex Hull Analysis

PCHA constrains archetypes to lie within the convex hull:

$$X \approx XC S$$

where:
- C ∈ ℝ^{n×k}: Convex combination coefficients for archetypes
- S ∈ ℝ^{k×n}: Sample memberships to archetypes

Constraints:
- C ≥ 0, columns of C sum to 1 (archetypes are convex combinations of data)
- S ≥ 0, columns of S sum to 1 (samples are convex combinations of archetypes)

### Facility Location Problem

Select k facilities to maximize coverage:

$$\max_{|F| \leq k} \sum_{i=1}^{n} w_i \cdot \max_{f \in F} s(i, f)$$

where s(i, f) is similarity between client i and facility f.

**Approximation Guarantee**: CELF achieves (1 - 1/e) ≈ 0.632 of optimal (submodular maximization).

### Shapley Values

For a coalitional game (N, v), the Shapley value of player i is:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

**Axiomatic Properties**:

1. **Efficiency**: $\sum_i \phi_i = v(N) - v(\emptyset)$
2. **Symmetry**: If v(S ∪ {i}) = v(S ∪ {j}) for all S, then φ_i = φ_j
3. **Null player**: If v(S ∪ {i}) = v(S) for all S, then φ_i = 0
4. **Linearity**: φ(v + w) = φ(v) + φ(w)

**Permutation-based Approximation**:

$$\hat{\phi}_i = \frac{1}{M} \sum_{m=1}^{M} [v(S_i^{\pi_m} \cup \{i\}) - v(S_i^{\pi_m})]$$

Convergence: O(1/√M) by CLT.

---

## References

### Archetypal Analysis

- Cutler, A., & Breiman, L. (1994). Archetypal analysis. Technometrics, 36(4), 338-347.
- Mørup, M., & Hansen, L. K. (2012). Archetypal analysis for machine learning and data mining. Neurocomputing, 80, 54-63.

### Non-negative Matrix Factorization

- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755), 788-791.
- Boutsidis, C., & Gallopoulos, E. (2008). SVD based initialization: A head start for nonnegative matrix factorization. Pattern recognition, 41(4), 1350-1362.

### Facility Location

- Cornuejols, G., Nemhauser, G. L., & Wolsey, L. A. (1990). The uncapacitated facility location problem. Discrete location theory, 119-171.
- Leskovec, J., et al. (2007). Cost-effective outbreak detection in networks. KDD.

### Shapley Values

- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games, 2(28), 307-317.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Castro, J., et al. (2009). Polynomial calculation of the Shapley value based on sampling. Computers & Operations Research, 36(5), 1726-1730.

### Implementation

- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
- Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.
- Lam, S. K., et al. (2015). Numba: A LLVM-based Python JIT compiler. LLVM workshop.

---

*DataTypical v0.7 - Computation Guide*
*A comprehensive reference for advanced users*
