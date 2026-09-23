"""
DataTypical v0.8.1 --- Dual-Perspective Significance with Shapley Explanations
===========================================================================

Revolutionary framework combining geometric and influence-based significance.

Key Innovation:
- Actual significance: Samples that ARE archetypal/prototypical/stereotypical (geometric)
- Formative instances: Samples that MAKE the dataset archetypal/prototypical/stereotypical (Shapley)
- Local explanations: WHY each sample is significant (feature attributions)

Two complementary perspectives:
1. LOCAL: "This sample IS significant because features X, Y contribute most"
2. GLOBAL: "This sample CREATES significance by defining the distribution and boundary"

What's new in v0.8.0 (bug fixes, plus the additions listed at the end):
- FIXED: archetypal_method='aa' no longer substitutes another method silently.
  Through v0.7.7, if py_pcha was not installed, 'aa' fell through to ConvexHull
  (itself skipped when n_features > 20) and then to NMF, with the only notice
  behind verbose=False. Fits reported as archetypal analysis could therefore
  return NMF output with nothing to distinguish it. 'aa' now raises ConfigError
  when PCHA is unavailable or fails.
    * New archetypal_method='auto' keeps the old permissive cascade
      PCHA -> ConvexHull -> NMF, but emits a RuntimeWarning at each downgrade.
    * New public attribute archetypal_backend_ records what actually ran:
      'pcha', 'convexhull' or 'nmf'. It is also stored in settings_.
    * py_pcha is now a declared dependency, so 'aa' works on a fresh install.
  Any 'aa' result produced by v0.7.7 or earlier in an environment without
  py_pcha is an NMF approximation and should be re-run.
- FIXED: UnboundLocalError in stereotypical Shapley explanations. The value
  function was defined inside the core-samples branch but also called from the
  secondary-samples branch, which crashed whenever core_samples was empty and
  secondary_samples was not, i.e. shapley_mode=True with a stereotype_column and
  shapley_top_n less than the number of rows. It never fired at
  shapley_top_n == len(df), which is why it went unnoticed.
- FIXED: a non-numeric stereotype_column (a Yes/No field, say) failed deep
  inside pandas with "could not convert string to float". The column is now
  validated where it is selected, ordered categoricals and booleans are encoded,
  and anything else raises ConfigError naming the column and the offending
  values.
- DOCS: noted that stereotypical_rank is a deterministic function of
  stereotype_column (Methods 2.3.1, eq. 26), so putting an outcome-derived
  quantity in that column and evaluating the rank on held-out rows is circular.

What's new in v0.7.7:
- Streaming formative-Shapley computation: the per-permutation coalition walk
  now updates each value function incrementally instead of re-evaluating it from
  scratch on every prefix. Output is numerically identical (rankings unchanged);
  only runtime improves.
    * archetypal / stereotypical: O(M*n^2) -> O(M*n) per fit
    * prototypical:               O(M*n^3) -> O(M*n^2) per fit
  In practice the formative step drops from hours to seconds at n=10,000, making
  publication-mode fits on large datasets tractable.
- Console output is now ASCII-only (no Unicode glyphs), so verbose logs and the
  test suites run cleanly under any terminal encoding (e.g. Windows cp1252).

What's new in v0.7.6:
- selected_significance parameter: compute only one significance type at a time
- Fixed prototype feature storage so transform() on new data uses correct vectors
- Full Shapley analysis (formative + explanations) now runs on text data paths
- Fixed iterator exhaustion in all text fit/transform methods
- Fixed local/global index mismatch in stereotypical Shapley explanations
- Improved error messages when a significance type was not fitted

What's new in v0.7:
- shapley_mode parameter (True/False)
- When True: computes explanations + formative instances
- Dual rankings: *_rank (actual) + *_shapley_rank (formative)
- Novel value functions: convex hull, coverage, extremeness
- Parallel Shapley computation with Option A (accurate v0.4 explanations)

All v0.6 features retained:
- Local explanations via get_shapley_explanations()
- Global explanations to identify formative instances

All v0.5 features retained:
- Tabular/Text/Graph support
- Label column preservation
- Graph topology features

All v0.4 features retained:
- User-configurable stereotypes

Sections:
  [A] Exceptions & Globals
  [B] Thread Control
  [C] Helpers (sparse/dense math)
  [D] Facility-Location (CELF, deterministic)
  [E] Shapley Significance Engine (NEW in v0.6)
  [F] DataTypical API
  [G] Graph Topology Features
  [H] Stereotype Computation
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Iterable, List, Optional, Dict, Tuple, Union, Callable

import heapq
import math
import gc
import warnings
import hashlib    
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

try:
    from numba import jit, prange
    import numpy as np
    
    # Test if Numba actually works with the current NumPy
    @jit(nopython=True)
    def _test(x): return x
    _test(np.array([1]))
    
    NUMBA_AVAILABLE = True
except (ImportError, Exception): 
    # This catches the "NumPy version too new" error too!
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        return lambda f: f
    def prange(n):
        return range(n)
        
try:
    import scipy.sparse as sp
except Exception:
    sp = None
ArrayLike = Union[np.ndarray, "sp.spmatrix"]

try:
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
except Exception:
    ConvexHull = None
    cdist = None

# py_pcha calls np.mat, which NumPy REMOVED in 2.0. Restoring the alias is
# what makes archetypal_method='aa' work on a current NumPy at all: without
# it a fresh install raises, and before v0.8.0 it fell through to ConvexHull
# silently, which is how published results were computed on the wrong
# backend without anyone noticing. np.mat is exactly np.asmatrix, so this
# restores a removed spelling rather than changing behaviour: PCHA returns
# bit-identical archetypes either way. Additive and guarded, so it cannot
# disturb an environment that already works.
if not hasattr(np, "mat") and hasattr(np, "asmatrix"):  # NumPy >= 2.0
    np.mat = np.asmatrix

try:
    from py_pcha import PCHA
except ImportError:
    PCHA = None

# ============================================================
# [A] Exceptions & Globals
# ============================================================
class DataTypicalError(Exception):
    pass

class MemoryBudgetError(DataTypicalError):
    pass

class ConfigError(DataTypicalError):
    pass

def _seed_everything(seed: int) -> None:
    np.random.seed(seed)


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman correlation between two score vectors, without pulling in scipy.

    Used for the split-half reliability of a Shapley estimate: rank each half
    and correlate the ranks. Returns 0.0 when either side is constant, which is
    the honest reading of "the two halves agree on nothing".
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3 or a.size != b.size:
        return float("nan")
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 3:
        return float("nan")

    def _ranks(v: np.ndarray) -> np.ndarray:
        # Ties must share a rank. A plain argsort-of-argsort hands a constant
        # vector the ranks 0, 1, 2, ..., which then correlates perfectly with
        # anything increasing, so a degenerate half would report itself as
        # having converged. That is the exact failure this diagnostic exists
        # to catch, so it must not commit it itself.
        order = np.argsort(v, kind="mergesort")
        out = np.empty(v.size, dtype=np.float64)
        out[order] = np.arange(v.size, dtype=np.float64)
        sorted_v = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            if j > i:
                out[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return out

    ra = _ranks(a[good])
    rb = _ranks(b[good])
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(np.dot(ra, ra)) * float(np.dot(rb, rb)))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(ra, rb) / denom)


def _stereotype_values_as_float(values: pd.Series, column: Optional[str] = None) -> np.ndarray:
    """
    Coerce a stereotype source column to float64, naming the column on failure.

    v0.8.0: ``_compute_stereotypical_rank`` previously called
    ``.to_numpy(dtype=np.float64)`` with no check, so a reasonable request such as
    ``stereotype_column='Menopause'`` on a Yes/No column failed several frames
    below the caller with ``could not convert string to float: 'No'`` and no
    mention of which column was at fault.

    Ordered categoricals are encoded by their category order and booleans become
    0.0/1.0. Unordered categoricals and free-text columns raise ``ConfigError``:
    stereotypical significance ranks by distance to a target, so an ordering is
    required, and inventing one is the caller's decision to make explicitly.
    """
    name = column if column is not None else getattr(values, "name", None)
    label = f"stereotype_column '{name}'" if name is not None else "the stereotype column"

    if isinstance(values.dtype, pd.CategoricalDtype):
        if values.dtype.ordered:
            codes = values.cat.codes.to_numpy()
            out = codes.astype(np.float64)
            out[codes < 0] = np.nan
            return out
        raise ConfigError(
            f"{label} is an unordered categorical with categories "
            f"{list(values.dtype.categories)}. Stereotypical significance ranks by "
            "distance to a target, which needs an ordering. Rebuild the column with "
            "pd.Categorical(..., ordered=True), or map it to numbers before fitting."
        )

    if pd.api.types.is_bool_dtype(values.dtype) or pd.api.types.is_numeric_dtype(values.dtype):
        return values.to_numpy(dtype=np.float64)

    converted = pd.to_numeric(values, errors="coerce")
    unreadable = values.notna().to_numpy() & converted.isna().to_numpy()
    if not unreadable.any():
        return converted.to_numpy(dtype=np.float64)

    sample = [repr(v) for v in pd.unique(values[unreadable])[:4]]
    raise ConfigError(
        f"{label} has dtype '{values.dtype}' and could not be read as numbers "
        f"(values include {', '.join(sample)}). Stereotypical significance ranks by "
        "distance to a numeric target, so the column must be numeric, boolean, or an "
        "ordered categorical. Encode it before fitting, for example "
        "df['col'] = df['col'].map({'No': 0, 'Yes': 1})."
    )


# ============================================================
# [B] Thread Control
# ============================================================
class _ThreadControl:
    def __init__(self, deterministic: bool = True):
        self.deterministic = deterministic
        self._ctx = None
        self.effective_limit = None

    def __enter__(self):
        if self.deterministic:
            self._ctx = threadpool_limits(limits=1)
            self.effective_limit = 1
        else:
            self._ctx = threadpool_limits(limits=None)
            self.effective_limit = None
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)
            

# ============================================================
# [C] Helpers (sparse/dense math)
# ============================================================
def _cleanup_memory(*arrays, force_gc: bool = False) -> None:
    """
    Explicitly delete arrays and optionally force garbage collection.
    
    MEMORY OPTIMIZED: Python's GC doesn't always free memory immediately.
    This forces cleanup of large temporaries to reduce peak memory usage.
    """
    for arr in arrays:
        if arr is not None:
            del arr
    
    if force_gc:
        gc.collect()


def _l2_normalize_rows_dense(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms



def _sparse_l2_normalize_rows(X: "sp.spmatrix") -> "sp.spmatrix":
    if sp is None:
        raise ImportError("scipy is required for sparse operations.")
    if not sp.isspmatrix_csr(X):
        X = X.tocsr(copy=False)
    sq = X.multiply(X).sum(axis=1)
    norms = np.sqrt(np.maximum(np.asarray(sq).ravel(), 0.0))
    norms[norms == 0.0] = 1.0
    D = sp.diags(1.0 / norms)
    return D @ X


def _sparse_minmax_0_1_nonneg(M: "sp.spmatrix") -> "sp.spmatrix":
    if sp is None:
        raise ImportError("scipy is required for sparse operations.")
    if not sp.isspmatrix(M):
        raise TypeError("Expected a scipy.sparse matrix.")
    A = M.tocsc(copy=False)
    # CRITICAL: Must use .toarray() to convert sparse matrix to dense
    col_max = A.max(axis=0).toarray().ravel()
    col_max[col_max == 0.0] = 1.0
    return (A @ sp.diags(1.0 / col_max)).tocsr()


def _chunk_len(n_left: int, n_right: int, bytes_per: int, max_memory_mb: int) -> int:
    if max_memory_mb <= 0:
        raise MemoryBudgetError("max_memory_mb must be positive")
    max_bytes = max_memory_mb * 1024 * 1024
    return max(1, min(n_right, int(max_bytes // max(8, n_left * bytes_per))))
    

def _ensure_dtype(X: np.ndarray, dtype: str = 'float32') -> np.ndarray:
    """
    Ensure array has specified dtype, converting if necessary.
    
    MEMORY OPTIMIZED: Default to float32 (4 bytes) instead of float64 (8 bytes).
    """
    target_dtype = np.float32 if dtype == 'float32' else np.float64
    
    if X.dtype != target_dtype:
        return X.astype(target_dtype, copy=False)
    return X


def _euclidean_min_to_set_dense(
    X: np.ndarray, Y: np.ndarray, max_memory_mb: int = 2048
) -> np.ndarray:
    """
    Compute minimum Euclidean distance from each row of X to any row in Y.
    
    OPTIMIZED: Uses Numba JIT for 2-3x speedup and better memory efficiency.
    """
    n, d = X.shape
    m = Y.shape[0]
    
    # For small problems, use JIT-compiled direct computation
    if n * m < 100000:
        return _euclidean_min_jit(X, Y)
    
    # For large problems, use chunked computation with JIT
    best = np.full(n, np.inf, dtype=np.float64)
    block = _chunk_len(n, m, bytes_per=8, max_memory_mb=max_memory_mb)
    
    # Pre-compute X squared norms once
    x2 = np.sum(X * X, axis=1)
    
    for s in range(0, m, block):
        e = min(m, s + block)
        YY = Y[s:e]
        
        # Use JIT-compiled function for this chunk
        chunk_dists = _euclidean_chunk_jit(X, YY, x2)
        best = np.minimum(best, chunk_dists)
    
    return best


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_min_jit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    JIT-compiled minimum Euclidean distance computation.
    
    Uses parallel loops for multi-core acceleration.
    """
    n = X.shape[0]
    m = Y.shape[0]
    d = X.shape[1]
    
    min_dists = np.empty(n, dtype=np.float64)
    
    # Parallel loop over X samples (explicit with prange)
    for i in prange(n):  # Changed from range to prange
        min_dist = np.inf
        
        for j in range(m):
            dist_sq = 0.0
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                dist_sq += diff * diff
            
            if dist_sq < min_dist:
                min_dist = dist_sq
        
        min_dists[i] = np.sqrt(max(min_dist, 0.0))
    
    return min_dists


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _euclidean_chunk_jit(
    X: np.ndarray,
    Y_chunk: np.ndarray,
    x2: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled chunked distance computation using pre-computed norms.
    
    Computes: sqrt(||x||^2 + ||y||^2 - 2<x,y>) efficiently.
    """
    n = X.shape[0]
    m = Y_chunk.shape[0]
    d = X.shape[1]
    
    min_dists = np.empty(n, dtype=np.float64)
    
    # Pre-compute Y chunk squared norms
    y2 = np.empty(m, dtype=np.float64)
    for j in range(m):
        y2_val = 0.0
        for k in range(d):
            y2_val += Y_chunk[j, k] * Y_chunk[j, k]
        y2[j] = y2_val
    
    # Parallel loop over X samples
    for i in prange(n):
        min_dist_sq = np.inf

        for j in range(m):
            # Compute dot product
            dot = 0.0
            for k in range(d):
                dot += X[i, k] * Y_chunk[j, k]
            
            # Distance squared using pre-computed norms
            dist_sq = x2[i] + y2[j] - 2.0 * dot
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        min_dists[i] = np.sqrt(max(min_dist_sq, 0.0))
    
    return min_dists

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _pairwise_euclidean_jit(X: np.ndarray) -> np.ndarray:
    """
    JIT-compiled pairwise Euclidean distance matrix.
    
    Returns upper triangle only to save memory.
    """
    n = X.shape[0]
    d = X.shape[1]
    
    # Compute full distance matrix (symmetric)
    dists = np.zeros((n, n), dtype=np.float64)
    
    # Parallel outer loop
    for i in prange(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                dist_sq += diff * diff
            
            dist = np.sqrt(max(dist_sq, 0.0))
            dists[i, j] = dist
            dists[j, i] = dist  # Symmetric
    
    return dists


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _cosine_similarity_jit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    JIT-compiled cosine similarity between L2-normalized vectors.
    
    For L2-normalized data, this is just the dot product.
    """
    n = X.shape[0]
    m = Y.shape[0]
    d = X.shape[1]
    
    sims = np.empty((n, m), dtype=np.float64)
    
    # Parallel loop over X samples
    for i in prange(n):
        for j in range(m):
            dot = 0.0
            for k in range(d):
                dot += X[i, k] * Y[j, k]
            sims[i, j] = max(dot, 0.0)  # Clip negative similarities
    
    return sims


# ============================================================
# [D] Facility-Location (CELF, deterministic)
# ============================================================
@dataclass
class FacilityLocationSelector:
    def __init__(self, n_prototypes=10, deterministic=True, speed_mode=False, verbose=False):
        self.n_prototypes = int(n_prototypes)
        self.deterministic = bool(deterministic)
        self.speed_mode = bool(speed_mode)
        self.verbose = bool(verbose)
        
    def select(self, X_l2, weights=None, forbidden=None):
        """
        Deterministic CELF for facility-location with:
          - content-based tie-breaking (perm-invariant),
          - optional client weights (e.g., density),
          - optional forbidden candidate set (still count as clients).
        Expects rows to be L2-normalized. Works with dense or sparse input.
        Returns: (selected_indices, marginal_gains)
        """
        import numpy as np, heapq, hashlib
    
        # --- dense float64 view
        if sp is not None and sp.isspmatrix(X_l2):
            X = X_l2.toarray().astype(np.float64, copy=False)
        else:
            X = np.asarray(X_l2, dtype=np.float64)
        n = X.shape[0]
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
    
        # --- client weights (normalize to mean 1 for scale stability)
        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).ravel()
            m = float(w.mean())
            w = w / m if m > 0 else np.ones_like(w)
    
        # --- forbidden candidates (excluded from selection, included as clients)
        forb = np.zeros(n, dtype=bool)
        if forbidden is not None:
            forb_idx = np.asarray(list(forbidden), dtype=int)
            forb_idx = forb_idx[(forb_idx >= 0) & (forb_idx < n)]
            forb[forb_idx] = True
    
        # --- target number of prototypes (cap to available candidates)
        k_req = int(getattr(self, "n_prototypes", min(10, n)))
        available = n - int(forb.sum())
        k = max(0, min(k_req, available))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
    
        # --- permutation-invariant tie-breaker: hash of row content
        def row_key(i: int) -> int:
            h = hashlib.blake2b(X[i].tobytes(), digest_size=8)
            return int.from_bytes(h.digest(), "big", signed=False)
        keys = np.fromiter((row_key(i) for i in range(n)), dtype=np.uint64, count=n)
    
        # --- CELF init
        best = np.zeros(n, dtype=np.float64)          # current best similarity per client
        last_eval = np.full(n, -1, dtype=np.int64)    # last #selected when gain was computed
        last_gain = np.zeros(n, dtype=np.float64)
    
        # Initial exact gains: g0[c] = sum_i w_i * max(0, <x_i, x_c>)
        g0 = np.zeros(n, dtype=np.float64)
        # block multiply to limit memory
        target_bytes = 256 * 1024 * 1024  # 256MB scratch
        item = np.dtype(np.float64).itemsize
        max_b = max(1, int(target_bytes // max(1, n * item)))
        bsz = max(1, min(n, max_b))
        XT = X.T
        for s in range(0, n, bsz):
            e = min(n, s + bsz)
            S = X[s:e] @ XT           # (e-s, n)
            np.maximum(S, 0.0, out=S)
            g0 += (w[s:e, None] * S).sum(axis=0, dtype=np.float64)
    
        last_gain[:] = g0
        last_eval[:] = 0
    
        # heap items: (-gain_estimate, key, idx) -- ties broken by content key
        heap = [(-float(g0[c]), int(keys[c]), int(c)) for c in range(n) if not forb[c]]
        heapq.heapify(heap)
    
        selected: list[int] = []
        gains: list[float] = []
        it = 0
        while len(selected) < k and heap:
            neg_g_est, _, c = heapq.heappop(heap)
            if last_eval[c] == it:
                # accept candidate
                selected.append(c)
                gains.append(float(last_gain[c]))
                s = X @ X[c]
                np.maximum(s, 0.0, out=s)
                np.maximum(best, s, out=best)
                it += 1
                continue
            # refresh exact marginal gain vs current 'best'
            s = X @ X[c]
            improv = s - best
            improv[improv < 0.0] = 0.0
            g_exact = float((w * improv).sum(dtype=np.float64))
            last_gain[c] = g_exact
            last_eval[c] = it
            heapq.heappush(heap, (-g_exact, int(keys[c]), int(c)))
    
        return np.asarray(selected, dtype=int), np.asarray(gains, dtype=float)


# ============================================================
# [E] Shapley Significance Engine (NEW in v0.6)
# ============================================================

class ShapleyEarlyStopping:
    """Early stopping for Shapley convergence using relative change."""
    
    def __init__(self, patience: int = 10, tolerance: float = 0.01):
        self.patience = patience
        self.tolerance = tolerance
        self.history = []
        self.stable_count = 0
        
    def update(self, shapley_estimates: np.ndarray, n_perms: int) -> Tuple[bool, Dict]:
        if n_perms < 20:
            return False, {'converged': False, 'n_permutations': n_perms}
        
        self.history.append(shapley_estimates.copy())
        
        if len(self.history) < 2:
            return False, {'converged': False, 'n_permutations': n_perms}
        
        old = self.history[-2]
        new = self.history[-1]
        
        denom = np.abs(old) + 1e-12
        rel_change = np.abs(new - old) / denom
        max_rel_change = np.max(rel_change)
        mean_rel_change = np.mean(rel_change)
        
        if mean_rel_change < self.tolerance:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        should_stop = self.stable_count >= self.patience
        
        info = {
            'converged': should_stop,
            'n_permutations': n_perms,
            'mean_rel_change': float(mean_rel_change),
            'max_rel_change': float(max_rel_change),
            'stable_iterations': self.stable_count
        }
        
        return should_stop, info


@jit(nopython=True, cache=True, fastmath=True)
def _compute_marginals_jit(
    perm: np.ndarray,
    values: np.ndarray,
    n_samples: int,
    n_features: int
) -> np.ndarray:
    """
    JIT-compiled function to compute Shapley marginal contributions.
    
    This is the performance-critical inner loop - compiled to machine code by Numba.
    
    Parameters
    ----------
    perm : array of sample indices in permutation order
    values : array of value function results for each coalition size
    n_samples : number of samples
    n_features : number of features
    
    Returns
    -------
    shapley_contrib : (n_samples, n_features) array of marginal contributions
    """
    shapley_contrib = np.zeros((n_samples, n_features), dtype=np.float64)
    
    for j in range(n_samples):
        sample_idx = perm[j]
        marginal = values[j+1] - values[j]
        
        # Broadcast marginal across all features
        for f in range(n_features):
            shapley_contrib[sample_idx, f] = marginal / n_features
    
    return shapley_contrib


@jit(nopython=True, cache=True, fastmath=True)
def _compute_feature_marginals_jit(
    perm: np.ndarray,
    values: np.ndarray,
    n_features: int
) -> np.ndarray:
    """
    JIT-compiled function to compute feature-level Shapley marginal contributions.
    
    Parameters
    ----------
    perm : array of feature indices in permutation order
    values : array of value function results for each feature coalition size
    n_features : number of features
    
    Returns
    -------
    shapley_contrib : (n_features,) array of marginal contributions
    """
    shapley_contrib = np.zeros(n_features, dtype=np.float64)
    
    for j in range(n_features):
        feat_idx = perm[j]
        marginal = values[j+1] - values[j]
        shapley_contrib[feat_idx] = marginal
    
    return shapley_contrib
    

class ShapleySignificanceEngine:
    """
    Compute Shapley values for dual-perspective significance analysis.
    
    Supports two modes:
    1. Explanations: Why is this sample archetypal/prototypical/stereotypical?
    2. Formative: Which samples create the archetypal/prototypical/stereotypical structure?
    """
    
    def __init__(
        self,
        n_permutations: int = 100,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_patience: int = 10,
        early_stopping_tolerance: float = 0.01,
        verbose: bool = False
    ):
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tolerance = early_stopping_tolerance
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
        
    def compute_shapley_values(
        self,
        X: np.ndarray,
        value_function: Callable,
        value_function_name: str = "unknown",
        context: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute Shapley values using specified value function.
        
        OPTIMIZED: Uses shared memory for parallel processing to avoid data copying.
        MEMORY OPTIMIZED: Cleanup batch results immediately after accumulation.
        """
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"\n  Computing {value_function_name}...")
            print(f"    Samples: {n_samples}, Features: {n_features}")
            print(f"    Max permutations: {self.n_permutations}")
        
        early_stop = ShapleyEarlyStopping(
            patience=self.early_stopping_patience,
            tolerance=self.early_stopping_tolerance
        )
        
        shapley_sum = np.zeros((n_samples, n_features), dtype=np.float64)
        # v0.8.0: a second accumulator over every other permutation, so the
        # estimate can report its own reliability. Splitting the permutations
        # in half and correlating the two independent estimates is the
        # cheapest honest answer to 'has this converged', and it needs no
        # extra sampling.
        shapley_half = np.zeros((n_samples, n_features), dtype=np.float64)
        n_perms_used = 0
        
        batch_size = 10
        info = {'converged': False, 'mean_rel_change': 0.0}
        
        # OPTIMIZED: Decide parallelization strategy based on data size
        use_parallel = self.n_jobs != 1 and n_samples >= 20
        
        # OPTIMIZED: For small datasets or single-threaded, use direct computation
        if not use_parallel:
            for batch_start in range(0, self.n_permutations, batch_size):
                batch_end = min(batch_start + batch_size, self.n_permutations)
                batch_perms = [self.rng.permutation(n_samples) for _ in range(batch_end - batch_start)]
                
                for perm in batch_perms:
                    shapley_contrib = self._process_single_permutation(perm, X, value_function, context)
                    shapley_sum += shapley_contrib
                    if n_perms_used % 2 == 0:
                        shapley_half += shapley_contrib
                    n_perms_used += 1
                
                # MEMORY CLEANUP: Free batch permutations immediately
                _cleanup_memory(batch_perms)
                
                current_estimate = shapley_sum / n_perms_used
                should_stop, info = early_stop.update(current_estimate, n_perms_used)
                
                if should_stop and n_perms_used >= 50:
                    if self.verbose:
                        print(f"    Early stop at {n_perms_used} perms (change: {info['mean_rel_change']:.6f})")
                    break
        else:
            # OPTIMIZED: Use threading backend for better memory sharing
            for batch_start in range(0, self.n_permutations, batch_size):
                batch_end = min(batch_start + batch_size, self.n_permutations)
                batch_perms = [self.rng.permutation(n_samples) for _ in range(batch_end - batch_start)]
                
                # Use threading backend for shared memory access
                batch_results = Parallel(
                    n_jobs=self.n_jobs, 
                    backend='threading',
                    verbose=0
                )(
                    delayed(self._process_single_permutation)(perm, X, value_function, context)
                    for perm in batch_perms
                )
                
                # Accumulate results efficiently
                for shapley_contrib in batch_results:
                    shapley_sum += shapley_contrib
                    if n_perms_used % 2 == 0:
                        shapley_half += shapley_contrib
                    n_perms_used += 1
                
                # MEMORY CLEANUP: Free batch results and permutations immediately
                _cleanup_memory(batch_results, batch_perms)
                
                current_estimate = shapley_sum / n_perms_used
                should_stop, info = early_stop.update(current_estimate, n_perms_used)
                
                if should_stop and n_perms_used >= 50:
                    if self.verbose:
                        print(f"    Early stop at {n_perms_used} perms (change: {info['mean_rel_change']:.6f})")
                    break
        
        Phi = shapley_sum / n_perms_used
        
        # Verify additivity
        all_indices = np.arange(n_samples)
        if context is not None:
            total_actual = value_function(X, all_indices, context)
        else:
            total_actual = value_function(X, all_indices)
        
        total_from_shapley = np.sum(Phi)
        additivity_error = abs(total_from_shapley - total_actual) / (abs(total_actual) + 1e-12)
        
        # v0.8.0: split-half reliability of the RANKING this estimate produces.
        # The additivity error above says the values sum to the right total; it
        # says nothing about whether the order of the samples is reproducible,
        # and the order is what gets reported as the formative instances.
        # Measured on synthetic data, the default 100 permutations gives an
        # archetypal formative ranking indistinguishable from chance, so this
        # number belongs in front of the user.
        _other_half = shapley_sum - shapley_half
        _split_rho = _rank_correlation(shapley_half.sum(axis=1),
                                       _other_half.sum(axis=1))

        info = {
            'n_permutations_used': n_perms_used,
            'converged': info.get('converged', False) if n_perms_used < self.n_permutations else True,
            'mean_rel_change': info.get('mean_rel_change', 0.0),
            'additivity_error': float(additivity_error),
            'total_shapley': float(total_from_shapley),
            'total_actual': float(total_actual),
            'split_half_rho': _split_rho,
        }

        if np.isfinite(_split_rho) and _split_rho < 0.7 and n_perms_used >= 4:
            warnings.warn(
                f"The Shapley estimate for {value_function_name} has not "
                f"converged: splitting the {n_perms_used} permutations in half "
                f"gives two rankings that correlate only {_split_rho:.2f}. The "
                "values sum correctly, but the ORDER of the samples, which is "
                "what gets reported as the formative instances, is largely "
                "sampling noise at this permutation count. Raise "
                "shapley_n_permutations substantially and check that this "
                "number rises with it before reading anything into the ranking.",
                RuntimeWarning, stacklevel=3
            )
        
        if self.verbose:
            print(f"    {n_perms_used} perms, additivity error: {additivity_error:.6f}")
        
        # MEMORY CLEANUP: Free shapley_sum before returning Phi (they're different objects)
        _cleanup_memory(shapley_sum, shapley_half, _other_half)
        
        return Phi, info
    
    
    def compute_feature_shapley_values(
        self,
        X: np.ndarray,
        value_function: Callable,
        value_function_name: str = "unknown",
        context: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute feature-level Shapley values for each sample.
        
        OPTIMIZED: Uses threading backend for better memory sharing.
        """
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"\n  Computing feature-level {value_function_name}...")
            print(f"    Samples: {n_samples}, Features: {n_features}")
            print(f"    Max permutations: {self.n_permutations}")
        
        early_stop = ShapleyEarlyStopping(
            patience=self.early_stopping_patience,
            tolerance=self.early_stopping_tolerance
        )
        
        shapley_sum = np.zeros((n_samples, n_features), dtype=np.float64)
        n_perms_used = 0
        
        batch_size = 10
        info = {'converged': False, 'mean_rel_change': 0.0}
        
        # OPTIMIZED: Decide parallelization strategy
        use_parallel = self.n_jobs != 1 and n_features >= 10
        
        for batch_start in range(0, self.n_permutations, batch_size):
            batch_end = min(batch_start + batch_size, self.n_permutations)
            
            # Generate feature permutations for this batch
            batch_perms = [self.rng.permutation(n_features) for _ in range(batch_end - batch_start)]
            
            # Process each sample
            for sample_idx in range(n_samples):
                if use_parallel:
                    # OPTIMIZED: Threading backend for memory sharing
                    batch_results = Parallel(
                        n_jobs=self.n_jobs, 
                        backend='threading',
                        verbose=0
                    )(
                        delayed(self._process_feature_permutation)(
                            sample_idx, perm, X, value_function, value_function_name, context
                        )
                        for perm in batch_perms
                    )
                else:
                    # Direct computation for small problems
                    batch_results = [
                        self._process_feature_permutation(
                            sample_idx, perm, X, value_function, value_function_name, context
                        )
                        for perm in batch_perms
                    ]
                
                for shapley_contrib in batch_results:
                    shapley_sum[sample_idx, :] += shapley_contrib
                    
            n_perms_used += len(batch_perms)
            
            current_estimate = shapley_sum / n_perms_used
            should_stop, info = early_stop.update(current_estimate, n_perms_used)
            
            if should_stop and n_perms_used >= 50:
                if self.verbose:
                    print(f"    Early stop at {n_perms_used} perms (change: {info['mean_rel_change']:.6f})")
                break
        
        Phi = shapley_sum / n_perms_used
        
        # Compute additivity error
        total_errors = []
        for sample_idx in range(n_samples):
            shapley_total = np.sum(Phi[sample_idx, :])
            if context is not None:
                actual_value = value_function(X[sample_idx:sample_idx+1, :], np.array([sample_idx]), context)
            else:
                actual_value = value_function(X[sample_idx:sample_idx+1, :], np.array([sample_idx]))
            
            error = abs(shapley_total - actual_value) / (abs(actual_value) + 1e-12)
            total_errors.append(error)
        
        additivity_error = np.mean(total_errors)
        
        info_out = {
            'n_permutations_used': n_perms_used,
            'converged': info.get('converged', False) if n_perms_used < self.n_permutations else True,
            'mean_rel_change': info.get('mean_rel_change', 0.0),
            'additivity_error': float(additivity_error)
        }
        
        if self.verbose:
            print(f"    {n_perms_used} perms, mean additivity error: {additivity_error:.6f}")
        
        return Phi, info_out
    
    
    def _process_single_permutation(
        self,
        perm: np.ndarray,
        X: np.ndarray,
        value_function: Callable,
        context: Optional[Dict]
    ) -> np.ndarray:
        """
        Process one permutation to compute marginal contributions.
        
        OPTIMIZED: Delegates to JIT-compiled helper for massive speedup.
        """
        n_samples, n_features = X.shape

        # FAST PATH: formative value functions are decomposable along the
        # coalition prefix, so the whole vector of prefix values is produced
        # in one streaming pass (O(n) value work per permutation instead of
        # O(n) full re-evaluations). Results are identical to the loop below.
        prefix_fn = _PREFIX_VALUE_FUNCS.get(value_function)
        if prefix_fn is not None:
            values = prefix_fn(perm, X, context)
            return _compute_marginals_jit(perm, values, n_samples, n_features)

        # Generic path: re-evaluate the value function on every prefix.
        # (can't JIT this part due to the Python callable)
        values = np.zeros(n_samples + 1, dtype=np.float64)
        values[0] = 0.0

        for j in range(n_samples):
            subset_indices = perm[:j+1]
            X_subset = X[subset_indices]

            if context is not None:
                values[j+1] = value_function(X_subset, subset_indices, context)
            else:
                values[j+1] = value_function(X_subset, subset_indices)

        # Now use JIT-compiled function to compute marginal contributions
        shapley_contrib = _compute_marginals_jit(perm, values, n_samples, n_features)

        return shapley_contrib

    def _process_feature_permutation(
        self,
        sample_idx: int,
        perm: np.ndarray,
        X: np.ndarray,
        value_function: Callable,
        metric_name: str,
        context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Process one feature permutation for a single sample to compute per-feature contributions.
        
        OPTIMIZED: Uses JIT-compiled helper for faster computation.
        """
        n_features = X.shape[1]
        shapley_contrib = np.zeros(n_features, dtype=np.float64)
        
        # Compute all value function calls first (can't JIT this part)
        values = np.zeros(n_features + 1, dtype=np.float64)
        values[0] = 0.0
        
        for j in range(n_features):
            feature_subset = perm[:j+1]
            X_sample_subset = X[sample_idx:sample_idx+1, :][:, feature_subset]
            
            if context is not None:
                values[j+1] = value_function(X_sample_subset, np.array([sample_idx]), context)
            else:
                values[j+1] = value_function(X_sample_subset, np.array([sample_idx]))
        
        # Use JIT-compiled function to compute marginals
        shapley_contrib = _compute_feature_marginals_jit(perm, values, n_features)
        
        return shapley_contrib

# ============================================================
# Value Functions for Formative Instance Discovery
# ============================================================

def formative_archetypal_convex_hull(
    X_subset: np.ndarray,
    indices: np.ndarray,
    context: Optional[Dict] = None
) -> float:
    """
    Archetypal formative value function: Convex hull volume.
    
    Samples that expand the convex hull boundary are formative archetypes.
    
    SAFE: Falls back to range-based metric in high dimensions to avoid segfaults.
    """
    if len(X_subset) < 3:
        return 0.0
    
    n_samples, n_features = X_subset.shape
    
    # CRITICAL FIX: ConvexHull segfaults in high dimensions (>20D)
    # Always use safe fallback for high-dimensional data
    if n_features > 20 or ConvexHull is None or n_samples < n_features + 1:
        # Safe fallback: Feature range coverage (no segfault risk)
        ranges = X_subset.max(axis=0) - X_subset.min(axis=0)
        return float(np.prod(ranges + 1e-10))  # Product of ranges (volume proxy)
    
    # Low dimensions: Try ConvexHull with safety wrapper
    try:
        # Ensure data is float64 for numerical stability
        X_clean = np.asarray(X_subset, dtype=np.float64)
        
        # Remove duplicate points (causes ConvexHull to fail)
        X_unique = np.unique(X_clean, axis=0)
        
        if len(X_unique) < n_features + 1:
            # Not enough unique points for hull in this dimension
            ranges = X_unique.max(axis=0) - X_unique.min(axis=0)
            return float(np.prod(ranges + 1e-10))
        
        hull = ConvexHull(X_unique)
        return float(hull.volume)
        
    except Exception:
        # ConvexHull failed - use safe fallback
        ranges = X_subset.max(axis=0) - X_subset.min(axis=0)
        return float(np.prod(ranges + 1e-10))
        

def exact_formative_archetypal(
    X: np.ndarray,
    archetypes: np.ndarray
) -> np.ndarray:
    """
    Exact Shapley values for the archetypal formative game, without sampling.

    The value function is

        v(S) = -mean over archetypes of min distance from that archetype to S,
        v(empty) = 0,

    which is a mean of "minimum" games, one per archetype. Shapley is linear in
    the value function, so the answer is the mean of the per-archetype answers,
    and each of those has a closed form.

    For one archetype, sort the samples by distance. A sample contributes only
    when every closer sample is absent from the coalition it joins, which gives,
    for the sample at ascending rank r,

        phi_r = -d_r / n  +  sum over k > r of (d_k - d_r) / ((k + 1) * k)

    Both sums are suffix sums, so the whole thing is O(n log n) per archetype
    after the sort. Verified against exhaustive coalition enumeration to machine
    precision for several shapes.

    Why this exists (v0.8.0): the Monte Carlo estimator is unbiased but converges
    very slowly on this game. At the default 100 permutations two runs differing
    only in the seed produced formative rankings correlating 0.02, and the
    estimate was out by roughly twice the largest true value. This function
    removes the sampling entirely.

    Parameters
    ----------
    X : np.ndarray
        Sample matrix, shape (n_samples, n_features), in the same space as the
        archetypes.
    archetypes : np.ndarray
        Archetype matrix, shape (n_archetypes, n_features), normally ``H_``.

    Returns
    -------
    np.ndarray
        Exact Shapley value per sample, shape (n_samples,).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    archetypes = np.ascontiguousarray(archetypes, dtype=np.float64)
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)
    if n_samples == 1:
        # the single sample carries the whole value of the grand coalition
        diffs = archetypes - X[0]
        return np.array([-float(np.mean(np.sqrt((diffs ** 2).sum(axis=1))))])

    diffs = archetypes[:, None, :] - X[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))       # (n_archetypes, n_samples)

    # weights 1/((k+1)k) for k >= 1, with the k = 0 slot unused
    k = np.arange(n_samples, dtype=np.float64)
    weights = np.zeros(n_samples, dtype=np.float64)
    weights[1:] = 1.0 / ((k[1:] + 1.0) * k[1:])

    phi = np.zeros(n_samples, dtype=np.float64)
    for row in dists:
        order = np.argsort(row, kind="mergesort")
        ds = row[order]

        # suffix sums over k > r
        weighted = ds * weights
        suffix_wd = np.concatenate([np.cumsum(weighted[::-1])[::-1][1:], [0.0]])
        suffix_w = np.concatenate([np.cumsum(weights[::-1])[::-1][1:], [0.0]])

        phi_sorted = -ds / n_samples + suffix_wd - ds * suffix_w
        contribution = np.empty(n_samples, dtype=np.float64)
        contribution[order] = phi_sorted
        phi += contribution

    return phi / dists.shape[0]


def exact_formative_prototypical(
    X: np.ndarray,
    block: int = 512
) -> np.ndarray:
    """
    Exact Shapley values for the prototypical formative game, without sampling.

    The value function, matching ``formative_prototypical_coverage`` exactly
    including its clip at zero, is

        v(S) = (1/|S|) * sum over i in S of max(0, max over j in S, j != i, of
               cosine(x_i, x_j)),        v(empty) = v(single) = 0

    This is a mean of maxima, which does not reduce the way a mean of minimum
    games does. It decomposes instead. Sort the neighbours of i by descending
    similarity and keep the m_i with similarity above zero. The maximum is
    attained at the FIRST of those present in the coalition, so

        max(0, max_j s_ij) = sum over r of s_i,(r) * 1[l_(r) in S,
                                                      l_(1..r-1) not in S]

    and therefore

        v = sum over i, r of s_i,(r) * u(A, B),
            A = {i, l_(r)},  B = {l_(1), ..., l_(r-1)},
            u(A, B)(S) = 1[A subset of S, B disjoint from S] / |S|

    Shapley is linear in the value function, so the answer is the same weighted
    sum of the Shapley values of those little games, and each of those depends
    only on |A| (always 2), |B|, and whether the player is in A, in B, or in
    neither. Writing w(s) = s!(n-s-1)!/n! and p = n - 2 - |B|:

        in A       sum over s of C(p, s-1) w(s) / (s+1)
        in B      -sum over s of C(p, s-2) w(s) / s
        neither    sum over s of C(p-1, s-2) w(s) * (1/(s+1) - 1/s)

    Accumulating the "neither" term once as a scalar and the "in B" term as a
    suffix sum makes the whole thing O(n^2) after the per-row sorts, against
    O(2^n) for enumeration. Verified against exhaustive enumeration to machine
    precision on random, duplicated, antipodal, zero-row and all-identical
    inputs.

    Why this exists (v0.8.0): this game is the least convergent of the three
    under sampling. On the Wine dataset at the documented 100 permutations,
    five seeds produced five different top formative prototypes, with pairwise
    rank correlations of 0.03 to 0.17 and a top-ten overlap of 0 or 1 out of 10
    in nine of the ten seed pairs. A published ranking from that estimator is
    a property of the seed.

    Cost. O(n^2 log n) time and O(n * block) memory. The n by n similarity
    matrix is never formed: it would be 16 GB at n = 45,000. Rows are computed
    a block at a time and discarded, so memory is not the limit, time is.
    Measured on the released code: 0.6 s at n = 500, 7 s at n = 2000, 21 s at
    n = 5000, 94 s at n = 10,000, 372 s at n = 19,000. That is cheap next to sampling at the sizes
    where a ranking is read instance by instance, and expensive at very large
    n, where it is still the only way to get an answer that does not move when
    the seed does.
    Pass ``formative_method='monte_carlo'`` if the cost is not worth it, and
    read ``split_half_rho`` to see what that estimate is worth.

    Parameters
    ----------
    X : np.ndarray
        Sample matrix, shape (n_samples, n_features). Rows are L2 normalised
        internally, exactly as the sampled value function does, and a zero row
        is left as zero rather than divided by zero.
    block : int
        Row block size for the coefficient tables, trading memory for speed.

    Returns
    -------
    np.ndarray
        Exact Shapley value per sample, shape (n_samples,).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    # v0.8.1: a NaN cell makes that row's norm NaN, and the selection step
    # `vals > 0.0` compares False against NaN, so the affected similarities
    # were DISCARDED rather than propagated. The function then returned an
    # all-finite, fully rankable vector computed from whichever cells happened
    # to be clean, and the NaN-bearing sample took an ordinary position in the
    # ranking. Both sibling exact functions propagate NaN; this one hid it.
    if not np.all(np.isfinite(X)):
        n_bad = int(np.count_nonzero(~np.isfinite(X)))
        raise DataTypicalError(
            "exact_formative_prototypical received %d non-finite value(s). "
            "The selection step drops them rather than propagating them, so "
            "the result would be a confident finite ranking computed from an "
            "unknown subset of the data. Remove or impute them deliberately, "
            "or use formative_method='monte_carlo'." % n_bad)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Z = X / norms

    coef_a, coef_b, coef_c = _prototypical_coefficients(n, block=block)

    phi = np.zeros(n, dtype=np.float64)
    everyone = 0.0
    # The similarity matrix is n^2, which is 16 GB at n = 45,000, so it is never
    # formed. Rows are computed a block at a time and discarded, which leaves
    # the time cost unchanged and the memory cost at O(n * block).
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = Z[start:stop] @ Z.T
        for local, i in enumerate(range(start, stop)):
            row = sims[local].copy()
            self_sim = row[i]
            row[i] = -np.inf                  # i sorts last, then gets dropped
            order = np.argsort(-row, kind="stable")[: n - 1]
            vals = sims[local][order]
            m = int(np.count_nonzero(vals > 0.0))
            del self_sim
            if m == 0:
                continue
            order, vals = order[:m], vals[:m]
            ranks = np.arange(m)
            c = coef_c[ranks]

            everyone += float(vals @ c)
            gain = vals * (coef_a[ranks] - c)
            phi[i] += float(gain.sum())
            phi[order] += gain                # indices within one row are unique

            # the p-th better neighbour sits in B for every layer after p
            tail = vals * (coef_b[ranks] - c)
            suffix = np.empty(m, dtype=np.float64)
            suffix[:-1] = np.cumsum(tail[::-1])[::-1][1:]
            suffix[-1] = 0.0
            phi[order] += suffix
        del sims

    return phi + everyone


def _prototypical_coefficients(n: int, block: int = 512):
    """
    Shapley coefficients for u(A, B) with |A| = 2, for every |B| in 0..n-2.

    Computed in blocks of |B| so the peak memory is O(block * n) rather than
    O(n^2), which matters well before the time cost does.
    """
    from scipy.special import gammaln

    sizes = np.arange(n, dtype=np.float64)
    log_w = gammaln(sizes + 1) + gammaln(n - sizes) - gammaln(n + 1)

    def _term(top, k):
        """
        C(top, k) * w(s), computed as a single exponential.

        The two factors must never be formed separately. C(5000, 2500) is about
        10^1504, which overflows to inf, and w(s) underflows to zero, so the
        product comes out as nan rather than the O(1) number it actually is.
        Adding the logarithms first keeps every intermediate in range.
        """
        top, k = np.broadcast_arrays(np.asarray(top, dtype=np.float64),
                                     np.asarray(k, dtype=np.float64))
        out = np.zeros(top.shape, dtype=np.float64)
        ok = (k >= 0) & (k <= top) & (top >= 0)
        if not np.any(ok):
            return out
        log_binom = (gammaln(top[ok] + 1) - gammaln(k[ok] + 1)
                     - gammaln(top[ok] - k[ok] + 1))
        out[ok] = np.exp(log_binom + np.broadcast_to(log_w, top.shape)[ok])
        return out

    coef_a = np.empty(n - 1, dtype=np.float64)
    coef_b = np.empty(n - 1, dtype=np.float64)
    coef_c = np.empty(n - 1, dtype=np.float64)

    safe = np.where(sizes > 0, sizes, 1.0)
    for start in range(0, n - 1, block):
        stop = min(start + block, n - 1)
        b = np.arange(start, stop, dtype=np.float64)[:, None]
        pool = n - 2 - b
        coef_a[start:stop] = (_term(pool, sizes - 1) / (sizes + 1)).sum(axis=1)
        coef_b[start:stop] = -(_term(pool, sizes - 2) / safe).sum(axis=1)
        coef_c[start:stop] = (_term(pool - 1, sizes - 2)
                              * (1.0 / (sizes + 1) - 1.0 / safe)).sum(axis=1)

    for name, arr in (("a", coef_a), ("b", coef_b), ("c", coef_c)):
        if not np.all(np.isfinite(arr)):
            raise DataTypicalError(
                "the prototypical Shapley coefficients went non-finite at "
                "n=%d (table %s). This is a numerical failure, not a property "
                "of the data; please report it." % (n, name)
            )
    return coef_a, coef_b, coef_c


def exact_formative_stereotypical(
    target_values: np.ndarray,
    target: Union[str, float],
    median: Optional[float] = None
) -> np.ndarray:
    """
    Exact Shapley values for the stereotypical formative game, without sampling.

    The value function is a plain mean over the coalition, which has a closed
    form. For ``target='max'`` it is

        v(S) = (1/|S|) * sum over i in S of c_i,   c_i = max(0, v_i - median)
        v(empty) = 0

    Every coalition size carries weight 1/n, the marginal at size 0 is c_i, and
    at size s >= 1 it is (c_i - mbar_i)/(s + 1) in expectation, where mbar_i is
    the mean of c over the other players. Summing over s gives

        phi_i = (1/n) * [ c_i + (c_i - mbar_i) * (H_n - 1) ]

    with H_n the nth harmonic number. ``'min'`` is the same with the sign of the
    deviation flipped. A numeric target adds a constant on every non-empty
    coalition, contributing A/n each, minus a mean game on the distances.

    Verified against exhaustive coalition enumeration for all three target
    modes, and the values satisfy efficiency exactly.

    .. note::

       The result is an **affine function of c_i**, so this ranking is a monotone
       transform of the stereotype column and carries no information beyond it.
       That is a property of the value function, not of this implementation: the
       Monte Carlo estimate is a noisy version of the same thing. It means the
       stereotypical dual-perspective plot has both axes driven by one column.

    Parameters
    ----------
    target_values : np.ndarray
        The stereotype column, one value per sample.
    target : str or float
        ``'max'``, ``'min'``, or a numeric target.
    median : float, optional
        Median of ``target_values``; computed if not supplied.

    Returns
    -------
    np.ndarray
        Exact Shapley value per sample.
    """
    values = np.asarray(target_values, dtype=np.float64).ravel()
    n_samples = values.size
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)
    if median is None:
        median = float(np.median(values))

    def _mean_game(c: np.ndarray) -> np.ndarray:
        if n_samples == 1:
            return c.copy()
        harmonic = float(np.sum(1.0 / np.arange(1, n_samples + 1)))
        others_mean = (float(c.sum()) - c) / (n_samples - 1)
        return (c + (c - others_mean) * (harmonic - 1.0)) / n_samples

    if isinstance(target, str):
        if target == "max":
            return _mean_game(np.maximum(values - median, 0.0))
        if target == "min":
            return _mean_game(np.maximum(median - values, 0.0))
        raise ConfigError(
            f"target must be 'min', 'max' or numeric, got {target!r}"
        )

    numeric_target = float(target)
    constant = abs(median - numeric_target)
    return constant / n_samples - _mean_game(np.abs(values - numeric_target))


def formative_archetypal_pcha_cached(
    X_subset: np.ndarray,
    indices: np.ndarray,
    context: Dict
) -> float:
    """
    Archetypal formative value function using cached archetype geometry.

    Measures how well a subset of samples supports the pre-computed archetypal
    geometry (PCHA or NMF archetypes stored in H_). Value equals the negative
    mean minimum distance from each archetype to its nearest sample in the subset.

    Using cached archetypes ensures that both the actual significance axis
    (samples that ARE archetypal) and the formative significance axis (samples
    that CREATE archetypal structure) reference the same geometric model. This
    preserves the scientific integrity of the dual-perspective scatter plot.

    Replaces formative_archetypal_convex_hull, which required ConvexHull refitting
    on every permutation subset. ConvexHull complexity is O(n^(d/2)), making it
    intractable for d > 8. This function uses O(n_archetypes x n_subset x n_features)
    distance arithmetic, which is fast regardless of dimensionality.

    Parameters
    ----------
    X_subset : np.ndarray
        Feature matrix for samples in the current subset, shape (n_subset, n_features).
    indices : np.ndarray
        Indices of samples in the subset (unused, retained for API consistency).
    context : dict
        Must contain key 'archetypes': np.ndarray of shape (n_archetypes, n_features),
        taken from H_ of the fitted DataTypical model.

    Returns
    -------
    float
        Negative mean minimum distance from each archetype to its nearest subset
        sample. Higher (less negative) values indicate better archetypal coverage.
    """
    archetypes = context['archetypes']  # (n_archetypes, n_features)

    if len(X_subset) < 1:
        return 0.0

    # Pairwise squared distances: (n_archetypes, n_subset)
    # Broadcasting: archetypes (n_arch, 1, n_feat) - X_subset (1, n_sub, n_feat)
    diffs = archetypes[:, np.newaxis, :] - X_subset[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))  # (n_archetypes, n_subset)

    # For each archetype, find the distance to its nearest subset member
    min_dists = dists.min(axis=1)  # (n_archetypes,)

    # Negative mean distance: higher = better coverage of archetypal corners
    return float(-np.mean(min_dists))
    

def formative_prototypical_coverage(
    X_subset: np.ndarray,
    indices: np.ndarray,
    context: Optional[Dict] = None
) -> float:
    """
    Prototypical formative value function: Coverage/representativeness.
    
    Samples that maximize pairwise similarity coverage are formative prototypes.
    """
    if len(X_subset) < 2:
        return 0.0
    
    # L2 normalize
    norms = np.linalg.norm(X_subset, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X_l2 = X_subset / norms
    
    # Pairwise cosine similarities
    similarities = X_l2 @ X_l2.T
    np.fill_diagonal(similarities, 0)
    
    if similarities.size == 0:
        return 0.0
    
    max_sims = np.max(similarities, axis=1) if similarities.shape[0] > 0 else np.array([0.0])
    return float(np.mean(max_sims))


def formative_stereotypical_extremeness(
    X_subset: np.ndarray,
    indices: np.ndarray,
    context: Dict
) -> float:
    """
    Stereotypical formative value function: Extremeness from median.
    
    Samples that pull the distribution toward the target are formative stereotypes.
    """
    if len(X_subset) == 0:
        return 0.0
    
    target_values = context['target_values']
    target = context['target']
    median = context.get('median', np.median(target_values))
    
    subset_vals = target_values[indices]
    
    if target == 'max':
        # How far above median?
        extremeness = np.mean(np.maximum(subset_vals - median, 0))
    elif target == 'min':
        # How far below median?
        extremeness = np.mean(np.maximum(median - subset_vals, 0))
    else:
        # How much closer to target than median?
        target_val = float(target)
        median_dist = abs(median - target_val)
        subset_dist = np.mean(np.abs(subset_vals - target_val))
        extremeness = median_dist - subset_dist
    
    return float(extremeness)


# ============================================================
# [D2] Streaming (prefix) value functions for formative Shapley
# ============================================================
#
# Monte-Carlo Shapley walks each permutation through coalitions of size
# 1, 2, ..., n and needs the value-function result at every prefix. The
# generic path (ShapleySignificanceEngine._process_single_permutation)
# re-evaluates the value function from scratch on each prefix, costing
# O(n) full evaluations per permutation -- i.e. O(M * n^2) (or worse)
# overall, which is the dominant cost of publication-mode fits.
#
# All three formative value functions are decomposable along a growing
# coalition, so the full vector of prefix values can be produced in a
# single streaming pass that is mathematically identical to the
# per-prefix re-evaluation:
#
#   archetypal   : value = -mean_k min_{i in S} dist(archetype_k, x_i).
#                  Adding a point can only lower a running per-archetype
#                  minimum -> maintain min_dist[k] and update in O(k*d).
#   stereotypical: value = mean over S of a per-sample scalar term ->
#                  a running cumulative mean, O(1) per added point.
#   prototypical : value = mean over S of each member's max cosine
#                  similarity to another member (diagonal treated as 0).
#                  Adding a point can only raise an existing member's best
#                  similarity -> maintain best[i] and a running sum.
#
# Each kernel returns ``values`` of length n+1 with values[0] = 0.0,
# matching the empty-coalition baseline hard-coded in the generic path.

@jit(nopython=True, cache=True)
def _prefix_archetypal_kernel(perm, X, archetypes):
    n = X.shape[0]
    d = X.shape[1]
    k = archetypes.shape[0]
    values = np.zeros(n + 1, dtype=np.float64)
    min_dist = np.full(k, np.inf)
    for j in range(n):
        xi = perm[j]
        for a in range(k):
            s = 0.0
            for f in range(d):
                diff = archetypes[a, f] - X[xi, f]
                s += diff * diff
            dist = np.sqrt(s)
            if dist < min_dist[a]:
                min_dist[a] = dist
        msum = 0.0
        for a in range(k):
            msum += min_dist[a]
        values[j + 1] = -(msum / k)
    return values


@jit(nopython=True, cache=True)
def _prefix_cummean_kernel(perm, terms):
    n = terms.shape[0]
    values = np.zeros(n + 1, dtype=np.float64)
    cum = 0.0
    for j in range(n):
        cum += terms[perm[j]]
        values[j + 1] = cum / (j + 1)
    return values


@jit(nopython=True, cache=True)
def _prefix_prototypical_kernel(perm, X_l2):
    n = X_l2.shape[0]
    d = X_l2.shape[1]
    values = np.zeros(n + 1, dtype=np.float64)
    best = np.zeros(n, dtype=np.float64)   # diagonal baseline of 0.0
    bsum = 0.0                             # running sum of best[0..j]
    for j in range(1, n):
        pj = perm[j]
        newbest = 0.0
        for t in range(j):
            pt = perm[t]
            s = 0.0
            for f in range(d):
                s += X_l2[pj, f] * X_l2[pt, f]
            if s > best[t]:
                bsum += s - best[t]
                best[t] = s
            if s > newbest:
                newbest = s
        best[j] = newbest
        bsum += newbest
        values[j + 1] = bsum / (j + 1)
    return values


def _prefix_values_archetypal(perm, X, context):
    archetypes = np.ascontiguousarray(context['archetypes'], dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    return _prefix_archetypal_kernel(perm.astype(np.int64), X, archetypes)


def _prefix_values_prototypical(perm, X, context=None):
    X = np.ascontiguousarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X_l2 = np.ascontiguousarray(X / norms, dtype=np.float64)
    return _prefix_prototypical_kernel(perm.astype(np.int64), X_l2)


def _prefix_values_stereotypical(perm, X, context):
    target_values = np.asarray(context['target_values'], dtype=np.float64)
    target = context['target']
    median = context.get('median', float(np.median(target_values)))
    perm64 = perm.astype(np.int64)

    if target == 'max':
        terms = np.maximum(target_values - median, 0.0)
        return _prefix_cummean_kernel(perm64, terms)
    elif target == 'min':
        terms = np.maximum(median - target_values, 0.0)
        return _prefix_cummean_kernel(perm64, terms)
    else:
        target_val = float(target)
        median_dist = abs(median - target_val)
        terms = np.abs(target_values - target_val)
        cummean = _prefix_cummean_kernel(perm64, terms)
        values = median_dist - cummean
        values[0] = 0.0   # empty coalition baseline (matches generic path)
        return values


# Registry consulted by ShapleySignificanceEngine._process_single_permutation.
# Keyed by value-function identity so only the formative path is affected.
_PREFIX_VALUE_FUNCS = {
    formative_archetypal_pcha_cached:   _prefix_values_archetypal,
    formative_prototypical_coverage:    _prefix_values_prototypical,
    formative_stereotypical_extremeness: _prefix_values_stereotypical,
}


# ============================================================
# [E] DataTypical API
# ============================================================
@dataclass
class DataTypical:
    # ---- Core Config ----
    nmf_rank: int = 8
    n_prototypes: int = 20
    scale: str = "minmax"
    distance_metric: str = "euclidean"
    similarity_metric: str = "cosine"
    deterministic: bool = True
    n_jobs: int = -1
    max_iter_nmf: int = 400
    tol_nmf: float = 1e-4
    feature_weights: Optional[np.ndarray] = None
    speed_mode: bool = False
    dtype: str = "float32"
    random_state: int = 42
    max_memory_mb: int = 2048
    return_ranks_only: bool = False
    auto_n_prototypes: Optional[str] = None
    verbose: bool = False
    max_missing_frac: float = 1.0

    # ---- Stereotype Configuration (NEW in v0.4) ----
    # NOTE (v0.8.0): stereotypical_rank is a deterministic, monotone function of
    # stereotype_column alone -- see Methods 2.3.1, eq. 26:
    #     s_stereo_i = 1 - |y_i - tau| / max_j |y_j - tau|
    # It carries no information beyond the distance from the target on that one
    # column. Placing an outcome-derived quantity in stereotype_column and then
    # evaluating the resulting rank on held-out rows is therefore circular, and
    # will return a perfect correlation that means nothing. This is by design:
    # stereotypical significance depends on an external specification of what
    # counts as interesting, unlike archetypal and prototypical significance,
    # which derive from the intrinsic geometry of the data.
    stereotype_column: Optional[str] = None
    stereotype_target: Union[str, float] = "max"
    label_columns: Optional[List[str]] = None
    stereotype_keywords: Optional[List[str]] = None
    graph_topology_features: Optional[List[str]] = None

    # ---- Data Type Configuration (NEW in v0.5) ----
    data_type: Optional[str] = None
    
    # ---- Shapley Configuration (NEW in v0.6) ----
    shapley_mode: bool = False
    shapley_n_permutations: int = 100
    shapley_top_n: Optional[Union[int, float]] = None  # CHANGED: Now supports float
    shapley_early_stopping_patience: int = 10
    shapley_early_stopping_tolerance: float = 0.01
    shapley_compute_formative: Optional[bool] = None  # NEW in v0.7: None = auto from fast_mode
    # v0.8.0: how the formative Shapley values are computed.
    #   'exact'       (default) the closed form where one exists: no sampling,
    #                 seed-independent, O(n log n)
    #   'monte_carlo' permutation sampling everywhere, the pre-0.8.0 behaviour
    # All three games now have exact algorithms. The ARCHETYPAL game is a mean
    # of per-archetype minimum games, the STEREOTYPICAL game is a plain mean
    # game, and the PROTOTYPICAL game is a mean of maxima that decomposes into
    # weighted indicator games. Costs are O(n log n), O(n log n) and O(n^2).
    #
    # Why this is the default: at the documented 100 permutations the sampled
    # archetypal ranking does not converge. Two fits differing only in
    # random_state correlated 0.02 on synthetic data and 0.19 on a real cohort,
    # and the sampled ranking correlates 0.40 to 0.55 with the exact answer.
    # Pass formative_method='monte_carlo' to reproduce a pre-0.8.0 result.
    formative_method: str = "exact"

    # ---- Performance Mode (NEW in v0.7) ----
    fast_mode: bool = False
    archetypal_method: Optional[str] = None

    # ---- Significance Selection ----
    # None = compute all three; "archetypal" / "prototypical" / "stereotypical" = one only
    selected_significance: Optional[str] = None

    # ---- Artifacts ----
    W_: Optional[np.ndarray] = field(default=None, init=False)
    H_: Optional[np.ndarray] = field(default=None, init=False)
    reconstruction_error_: Optional[float] = field(default=None, init=False)

    n_archetypes_: Optional[int] = field(default=None, init=False)
    # v0.8.0: which backend actually produced the archetypes, so a completed fit
    # can be audited: 'pcha', 'convexhull' or 'nmf'. See _fit_archetypal_aa.
    archetypal_backend_: Optional[str] = field(default=None, init=False)
    prototype_indices_: Optional[np.ndarray] = field(default=None, init=False)
    prototype_rows_: Optional[np.ndarray] = field(default=None, init=False)
    prototype_features_: Optional[np.ndarray] = field(default=None, init=False)
    prototype_features_l2_: Optional[np.ndarray] = field(default=None, init=False)
    marginal_gains_: Optional[np.ndarray] = field(default=None, init=False)
    assignments_: Optional[np.ndarray] = field(default=None, init=False)
    coverage_: Optional[np.ndarray] = field(default=None, init=False)
    knee_: Optional[int] = field(default=None, init=False)

    scaler_: Optional[MinMaxScaler] = field(default=None, init=False)
    vectorizer_: Optional[TfidfVectorizer] = field(default=None, init=False)
    nmf_model_: Optional[NMF] = field(default=None, init=False)

    settings_: Dict = field(default_factory=dict, init=False)
    ideals_: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    dropped_columns_: List[str] = field(default_factory=list, init=False)
    missingness_: Dict[str, float] = field(default_factory=dict, init=False)
    train_index_: Optional[pd.Index] = field(default=None, init=False)

    # Feature selection for tables (numeric-only)
    feature_columns_: Optional[List[str]] = field(default=None, init=False)
    impute_median_: Optional[np.ndarray] = field(default=None, init=False)
    keep_mask_: Optional[np.ndarray] = field(default=None, init=False)

    # NEW in v0.4: Stereotype artifacts
    _df_original_fit: Optional[pd.DataFrame] = field(default=None, init=False)
    label_df_: Optional[pd.DataFrame] = field(default=None, init=False)
    text_metadata_: Optional[pd.DataFrame] = field(default=None, init=False)
    stereotype_keyword_scores_: Optional[np.ndarray] = field(default=None, init=False)
    graph_topology_df_: Optional[pd.DataFrame] = field(default=None, init=False)
        
    # Data type detection (NEW in v0.5)
    _detected_data_type: Optional[str] = field(default=None, init=False)

    # ---- Shapley Artifacts (NEW in v0.6) ----
    Phi_archetypal_explanations_: Optional[np.ndarray] = field(default=None, init=False)
    Phi_prototypical_explanations_: Optional[np.ndarray] = field(default=None, init=False)
    Phi_stereotypical_explanations_: Optional[np.ndarray] = field(default=None, init=False)

    Phi_archetypal_formative_: Optional[np.ndarray] = field(default=None, init=False)
    Phi_prototypical_formative_: Optional[np.ndarray] = field(default=None, init=False)
    Phi_stereotypical_formative_: Optional[np.ndarray] = field(default=None, init=False)

    shapley_info_: Dict = field(default_factory=dict, init=False)
    _stereotype_source_fit_: Optional[pd.Series] = field(default=None, init=False)


    # --------------------------
    # Auto-Detection and Routing
    # --------------------------
    
    def _auto_detect_data_type(self, X, **kwargs) -> str:
        """
        Auto-detect data type based on input format.
        
        Priority:
        1. Graph: If edges/edge_index parameter present
        2. Text: If X is list/tuple of strings
        3. Tabular: If X is DataFrame or array
        
        Parameters
        ----------
        X : various
            Input data
        **kwargs
            Additional parameters (checked for edges/edge_index)
        
        Returns
        -------
        data_type : str
            One of 'graph', 'text', 'tabular'
        
        Raises
        ------
        ValueError
            If data type cannot be determined
        """
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
        
        # Cannot determine
        raise ValueError(
            f"Cannot auto-detect data type from input of type {type(X)}. "
            f"Supported formats: DataFrame/array (tabular), list of strings (text), "
            f"or provide edges parameter (graph). "
            f"Alternatively, specify data_type='tabular'/'text'/'graph' explicitly."
        )
    
    def _validate_data_type(self, detected: str) -> str:
        """
        Validate and resolve data_type configuration.
        
        If data_type is specified in config, validate it matches expected values.
        Otherwise use auto-detected type.
        
        Parameters
        ----------
        detected : str
            Auto-detected data type
        
        Returns
        -------
        data_type : str
            Final data type to use
        
        Raises
        ------
        ValueError
            If configured data_type is invalid
        """
        if self.data_type is not None:
            # Manual override provided
            valid_types = {'tabular', 'text', 'graph'}
            if self.data_type not in valid_types:
                raise ValueError(
                    f"Invalid data_type='{self.data_type}'. "
                    f"Must be one of {valid_types} or None (auto-detect)."
                )
            if self.verbose:
                if detected != self.data_type:
                    print(f"Using configured data_type='{self.data_type}' "
                          f"(auto-detected: '{detected}')")
            return self.data_type
        else:
            # Use auto-detected
            if self.verbose:
                print(f"Auto-detected data_type: '{detected}'")
            return detected
    
    def _reset_fit_state(self) -> None:
        """
        Clear everything a previous fit left behind.

        v0.8.0: several attributes were written only on the path that produces
        them and read back with `hasattr` or an is-not-None check, so they
        survived into the next fit on the same object:

        * refitting on clean data still reported the previous fit's
          `dropped_columns_`;
        * refitting with `stereotype_column=None` kept the previous stereotype
          source;
        * fitting text after tabular left `_df_original_fit`,
          `feature_columns_` and `keep_mask_` pointing at the tabular frame, so
          `profile_plot` and `heatmap` would have described the wrong data
          instead of refusing.

        Called from every public fit entry point, before any work.
        """
        self.dropped_columns_ = []
        self.missingness_ = {}
        self._df_original_fit = None
        self.label_df_ = None
        self.text_metadata_ = None
        self.stereotype_keyword_scores_ = None
        self.graph_topology_df_ = None
        self._stereotype_source_fit_ = None

        self.feature_columns_ = None
        self.keep_mask_ = None
        self.impute_median_ = None
        self.scaler_ = None
        self.vectorizer_ = None

        if hasattr(self, "_union_core_samples"):
            del self._union_core_samples

    def _apply_fast_mode_defaults(self) -> None:
        """
        Apply fast_mode preset defaults if parameters not explicitly set.
        
        fast_mode=True:  Exploration (NMF + explanations only + subsample)
        fast_mode=False: Publication (AA + formative + full dataset)
        
        Users can override any individual parameter by setting it explicitly.

        v0.8.0: this used to run once per object, guarded by an
        `_fast_mode_applied` flag, so changing `fast_mode` and refitting left
        the previous mode's presets in place. It now runs on every fit. A value
        this method filled in itself is released first, and only when it still
        holds exactly what was put there, so an explicit choice made in between
        is never overwritten.
        """
        _previous = getattr(self, "_autofilled_presets_", {})
        for _name, _value in _previous.items():
            if getattr(self, _name) == _value:
                setattr(self, _name,
                        100 if _name == "shapley_n_permutations" else None)
        _autofilled = {}

        if self.fast_mode:
            # Fast mode defaults (exploration)
            if self.archetypal_method is None:
                self.archetypal_method = 'nmf'
                _autofilled["archetypal_method"] = 'nmf'
            
            # Reduce Shapley permutations for speed
            if self.shapley_n_permutations == 100:  # Default value, not overridden
                self.shapley_n_permutations = 30
                _autofilled["shapley_n_permutations"] = 30
            
            # Subsample explanations to top 50%
            if self.shapley_top_n is None:
                self.shapley_top_n = 0.5  # 50% of instances
                _autofilled["shapley_top_n"] = 0.5
            
            # Skip formative in fast mode (explanations only)
            if self.shapley_compute_formative is None:
                self.shapley_compute_formative = False
                _autofilled["shapley_compute_formative"] = False
                
        else:
            # Publication mode defaults (rigorous)
            if self.archetypal_method is None:
                self.archetypal_method = 'aa'  # True archetypal analysis
                _autofilled["archetypal_method"] = 'aa'
            
            # Keep shapley_n_permutations=100 (default)
            # Keep shapley_top_n=None (compute for all instances) 
            # Compute formative in publication mode
            if self.shapley_compute_formative is None:
                self.shapley_compute_formative = True
                _autofilled["shapley_compute_formative"] = True
        
        self._autofilled_presets_ = _autofilled

        # v0.8.0: scale, distance_metric and similarity_metric were declared as
        # parameters but never read anywhere in the module. Any value was
        # accepted, including nonsense, and the pipeline went on using MinMax
        # scaling, Euclidean distance and cosine similarity regardless. A caller
        # asking for standardised features, or a cosine distance, silently got
        # neither. Rather than change the numerics in a bug-fix release, the
        # implemented value is now the only one accepted, so the parameter can
        # no longer claim something the code does not do.
        # v0.8.0: only the exact string 'kneedle' did anything. Any other
        # value, including a near miss like 'knee', was accepted and then
        # ignored, so a caller who asked for automatic selection quietly got
        # none.
        if self.auto_n_prototypes not in (None, "kneedle"):
            raise ConfigError(
                f"auto_n_prototypes={self.auto_n_prototypes!r} is not "
                "recognised. Use 'kneedle' for automatic selection, or None to "
                "keep n_prototypes as given. Earlier versions accepted any "
                "value here and then ignored it."
            )

        if self.formative_method not in ("monte_carlo", "exact"):
            raise ConfigError(
                f"formative_method={self.formative_method!r} is not recognised. "
                "Use 'monte_carlo' for permutation sampling, the default, or "
                "'exact' for the closed-form archetypal formative values."
            )

        _implemented = {
            "scale": ("minmax", "MinMax scaling to [0, 1]"),
            "distance_metric": ("euclidean", "Euclidean distance"),
            "similarity_metric": ("cosine", "cosine similarity"),
        }
        for _name, (_only, _what) in _implemented.items():
            _value = getattr(self, _name)
            if _value != _only:
                raise ConfigError(
                    f"{_name}={_value!r} is not implemented. DataTypical uses "
                    f"{_what} throughout, and only {_name}={_only!r} is "
                    f"accepted. Earlier versions accepted any value here and "
                    f"then ignored it, so a fit that asked for {_value!r} was "
                    f"silently computed with {_only!r}."
                )

        # v0.8.0: max_memory_mb only reached _chunk_len on the chunked path,
        # so a nonsensical budget was accepted in silence on anything small
        # enough to skip chunking, then raised much later on a larger dataset.
        if not isinstance(self.max_memory_mb, (int, float)) or self.max_memory_mb <= 0:
            raise ConfigError(
                f"max_memory_mb must be a positive number of megabytes, got "
                f"{self.max_memory_mb!r}."
            )

        # Validate archetypal_method
        if self.archetypal_method not in ['nmf', 'aa', 'auto']:
            raise ValueError(
                f"archetypal_method must be 'nmf', 'aa' or 'auto', "
                f"got '{self.archetypal_method}'"
            )
        
        if self.verbose:
            mode_name = "Fast" if self.fast_mode else "Publication"
            print(f"\n{mode_name} mode defaults:")
            print(f"  archetypal_method: {self.archetypal_method}")
            print(f"  shapley_n_permutations: {self.shapley_n_permutations}")
            print(f"  shapley_top_n: {self.shapley_top_n if self.shapley_top_n else 'all instances'}")
            print(f"  shapley_compute_formative: {self.shapley_compute_formative}")

    # --------------------------
    # Unified Interface
    # --------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray, List[str]], **kwargs):
        """
        Fit DataTypical on input data (auto-detects format).
        
        Automatically detects whether input is tabular, text, or graph data
        based on format and parameters. Can be overridden with data_type parameter.
        
        Parameters
        ----------
        X : DataFrame, array, or list of strings
            Input data:
            - Tabular: DataFrame or 2D array
            - Text: List of string documents
            - Graph: Node features (with edges parameter)
        
        **kwargs : optional
            Additional parameters for specific data types:
            
            For text:
                vectorizer : str, default 'tfidf'
                text_metadata : pd.DataFrame, optional
            
            For graph:
                edges : np.ndarray (required for graph detection)
                    Edge list as (2, n_edges) or (n_edges, 2)
                edge_index : np.ndarray (alias for edges)
                compute_topology : bool, default True
        
        Returns
        -------
        self : DataTypical
            Fitted estimator
        
        Examples
        --------
        >>> # Tabular (auto-detected)
        >>> dt = DataTypical()
        >>> dt.fit(dataframe)
        
        >>> # Text (auto-detected from list of strings)
        >>> dt = DataTypical(stereotype_keywords=['protein'])
        >>> dt.fit(corpus)
        
        >>> # Graph (auto-detected from edges parameter)
        >>> dt = DataTypical(graph_topology_features=['degree'])
        >>> dt.fit(node_features, edges=edge_list)
        
        >>> # Manual override
        >>> dt = DataTypical(data_type='tabular')
        >>> dt.fit(data)
        """
        # v0.8.0: both of these run on every fit. The presets used to be
        # applied once per object, so changing fast_mode and refitting left the
        # previous mode in place, and nothing cleared the state a previous fit
        # had written.
        self._reset_fit_state()
        self._apply_fast_mode_defaults()
        
        # Auto-detect data type
        detected = self._auto_detect_data_type(X, **kwargs)
        
        # Validate and resolve final type
        final_type = self._validate_data_type(detected)
        self._detected_data_type = final_type
        
        # Route to appropriate internal method
        if final_type == 'tabular':
            return self._fit_tabular(X)
        elif final_type == 'text':
            vectorizer = kwargs.get('vectorizer', 'tfidf')
            text_metadata = kwargs.get('text_metadata', None)
            return self._fit_text(X, vectorizer, text_metadata)
        elif final_type == 'graph':
            edges = kwargs.get('edges', kwargs.get('edge_index', None))
            compute_topology = kwargs.get('compute_topology', True)
            return self._fit_graph(X, edges, compute_topology)
        else:
            raise RuntimeError(f"Unknown data type: {final_type}")
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray, List[str]], **kwargs):
        """
        Transform data using fitted model (uses detected format from fit).
        
        Parameters
        ----------
        X : DataFrame, array, or list of strings
            Input data (same format as used in fit)
        **kwargs : optional
            Additional parameters (same as fit)
        
        Returns
        -------
        results : pd.DataFrame
            Significance rankings and diagnostics
        
        Examples
        --------
        >>> dt = DataTypical()
        >>> dt.fit(train_data)
        >>> results = dt.transform(test_data)
        """
        if self._detected_data_type is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return_ranks_only = kwargs.get('return_ranks_only', self.return_ranks_only)
        
        if self._detected_data_type == 'tabular':
            return self._transform_tabular(X, return_ranks_only)
        elif self._detected_data_type == 'text':
            return self._transform_text(X, return_ranks_only)
        elif self._detected_data_type == 'graph':
            # Graph transform needs to recompute topology if edges provided
            edges = kwargs.get('edges', kwargs.get('edge_index', None))
            return self._transform_graph(X, edges, return_ranks_only)
        else:
            raise RuntimeError(f"Unknown detected type: {self._detected_data_type}")
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, List[str]],
        return_ranks_only: Optional[bool] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fit and transform in one step (auto-detects format).
        
        Parameters
        ----------
        X : DataFrame, array, or list of strings
            Input data
        return_ranks_only : bool, optional
            If True, return only rank columns
        **kwargs : optional
            Additional parameters (see fit() for details)
        
        Returns
        -------
        results : pd.DataFrame
            Significance rankings and diagnostics
        
        Examples
        --------
        >>> # Tabular
        >>> dt = DataTypical()
        >>> results = dt.fit_transform(data)
        
        >>> # Text
        >>> dt = DataTypical(stereotype_keywords=['keyword'])
        >>> results = dt.fit_transform(corpus)
        
        >>> # Graph
        >>> dt = DataTypical(graph_topology_features=['degree'])
        >>> results = dt.fit_transform(node_features, edges=edges)
        """
        self.fit(X, **kwargs)
        if return_ranks_only is not None:
            kwargs['return_ranks_only'] = return_ranks_only
        return self.transform(X, **kwargs)

    # --------------------------
    # Internal Methods (Type-Specific)
    # --------------------------
    def _fit_tabular(self, X: Union[pd.DataFrame, np.ndarray]):
        """Internal method for fitting tabular data."""
        self._validate_stereotype_config()
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        self.train_index_ = df.index.copy()

        # v0.8.0: validate the stereotype column here, at fit time, whatever
        # shapley_mode is set to. Previously it was only read when
        # shapley_mode=True, or later at transform, so a missing or non-numeric
        # column surfaced well after the call that introduced it.
        if self.stereotype_column is not None:
            self._get_stereotype_source_table(df)

        with _ThreadControl(self.deterministic and not self.speed_mode) as tc:
            _seed_everything(self.random_state)
            X_scaled, X_l2 = self._preprocess_table_fit(df)
            self._fit_components(X_scaled, X_l2, df.index)

            # Store stereotype source for Shapley
            if self.stereotype_column is not None and self.shapley_mode:
                self._stereotype_source_fit_ = self._get_stereotype_source_table(df)

            # NEW: Shapley analysis
            if self.shapley_mode:
                if self.verbose:
                    print("\n" + "="*70)
                    print("SHAPLEY DUAL-PERSPECTIVE ANALYSIS")
                    print("="*70)
                self._fit_shapley_dual_perspective(X_scaled, X_l2, df.index)
            self._record_settings(tc)
        return self
    
    def _fit_text(
        self,
        corpus: Union[List[str], Iterable[str]],
        vectorizer: str = "tfidf",
        text_metadata: Optional[pd.DataFrame] = None
    ):
        """Internal method for fitting text data."""
        corpus = list(corpus)  # materialize once; _preprocess_text_fit may consume it
        self._validate_stereotype_config()
        with _ThreadControl(self.deterministic and not self.speed_mode) as tc:
            _seed_everything(self.random_state)
            X_scaled, X_l2 = self._preprocess_text_fit(corpus, vectorizer, text_metadata)

            # v0.8.0: validate the stereotype configuration against the metadata
            # preprocessing has just attached. _get_stereotype_source_text holds
            # the right errors but nothing ever called it, so a stereotype_column
            # supplied without text_metadata silently fell back to extremeness --
            # the same class of silent substitution as the archetypal backend.
            _stereotype_check = self._get_stereotype_source_text()
            if _stereotype_check is not None and self.stereotype_column is not None:
                _stereotype_values_as_float(_stereotype_check, self.stereotype_column)

            idx = pd.RangeIndex(X_scaled.shape[0])
            self.train_index_ = idx
            self._fit_components(X_scaled, X_l2, idx)

            # Store stereotype source for Shapley (text: metadata column or keyword scores)
            if self.shapley_mode:
                if self.stereotype_column is not None and self.text_metadata_ is not None \
                        and self.stereotype_column in self.text_metadata_.columns:
                    self._stereotype_source_fit_ = self.text_metadata_[self.stereotype_column]
                elif self.stereotype_keyword_scores_ is not None:
                    self._stereotype_source_fit_ = pd.Series(self.stereotype_keyword_scores_)
                if self.verbose:
                    print("\n" + "="*70)
                    print("SHAPLEY DUAL-PERSPECTIVE ANALYSIS")
                    print("="*70)
                self._fit_shapley_dual_perspective(X_scaled, X_l2, idx)

            self._record_settings(tc)
        return self
    
    def _fit_graph(
        self,
        node_features: Union[pd.DataFrame, np.ndarray],
        edges: Optional[np.ndarray] = None,
        compute_topology: bool = True
    ):
        """Internal method for fitting graph data."""
        # Convert to DataFrame
        if isinstance(node_features, pd.DataFrame):
            df = node_features.copy()
        else:
            df = pd.DataFrame(node_features)
        
        n_nodes = len(df)
        
        # Compute topology features if edges provided
        self.graph_topology_df_ = None
        if edges is not None and compute_topology:
            topology_df = self._compute_graph_topology_features(edges, n_nodes)
            self.graph_topology_df_ = topology_df
            
            # Append to node features
            for col in topology_df.columns:
                if col not in df.columns:
                    df[col] = topology_df[col].values
                else:
                    warnings.warn(f"Topology feature '{col}' already exists, skipping")
        
        # Delegate to tabular processing
        return self._fit_tabular(df)
    
    def _transform_tabular(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_ranks_only: bool = False
    ) -> pd.DataFrame:
        """Internal method for transforming tabular data."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        with _ThreadControl(self.deterministic and not self.speed_mode):
            X_scaled, X_l2 = self._preprocess_table_transform(df)
            
            # Get stereotype source for transform
            stereotype_source = None
            if self.stereotype_column is not None:
                stereotype_source = self._get_stereotype_source_table(df)
            
            ranks = self._score_with_fitted(X_scaled, X_l2, df.index, stereotype_source)

            # Add Shapley formative rankings aligned to the current data index
            if self.shapley_mode:
                shapley_ranks = self._compute_shapley_formative_ranks().reindex(df.index)
                ranks = pd.concat([ranks, shapley_ranks], axis=1)
        
        if return_ranks_only:
            return ranks
        out = df.copy()
        for col in ranks.columns:
            out[col] = ranks[col]
        return out
    
    def _transform_text(
        self,
        corpus: Union[List[str], Iterable[str]],
        return_ranks_only: bool = False
    ) -> pd.DataFrame:
        """Internal method for transforming text data."""
        corpus = list(corpus)  # materialize once so keyword scoring can reuse it
        with _ThreadControl(self.deterministic and not self.speed_mode):
            X_scaled, X_l2 = self._preprocess_text_transform(corpus)
            idx = pd.RangeIndex(X_scaled.shape[0])
            
            # Get stereotype source (priority: metadata column > keywords > None)
            stereotype_source = None
            
            # Priority 1: Metadata column (from fit_text)
            if self.stereotype_column is not None and self.text_metadata_ is not None:
                if self.stereotype_column in self.text_metadata_.columns:
                    stereotype_source = self.text_metadata_[self.stereotype_column]
            
            # Priority 2: Keyword scores (recompute on new corpus)
            elif self.stereotype_keywords is not None:
                corpus_list = list(corpus)
                X_tfidf = self.vectorizer_.transform(corpus_list)
                keyword_scores = self._compute_keyword_scores(
                    X_tfidf, corpus_list, self.stereotype_keywords
                )
                stereotype_source = pd.Series(keyword_scores)
            
            ranks = self._score_with_fitted(X_scaled, X_l2, idx, stereotype_source)

            # Add Shapley formative rankings aligned to the current data index
            if self.shapley_mode:
                shapley_ranks = self._compute_shapley_formative_ranks().reindex(idx)
                ranks = pd.concat([ranks, shapley_ranks], axis=1)
            
            return ranks
    
    def _transform_graph(
        self,
        node_features: Union[pd.DataFrame, np.ndarray],
        edges: Optional[np.ndarray] = None,
        return_ranks_only: bool = False
    ) -> pd.DataFrame:
        """Internal method for transforming graph data."""
        # Convert to DataFrame
        if isinstance(node_features, pd.DataFrame):
            df = node_features.copy()
        else:
            df = pd.DataFrame(node_features)
        
        n_nodes = len(df)
        
        # Recompute topology features if edges provided and model was trained with them
        if edges is not None and self.graph_topology_df_ is not None:
            topology_df = self._compute_graph_topology_features(edges, n_nodes)
            
            # Append to node features
            for col in topology_df.columns:
                if col not in df.columns:
                    df[col] = topology_df[col].values
        
        # Delegate to tabular transform (which handles Shapley ranks)
        return self._transform_tabular(df, return_ranks_only)

    # ============================================================
    # Shapley Dual-Perspective Methods (NEW in v0.6)
    # ============================================================
        
    def _fit_shapley_dual_perspective(
            self,
            X_scaled: ArrayLike,
            X_l2: ArrayLike,
            index: pd.Index
        ) -> None:
            """
            Fit Shapley analysis with dual perspective:
            1. Explanations: Why is each sample significant? (always computed)
            2. Formative: Which samples create structure? (optional)
    
            v0.7.2: Archetypal formative computation now uses cached archetype geometry
            (self.H_) instead of refitting ConvexHull or PCHA on every permutation
            subset. This resolves intractable runtimes on datasets with more than 8
            features while preserving scientific consistency between the two axes of
            the dual-perspective scatter plot.
    
            MEMORY OPTIMIZED: Cleanup X_dense after Shapley computation.
            """
            X_dense = X_scaled.toarray() if (sp is not None and sp.isspmatrix(X_scaled)) \
                else np.asarray(X_scaled, dtype=np.float64)
            n_samples, n_features = X_dense.shape
    
            # Determine if we compute formative
            compute_formative = self.shapley_compute_formative if self.shapley_compute_formative is not None else True
    
            # SUBSAMPLE LOGIC: Only for explanations
            subsample_indices_explanations = None
    
            if self.shapley_top_n is not None:
                # Support both fraction and absolute count
                if isinstance(self.shapley_top_n, float) and 0 < self.shapley_top_n < 1:
                    n_subsample = max(1, int(self.shapley_top_n * n_samples))
                else:
                    n_subsample = int(self.shapley_top_n)
    
                if n_subsample < n_samples:
                    if self.verbose:
                        print(f"\n[Subsampling] Selecting top {n_subsample} samples per metric")
                        if compute_formative:
                            print("  Formative: Full dataset (required for structure)")
                        else:
                            print("  Formative: SKIPPED (fast_mode)")
    
                    # Get correct stereotype source for ranking
                    stereotype_source = getattr(self, '_stereotype_source_fit_', None)
                    temp_results = self._score_with_fitted(X_scaled, X_l2, index, stereotype_source)
    
                    # Get top n_subsample for each metric separately (skip NaN columns from selected_significance)
                    _run_arch  = self.selected_significance is None or self.selected_significance == "archetypal"
                    _run_proto = self.selected_significance is None or self.selected_significance == "prototypical"
                    _run_stereo = self.selected_significance is None or self.selected_significance == "stereotypical"
                    top_arch  = set(temp_results.nlargest(n_subsample, 'archetypal_rank').index)  if _run_arch  else set()
                    top_proto = set(temp_results.nlargest(n_subsample, 'prototypical_rank').index) if _run_proto else set()
                    top_stereo = set(temp_results.nlargest(n_subsample, 'stereotypical_rank').index) if _run_stereo else set()
    
                    # Union of all top samples - NO TRIMMING
                    # Ensures all top-N samples from each metric have Shapley values
                    top_indices_union = top_arch | top_proto | top_stereo
    
                    if self.verbose:
                        print(f"    Top {n_subsample} archetypal samples: {len(top_arch)}")
                        print(f"    Top {n_subsample} prototypical samples: {len(top_proto)}")
                        print(f"    Top {n_subsample} stereotypical samples: {len(top_stereo)}")
                        print(f"    Union: {len(top_indices_union)} unique samples")
                        print(f"    (Computing Shapley for all union samples - ensures no empty plots)")
    
                    # Identify core samples (appear in multiple metric top-N lists)
                    # Core samples get full permutations; secondary get reduced permutations
                    # When only one significance type is active, all samples are treated as core
                    n_active_metrics = sum([bool(top_arch), bool(top_proto), bool(top_stereo)])
                    sample_counts = {}
                    for idx in top_indices_union:
                        count = sum([idx in top_arch, idx in top_proto, idx in top_stereo])
                        sample_counts[idx] = count

                    if n_active_metrics <= 1:
                        # Single significance type: all top samples are equally important
                        core_samples_df_idx = list(top_indices_union)
                    else:
                        # Core = samples in 2+ metrics (most important)
                        core_samples_df_idx = [idx for idx, cnt in sample_counts.items() if cnt >= 2]
                    core_positions = sorted([index.get_loc(idx) for idx in core_samples_df_idx])
                    self._union_core_samples = np.array(core_positions)
    
                    if self.verbose:
                        print(f"    Core samples (in 2+ metrics): {len(core_samples_df_idx)}")
                        print(f"    Secondary samples (in 1 metric): {len(top_indices_union) - len(core_samples_df_idx)}")
    
                    # Convert to positional indices (deterministic order via sorting)
                    top_positions = sorted([index.get_loc(idx) for idx in top_indices_union])
                    subsample_indices_explanations = np.array(top_positions)
    
                    # MEMORY CLEANUP
                    _cleanup_memory(temp_results)
    
            # Initialize Shapley engine
            engine = ShapleySignificanceEngine(
                n_permutations=self.shapley_n_permutations,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                early_stopping_patience=self.shapley_early_stopping_patience,
                early_stopping_tolerance=self.shapley_early_stopping_tolerance,
                verbose=self.verbose
            )
    
            run_arch_shap  = self.selected_significance is None or self.selected_significance == "archetypal"
            run_proto_shap = self.selected_significance is None or self.selected_significance == "prototypical"
            run_stereo_shap = self.selected_significance is None or self.selected_significance == "stereotypical"

            # PERSPECTIVE 1: Formative Instances (optional)
            if compute_formative:
                if self.verbose:
                    print("\n[1] Computing Formative Instances (global perspective)...")
                    print("    Using FULL dataset (required to measure structure)")

                if run_arch_shap and self.H_ is not None:
                    if self.formative_method == "exact":
                        # v0.8.0: the closed form, so the ranking carries no
                        # sampling noise at all. Spread across the feature
                        # columns the same way the sampler does, so the shape
                        # and every downstream consumer are unchanged.
                        _exact = exact_formative_archetypal(
                            X_dense, self.H_.astype(np.float64))
                        n_feat = X_dense.shape[1]
                        self.Phi_archetypal_formative_ = np.repeat(
                            (_exact / max(n_feat, 1))[:, None], n_feat, axis=1)
                        self.shapley_info_['archetypal_formative'] = {
                            'method': 'exact',
                            'n_permutations_used': 0,
                            'converged': True,
                            'split_half_rho': 1.0,
                            'additivity_error': 0.0,
                        }
                        if self.verbose:
                            print("    Archetypal formative: exact closed form, "
                                  "no sampling")
                    else:
                        context_archetypal = {'archetypes': self.H_.astype(np.float64)}
                        self.Phi_archetypal_formative_, self.shapley_info_['archetypal_formative'] = \
                            engine.compute_shapley_values(
                                X_dense,
                                formative_archetypal_pcha_cached,
                                "Archetypal Formative (Cached Archetypes)",
                                context_archetypal
                            )
                        self.shapley_info_['archetypal_formative']['method'] = 'monte_carlo'
                        _cleanup_memory(context_archetypal)
                else:
                    self.Phi_archetypal_formative_ = None

                if run_proto_shap:
                    if self.formative_method == "exact":
                        # v0.8.0: the closed form. This game is the least
                        # convergent of the three under sampling, so the
                        # ranking it produced was a property of the seed.
                        # Spread across the feature columns the same way the
                        # sampler does, so every downstream consumer is
                        # unchanged.
                        _exact = exact_formative_prototypical(X_dense)
                        n_feat = X_dense.shape[1]
                        self.Phi_prototypical_formative_ = np.repeat(
                            (_exact / max(n_feat, 1))[:, None], n_feat, axis=1)
                        self.shapley_info_['prototypical_formative'] = {
                            'method': 'exact',
                            'n_permutations_used': 0,
                            'converged': True,
                            'split_half_rho': 1.0,
                            'additivity_error': 0.0,
                        }
                        if self.verbose:
                            print("    Prototypical formative: exact closed "
                                  "form, no sampling")
                    else:
                        self.Phi_prototypical_formative_, self.shapley_info_['prototypical_formative'] = \
                            engine.compute_shapley_values(
                                X_dense,
                                formative_prototypical_coverage,
                                "Prototypical Formative (Coverage)"
                            )
                        self.shapley_info_['prototypical_formative']['method'] = 'monte_carlo'
                else:
                    self.Phi_prototypical_formative_ = None

                if run_stereo_shap and self.stereotype_column is not None \
                        and getattr(self, '_stereotype_source_fit_', None) is not None:
                    target_values = self._stereotype_source_fit_.to_numpy(dtype=np.float64)
                    context = {
                        'target_values': target_values,
                        'target': self.stereotype_target,
                        'median': np.median(target_values)
                    }
                    if self.formative_method == "exact":
                        _exact = exact_formative_stereotypical(
                            target_values, self.stereotype_target,
                            float(np.median(target_values)))
                        n_feat = X_dense.shape[1]
                        self.Phi_stereotypical_formative_ = np.repeat(
                            (_exact / max(n_feat, 1))[:, None], n_feat, axis=1)
                        self.shapley_info_['stereotypical_formative'] = {
                            'method': 'exact',
                            'n_permutations_used': 0,
                            'converged': True,
                            'split_half_rho': 1.0,
                            'additivity_error': 0.0,
                        }
                        if self.verbose:
                            print("    Stereotypical formative: exact closed "
                                  "form, no sampling")
                    else:
                        self.Phi_stereotypical_formative_, self.shapley_info_['stereotypical_formative'] = \
                            engine.compute_shapley_values(
                                X_dense,
                                formative_stereotypical_extremeness,
                                "Stereotypical Formative (Extremeness)",
                                context
                            )
                        self.shapley_info_['stereotypical_formative']['method'] = 'monte_carlo'
                else:
                    self.Phi_stereotypical_formative_ = None
            else:
                # Skip formative computation (fast_mode)
                if self.verbose:
                    print("\n[1] Skipping Formative Instances (fast_mode)")

                self.Phi_archetypal_formative_ = None
                self.Phi_prototypical_formative_ = None
                self.Phi_stereotypical_formative_ = None
    
            # PERSPECTIVE 2: Explanations (always computed, optionally subsampled)
            if self.verbose:
                print("\n[2] Computing Local Explanations (why is each sample significant)...")
                if subsample_indices_explanations is not None:
                    print(f"    Computing for {len(subsample_indices_explanations)} samples (union of top-N per metric)")
                else:
                    print(f"    Computing for all {n_samples} instances")
    
            self._fit_shapley_explanations(
                X_dense, index, engine,
                subsample_indices_explanations
            )
    
            # MEMORY CLEANUP: Free X_dense copy (original X_scaled still needed)
            _cleanup_memory(X_dense, force_gc=True)
    
            if self.verbose:
                print("\n" + "="*70)
                if compute_formative:
                    print("Shapley Dual-Perspective Analysis Complete")
                else:
                    print("Shapley Explanations Complete (formative skipped)")
                print("="*70)
            

    def _fit_shapley_explanations(
        self,
        X_dense: np.ndarray,
        index: pd.Index,
        engine: ShapleySignificanceEngine,
        subsample_indices: Optional[np.ndarray] = None
    ) -> None:
        """
        Compute Shapley explanations with optional subsampling.
        
        OPTIMIZED: Two-tier permutation strategy for union samples.
        """
        
        n_samples, n_features = X_dense.shape
        
        # Determine which samples to compute for
        if subsample_indices is not None:
            samples_to_compute = subsample_indices
            
            # OPTIMIZATION: Two-tier permutation strategy
            # If we have union samples, use full permutations only for "core" samples
            # Core = samples that appear in multiple metric top-N lists
            if hasattr(self, '_union_core_samples'):
                core_samples = self._union_core_samples
                secondary_samples = np.setdiff1d(samples_to_compute, core_samples)
                
                if self.verbose and len(secondary_samples) > 0:
                    print(f"    Two-tier permutation strategy:")
                    print(f"      Core samples ({len(core_samples)}): {engine.n_permutations} permutations")
                    print(f"      Secondary samples ({len(secondary_samples)}): {engine.n_permutations // 2} permutations")
            else:
                core_samples = samples_to_compute
                secondary_samples = np.array([])
        else:
            samples_to_compute = np.arange(n_samples)
            core_samples = samples_to_compute
            secondary_samples = np.array([])
        
        run_arch_exp  = self.selected_significance is None or self.selected_significance == "archetypal"
        run_proto_exp = self.selected_significance is None or self.selected_significance == "prototypical"
        run_stereo_exp = self.selected_significance is None or self.selected_significance == "stereotypical"

        # Initialize full-size arrays only for selected significance types
        self.Phi_archetypal_explanations_ = np.zeros((n_samples, n_features), dtype=np.float64) if run_arch_exp else None
        self.Phi_prototypical_explanations_ = np.zeros((n_samples, n_features), dtype=np.float64) if run_proto_exp else None
        self.Phi_stereotypical_explanations_ = np.zeros((n_samples, n_features), dtype=np.float64) if run_stereo_exp else None
        
        # Value functions for explanations
        def explain_archetypal_features(X_subset, indices, ctx):
            """Archetypal score for single sample with feature subset."""
            if len(X_subset) == 0 or X_subset.shape[1] == 0:
                return 0.0
            dist_to_boundary = np.minimum(X_subset, 1.0 - X_subset)
            archetypal_contribution = np.mean(1.0 - 2.0 * dist_to_boundary, axis=1)
            return float(np.mean(archetypal_contribution))
        
        def explain_prototypical_features(X_subset, indices, ctx):
            """Prototypical score for single sample with feature subset."""
            if len(X_subset) == 0 or X_subset.shape[1] == 0:
                return 0.0
            return float(np.mean(np.var(X_subset, axis=1)))
        
        context = {'sample_mode': 'features'}

        # Pre-extract full target array so per-tier subsets can be sliced correctly.
        # Value functions receive LOCAL indices (0..len(X_subset)-1), so target_values
        # must be sized to match the X subset passed to compute_feature_shapley_values.
        _full_target_values: Optional[np.ndarray] = None
        if run_stereo_exp and self.stereotype_column is not None \
                and getattr(self, '_stereotype_source_fit_', None) is not None:
            _full_target_values = _stereotype_values_as_float(
                self._stereotype_source_fit_, self.stereotype_column
            )

        # v0.8.0 FIX: this closure and its context entries are defined at function
        # scope, not inside the core-samples branch. When subsample_indices is
        # supplied, core_samples comes from self._union_core_samples and can be
        # empty while secondary_samples is not; the secondary branch calls this
        # closure too, which previously raised UnboundLocalError.
        def explain_stereotypical_features(X_subset, indices, ctx):
            if len(X_subset) == 0 or X_subset.shape[1] == 0:
                return 0.0
            if ctx.get('target_values') is None:
                return 0.0

            sample_idx = indices[0]
            target_value = ctx['target_values'][sample_idx]
            target = ctx['stereotype_target']

            if isinstance(target, str):
                median = ctx.get('median', np.median(ctx['target_values']))
                if target == 'max':
                    distance = max(0, target_value - median)
                elif target == 'min':
                    distance = max(0, median - target_value)
                else:
                    distance = 0.0
            else:
                distance = -abs(target_value - target)

            feature_contrib = float(np.mean(np.abs(X_subset)))
            return distance * feature_contrib

        if run_stereo_exp and self.stereotype_column is not None:
            context['stereotype_target'] = self.stereotype_target
            context['median'] = np.median(_full_target_values) if _full_target_values is not None else 0.0

        # COMPUTE CORE SAMPLES (full permutations)
        if len(core_samples) > 0:
            if self.verbose:
                print(f"  Computing explanations for significance rankings...")

            X_core = X_dense[core_samples, :]

            if run_arch_exp:
                Phi_arch_core, info_arch = engine.compute_feature_shapley_values(
                    X_core,
                    explain_archetypal_features,
                    "Archetypal Explanations (Core)",
                    context
                )
                self.Phi_archetypal_explanations_[core_samples, :] = Phi_arch_core
                self.shapley_info_['archetypal_explanations'] = info_arch

            if run_proto_exp:
                if self.verbose:
                    print(f"  Computing prototypical explanations (core: {len(core_samples)} samples)...")

                Phi_proto_core, info_proto = engine.compute_feature_shapley_values(
                    X_core,
                    explain_prototypical_features,
                    "Prototypical Explanations (Core)",
                    context
                )
                self.Phi_prototypical_explanations_[core_samples, :] = Phi_proto_core
                self.shapley_info_['prototypical_explanations'] = info_proto

            # Stereotypical explanations (if applicable)
            if run_stereo_exp and self.stereotype_column is not None:
                if self.verbose:
                    print(f"  Computing stereotypical explanations (core: {len(core_samples)} samples)...")

                # Slice to core_samples so local index i == core_samples[i] globally
                context['target_values'] = _full_target_values[core_samples] if _full_target_values is not None else None
                
                Phi_stereo_core, info_stereo = engine.compute_feature_shapley_values(
                    X_core,
                    explain_stereotypical_features,
                    "Stereotypical Explanations (Core)",
                    context
                )
                self.Phi_stereotypical_explanations_[core_samples, :] = Phi_stereo_core
                self.shapley_info_['stereotypical_explanations'] = info_stereo
            
            # MEMORY CLEANUP
            _cleanup_memory(X_core)
        
        # COMPUTE SECONDARY SAMPLES (reduced permutations for speed)
        if len(secondary_samples) > 0:
            # Temporarily reduce permutations
            original_n_perms = engine.n_permutations
            engine.n_permutations = max(10, original_n_perms // 2)
            
            if self.verbose:
                print(f"  Computing explanations (secondary: {len(secondary_samples)} samples, {engine.n_permutations} perms)...")
            
            X_secondary = X_dense[secondary_samples, :]

            if run_arch_exp:
                Phi_arch_sec, _ = engine.compute_feature_shapley_values(
                    X_secondary, explain_archetypal_features,
                    "Archetypal Explanations (Secondary)", context
                )
                self.Phi_archetypal_explanations_[secondary_samples, :] = Phi_arch_sec

            if run_proto_exp:
                Phi_proto_sec, _ = engine.compute_feature_shapley_values(
                    X_secondary, explain_prototypical_features,
                    "Prototypical Explanations (Secondary)", context
                )
                self.Phi_prototypical_explanations_[secondary_samples, :] = Phi_proto_sec

            if run_stereo_exp and self.stereotype_column is not None:
                # Re-slice target_values for secondary indices so local indices align
                context['target_values'] = _full_target_values[secondary_samples] if _full_target_values is not None else None
                Phi_stereo_sec, _ = engine.compute_feature_shapley_values(
                    X_secondary, explain_stereotypical_features,
                    "Stereotypical Explanations (Secondary)", context
                )
                self.Phi_stereotypical_explanations_[secondary_samples, :] = Phi_stereo_sec
            
            # Restore original permutations
            engine.n_permutations = original_n_perms
            
            # MEMORY CLEANUP
            _cleanup_memory(X_secondary)
        else:
            self.Phi_stereotypical_explanations_ = None if self.stereotype_column is None else self.Phi_stereotypical_explanations_

    def _v04_archetypal_value(
        self,
        X_subset: np.ndarray,
        indices: np.ndarray,
        context: Dict
    ) -> float:
        """Value function: Mean archetypal rank from v0.4 NMF method (Option A)."""
        if len(X_subset) < context['nmf_rank']:
            return 0.0
        
        try:
            nmf = NMF(
                n_components=min(context['nmf_rank'], len(X_subset)-1),
                init='random',
                random_state=context['random_state'],
                max_iter=100,
                tol=0.01
            )
            
            X_nn = X_subset - X_subset.min() + 1e-6
            W_subset = nmf.fit_transform(X_nn)
            W_norm = W_subset / (W_subset.sum(axis=1, keepdims=True) + 1e-12)
            arch_scores = np.max(W_norm, axis=1)
            
            return float(np.mean(arch_scores))
        except Exception:
            return float(np.mean(np.ptp(X_subset, axis=0)))

    def _v04_prototypical_value(
        self,
        X_subset: np.ndarray,
        indices: np.ndarray,
        context: Dict
    ) -> float:
        """Value function: Coverage from v0.4 facility location."""
        if len(X_subset) < 2:
            return 0.0
        
        norms = np.linalg.norm(X_subset, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X_l2 = X_subset / norms
        
        sims = X_l2 @ X_l2.T
        np.fill_diagonal(sims, 0)
        
        max_sims = np.max(sims, axis=1) if sims.shape[0] > 0 else np.array([0.0])
        return float(np.mean(max_sims))

    def _v04_stereotypical_value(
        self,
        X_subset: np.ndarray,
        indices: np.ndarray,
        context: Dict
    ) -> float:
        """Value function: Target alignment from v0.4 stereotype targeting."""
        if context.get('target_values') is None:
            s = np.max(np.abs(X_subset - 0.5), axis=1) * 2.0
            return float(np.mean(s))
        
        target_vals = context['target_values'][indices]
        target = context['stereotype_target']
        
        if target == 'max':
            return float(np.mean(target_vals))
        elif target == 'min':
            return float(-np.mean(target_vals))
        else:
            return float(-np.mean(np.abs(target_vals - float(target))))

    def _compute_shapley_formative_ranks(self) -> pd.DataFrame:
        """Compute formative instance rankings from Shapley values.

        Handles any combination of None/non-None formative arrays so that
        selected_significance omissions produce None columns rather than errors.
        """
        n_samples = len(self.train_index_)

        def _norm_or_none(arr, name=""):
            if arr is None:
                return [None] * n_samples
            vals = arr.sum(axis=1)

            # v0.8.0: a non-finite value function used to land in the
            # equal-values branch below, because every comparison against NaN is
            # False, and every sample was then reported as 0.5. A mid-rank for
            # the whole dataset is a confident-looking number standing in for
            # "could not be computed". Propagate the NaN instead, and say so.
            if not np.all(np.isfinite(vals)):
                n_bad = int(np.count_nonzero(~np.isfinite(vals)))
                warnings.warn(
                    f"{n_bad} of {vals.size} formative {name} value(s) are not "
                    "finite, so the corresponding ranks are NaN rather than a "
                    "number. This usually means the value function could not be "
                    "evaluated on some coalition, often from degenerate or "
                    "constant data.",
                    RuntimeWarning, stacklevel=3
                )
                out = np.full(vals.shape, np.nan, dtype=np.float64)
                good = np.isfinite(vals)
                if good.sum() > 1:
                    g = vals[good]
                    g_min, g_max = float(g.min()), float(g.max())
                    if (g_max - g_min) > 1e-12:
                        out[good] = (g - g_min) / (g_max - g_min)
                    else:
                        out[good] = 0.5
                elif good.sum() == 1:
                    out[good] = 0.5
                return np.round(out, 10)

            r_min, r_max = vals.min(), vals.max()
            if (r_max - r_min) > 1e-12:
                normalized = (vals - r_min) / (r_max - r_min)
            else:
                normalized = np.ones_like(vals) * 0.5
            return np.round(normalized, 10)

        return pd.DataFrame({
            'archetypal_shapley_rank':    _norm_or_none(self.Phi_archetypal_formative_, "archetypal"),
            'prototypical_shapley_rank':  _norm_or_none(self.Phi_prototypical_formative_, "prototypical"),
            'stereotypical_shapley_rank': _norm_or_none(self.Phi_stereotypical_formative_, "stereotypical"),
        }, index=self.train_index_)

        
    def get_shapley_explanations(self, sample_idx: int) -> Dict[str, np.ndarray]:
        """Get Shapley feature attributions explaining why sample is archetypal/prototypical/stereotypical."""
        if not self.shapley_mode:
            raise RuntimeError("Shapley mode not enabled. Set shapley_mode=True when fitting.")

        all_none = (
            self.Phi_archetypal_explanations_ is None
            and self.Phi_prototypical_explanations_ is None
            and self.Phi_stereotypical_explanations_ is None
        )
        if all_none:
            raise RuntimeError("Shapley explanations not computed. Call fit() first.")
        
        # Convert DataFrame index to positional index
        if hasattr(self, 'train_index_') and self.train_index_ is not None:
            try:
                pos_idx = self.train_index_.get_loc(sample_idx)
            except KeyError:
                raise ValueError(f"Sample index {sample_idx} not found in training data")
        else:
            # Assume sample_idx is already positional
            pos_idx = sample_idx
        
        explanations = {}
        
        if self.Phi_archetypal_explanations_ is not None:
            explanations['archetypal'] = self.Phi_archetypal_explanations_[pos_idx]
        
        if self.Phi_prototypical_explanations_ is not None:
            explanations['prototypical'] = self.Phi_prototypical_explanations_[pos_idx]
        
        if self.Phi_stereotypical_explanations_ is not None:
            explanations['stereotypical'] = self.Phi_stereotypical_explanations_[pos_idx]
        
        return explanations

    def get_formative_attributions(self, sample_idx: int) -> Dict[str, np.ndarray]:
        """Get Shapley feature attributions showing how sample creates archetypal/prototypical/stereotypical structure."""
        if not self.shapley_mode:
            raise RuntimeError("Shapley mode not enabled. Set shapley_mode=True when fitting.")
        
        if self.Phi_archetypal_formative_ is None:
            raise RuntimeError(
                "Formative instances not computed. "
                "This occurs when fast_mode=True (formative skipped for speed). "
                "Use fast_mode=False to compute formative instances."
            )
        
        # Convert DataFrame index to positional index
        if hasattr(self, 'train_index_') and self.train_index_ is not None:
            try:
                pos_idx = self.train_index_.get_loc(sample_idx)
            except KeyError:
                raise ValueError(f"Sample index {sample_idx} not found in training data")
        else:
            # Assume sample_idx is already positional
            pos_idx = sample_idx
        
        attributions = {}
        
        if self.Phi_archetypal_formative_ is not None:
            attributions['archetypal'] = self.Phi_archetypal_formative_[pos_idx]
        
        if self.Phi_prototypical_formative_ is not None:
            attributions['prototypical'] = self.Phi_prototypical_formative_[pos_idx]
        
        if self.Phi_stereotypical_formative_ is not None:
            attributions['stereotypical'] = self.Phi_stereotypical_formative_[pos_idx]
        
        return attributions

    # --------------------------
    # Text (TF-IDF)
    # --------------------------
    def fit_text(
        self, 
        corpus: Iterable[str], 
        vectorizer: str = "tfidf",
        text_metadata: Optional[pd.DataFrame] = None
    ):
        """
        Fit on text corpus with optional metadata.
        
        Parameters
        ----------
        corpus : Iterable[str]
            Text documents
        vectorizer : str
            Vectorization method (default: 'tfidf')
        text_metadata : pd.DataFrame, optional
            Document-level properties for stereotype computation
            Must have same number of rows as documents in corpus
        """
        corpus = list(corpus)  # materialize once; _preprocess_text_fit may consume it
        self._reset_fit_state()
        self._apply_fast_mode_defaults()
        self._validate_stereotype_config()
        with _ThreadControl(self.deterministic and not self.speed_mode) as tc:
            _seed_everything(self.random_state)
            X_scaled, X_l2 = self._preprocess_text_fit(corpus, vectorizer, text_metadata)

            # v0.8.0: validate the stereotype configuration against the metadata
            # preprocessing has just attached. _get_stereotype_source_text holds
            # the right errors but nothing ever called it, so a stereotype_column
            # supplied without text_metadata silently fell back to extremeness --
            # the same class of silent substitution as the archetypal backend.
            _stereotype_check = self._get_stereotype_source_text()
            if _stereotype_check is not None and self.stereotype_column is not None:
                _stereotype_values_as_float(_stereotype_check, self.stereotype_column)

            idx = pd.RangeIndex(X_scaled.shape[0])
            self.train_index_ = idx
            self._fit_components(X_scaled, X_l2, idx)

            # Store stereotype source for Shapley (text: metadata column or keyword scores)
            if self.shapley_mode:
                if self.stereotype_column is not None and self.text_metadata_ is not None \
                        and self.stereotype_column in self.text_metadata_.columns:
                    self._stereotype_source_fit_ = self.text_metadata_[self.stereotype_column]
                elif self.stereotype_keyword_scores_ is not None:
                    self._stereotype_source_fit_ = pd.Series(self.stereotype_keyword_scores_)
                if self.verbose:
                    print("\n" + "="*70)
                    print("SHAPLEY DUAL-PERSPECTIVE ANALYSIS")
                    print("="*70)
                self._fit_shapley_dual_perspective(X_scaled, X_l2, idx)

            self._record_settings(tc)
        return self

    def transform_text(self, corpus: Iterable[str]) -> pd.DataFrame:
        """Transform text corpus."""
        corpus = list(corpus)  # materialize once so keyword scoring can reuse it
        with _ThreadControl(self.deterministic and not self.speed_mode):
            X_scaled, X_l2 = self._preprocess_text_transform(corpus)
            idx = pd.RangeIndex(X_scaled.shape[0])
            
            # Get stereotype source (priority: metadata column > keywords > None)
            stereotype_source = None
            
            # Priority 1: Metadata column (from fit_text)
            if self.stereotype_column is not None and self.text_metadata_ is not None:
                if self.stereotype_column in self.text_metadata_.columns:
                    stereotype_source = self.text_metadata_[self.stereotype_column]
            
            # Priority 2: Keyword scores (recompute on new corpus)
            elif self.stereotype_keywords is not None:
                X_tfidf = self.vectorizer_.transform(corpus)
                keyword_scores = self._compute_keyword_scores(
                    X_tfidf, corpus, self.stereotype_keywords
                )
                stereotype_source = pd.Series(keyword_scores)

            ranks = self._score_with_fitted(X_scaled, X_l2, idx, stereotype_source)

            # Add Shapley formative rankings aligned to the current data index
            if self.shapley_mode:
                shapley_ranks = self._compute_shapley_formative_ranks().reindex(idx)
                ranks = pd.concat([ranks, shapley_ranks], axis=1)

            return ranks

    def fit_transform_text(
        self,
        corpus: Iterable[str],
        vectorizer: str = "tfidf",
        text_metadata: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Fit and transform text in one step."""
        corpus = list(corpus)  # materialize once so both fit and transform can use it
        self.fit_text(corpus, vectorizer=vectorizer, text_metadata=text_metadata)
        return self.transform_text(corpus)

    # --------------------------
    # Signals / Graphs (numeric)
    # --------------------------
    def fit_transform_signals(self, X_signal: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        self.fit(X_signal)
        return self.transform(X_signal, return_ranks_only=True)

    def fit_transform_graph(
        self,
        node_features: Union[pd.DataFrame, np.ndarray],
        edges: Optional[np.ndarray] = None,
        edge_index: Optional[np.ndarray] = None,
        compute_topology: bool = True
    ) -> pd.DataFrame:
        """
        Fit and transform graph data.
        
        Parameters
        ----------
        node_features : DataFrame or array
            Node feature matrix (n_nodes, n_features)
        edges : np.ndarray, optional
            Edge list as (2, n_edges) or (n_edges, 2)
            Alias: edge_index
        compute_topology : bool
            Whether to compute and append topology features
            
        Returns
        -------
        results : pd.DataFrame
            Rankings with topology features if computed
        """
        # Handle edge_index alias
        if edges is None and edge_index is not None:
            edges = edge_index
        
        # Convert to DataFrame
        if isinstance(node_features, pd.DataFrame):
            df = node_features.copy()
        else:
            df = pd.DataFrame(node_features)
        
        n_nodes = len(df)
        
        # Compute topology features if edges provided
        # v0.8.0: computed here but published onto the estimator only after
        # self.fit() below, because fit() now clears the state a previous fit
        # left behind and would otherwise wipe this on its way past.
        topology_df = None
        if edges is not None and compute_topology:
            topology_df = self._compute_graph_topology_features(edges, n_nodes)
            
            # Append to node features
            for col in topology_df.columns:
                if col not in df.columns:
                    df[col] = topology_df[col].values
                else:
                    warnings.warn(f"Topology feature '{col}' already exists, skipping")
        
        # Standard tabular processing
        self.fit(df)
        self.graph_topology_df_ = topology_df
        
        # Use standard transform which preserves label columns
        results = self.transform(df, return_ranks_only=False)
        
        return results

    # --------------------------
    # Ideals (legacy stereotypes)
    # --------------------------
    def register_ideal(self, name: str, ideal_vector: Union[np.ndarray, List[float]]) -> None:
        """
        Store a named reference vector in ``ideals_``.

        .. warning::

           **This stores the vector and nothing else.** No scoring path reads
           ``ideals_``, so registering an ideal does not change any rank, any
           explanation or any output. It is a leftover from the pre-v0.4
           stereotype mechanism, kept so existing code does not break.

           To rank instances towards a target, use ``stereotype_column`` with
           ``stereotype_target``, which is the supported mechanism and is what
           the manuscript describes.

        The dimension check is still performed, so a mismatched vector is
        rejected rather than stored.
        """
        # v0.8.0: say so rather than leaving the name to imply an effect it
        # does not have.
        warnings.warn(
            "register_ideal() stores the vector in ideals_ and nothing reads "
            "it, so it has no effect on any rank or explanation. It is a "
            "leftover from the pre-v0.4 stereotype mechanism. Use "
            "stereotype_column with stereotype_target to rank towards a target.",
            DeprecationWarning, stacklevel=2
        )
        v = np.asarray(ideal_vector, dtype=np.float64).ravel()
        if self.scaler_ is None:
            raise RuntimeError("Call fit/fit_text before registering ideals.")
        d = self.H_.shape[1] if self.H_ is not None else self.scaler_.n_features_in_
        if v.shape[0] != d:
            raise ValueError(f"Ideal has dim {v.shape[0]} but data has {d} features.")
        self.ideals_[name] = v.copy()

    # --------------------------
    # Config / sklearn interop
    # --------------------------
    def to_config(self) -> Dict:
        cfg = {k: getattr(self, k) for k in [
            "nmf_rank","n_prototypes","scale","distance_metric","similarity_metric",
            "deterministic","n_jobs","max_iter_nmf","tol_nmf","speed_mode","dtype",
            "random_state","max_memory_mb","return_ranks_only","auto_n_prototypes",
            "verbose","max_missing_frac","data_type",
            "stereotype_column","stereotype_target","label_columns",
            "stereotype_keywords","graph_topology_features",
            "selected_significance",
            "shapley_mode","shapley_n_permutations","shapley_top_n",
            "shapley_early_stopping_patience","shapley_early_stopping_tolerance",
            "shapley_compute_formative","fast_mode","archetypal_method",
            "formative_method",
            # v0.8.0: feature_weights was missing, so from_config(to_config())
            # silently dropped it and produced a different fit. On a five-feature
            # frame that moved archetypal_rank by 0.33 and prototypical_rank by
            # 0.50, which is a different answer, not a rounding difference.
            "feature_weights",
        ]}
        cfg["version"] = "0.8.1"
        return cfg

    @classmethod
    def from_config(cls, cfg: Dict) -> "DataTypical":
        try:
            return cls(**{k: v for k, v in cfg.items() if k in {f.name for f in dc_fields(cls)}})
        except TypeError as e:
            raise ConfigError(str(e))

    def get_params(self, deep: bool = True) -> Dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self) if f.init}

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter {k}")
            setattr(self, k, v)
        return self

    # ============================================================
    # [F] Graph Topology Features (NEW in v0.4)
    # ============================================================
    def _compute_graph_topology_features(
        self,
        edge_index: np.ndarray,
        n_nodes: int,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute graph topology features.
        
        Parameters
        ----------
        edge_index : np.ndarray
            Edge list (2, n_edges) or (n_edges, 2)
        n_nodes : int
            Number of nodes
        feature_names : List[str], optional
            Which topology features to compute
            
        Returns
        -------
        topology_df : pd.DataFrame
            Computed topology features (n_nodes, n_features)
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for graph topology features. "
                "Install with: pip install networkx"
            )
        
        # Convert edge_index to NetworkX graph
        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2 or 2 not in edge_index.shape:
            raise ValueError(
                f"edges must be a 2D array shaped (2, n_edges) or (n_edges, 2), "
                f"got shape {edge_index.shape}."
            )

        if edge_index.shape == (2, 2):
            # v0.8.0: genuinely ambiguous. Two edges written as rows and two
            # nodes written as columns are the same array. Rows are the
            # documented form, so that is what is used, but say so rather than
            # picking silently.
            warnings.warn(
                "edges has shape (2, 2), which is ambiguous: it could be two "
                "edges as rows or two nodes as columns. Reading it as two edges, "
                "one per row. Transpose it if that is not what you meant.",
                RuntimeWarning, stacklevel=2
            )
            edges = edge_index
        elif edge_index.shape[0] == 2:
            edges = edge_index.T  # (n_edges, 2)
        else:
            edges = edge_index

        # v0.8.0: an edge naming a node outside range(n_nodes) used to be
        # accepted in silence. NetworkX simply added the extra node, so every
        # global measure was then normalised over a graph larger than the one
        # asked about: pagerank over the requested nodes summed to less than 1,
        # and betweenness and closeness were computed against phantom nodes. The
        # returned columns looked perfectly ordinary.
        if np.asarray(edges).size:
            flat = np.asarray(edges).ravel()
            bad = np.unique(flat[(flat < 0) | (flat >= n_nodes)])
            if bad.size:
                raise ValueError(
                    f"edges reference node indices outside range(0, {n_nodes}): "
                    f"{list(bad[:10])}{' and more' if bad.size > 10 else ''}. "
                    "Every topology feature would otherwise be computed on a "
                    "larger graph than the node features describe, which "
                    "silently changes pagerank, betweenness and closeness for "
                    "every node."
                )

        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        
        # Determine which features to compute
        if feature_names is None:
            feature_names = self.graph_topology_features or ['degree', 'clustering']
        
        topology_data = {}
        
        for feat_name in feature_names:
            if feat_name == 'degree':
                degree_dict = dict(G.degree())
                topology_data['degree'] = [degree_dict.get(i, 0) for i in range(n_nodes)]
            
            elif feat_name == 'clustering':
                clust_dict = nx.clustering(G)
                topology_data['clustering'] = [clust_dict.get(i, 0.0) for i in range(n_nodes)]
            
            elif feat_name == 'pagerank':
                pr_dict = nx.pagerank(G, max_iter=100)
                topology_data['pagerank'] = [pr_dict.get(i, 0.0) for i in range(n_nodes)]
            
            elif feat_name == 'triangles':
                tri_dict = nx.triangles(G)
                topology_data['triangles'] = [tri_dict.get(i, 0) for i in range(n_nodes)]
            
            elif feat_name == 'betweenness':
                bet_dict = nx.betweenness_centrality(G)
                topology_data['betweenness'] = [bet_dict.get(i, 0.0) for i in range(n_nodes)]
            
            elif feat_name == 'closeness':
                close_dict = nx.closeness_centrality(G)
                topology_data['closeness'] = [close_dict.get(i, 0.0) for i in range(n_nodes)]
            
            elif feat_name == 'eigenvector':
                try:
                    eigen_dict = nx.eigenvector_centrality(G, max_iter=100)
                    topology_data['eigenvector'] = [eigen_dict.get(i, 0.0) for i in range(n_nodes)]
                except Exception:
                    warnings.warn("Eigenvector centrality failed, using zeros")
                    topology_data['eigenvector'] = [0.0] * n_nodes
            
            else:
                warnings.warn(f"Unknown topology feature: {feat_name}")
        
        return pd.DataFrame(topology_data, index=range(n_nodes))

    # ============================================================
    # [G] Stereotype Computation (NEW in v0.4)
    # ============================================================
    def _validate_stereotype_config(self):
        """Validate stereotype configuration at fit time."""

        # Validate selected_significance
        if self.selected_significance is not None:
            _valid_sig = {'archetypal', 'prototypical', 'stereotypical'}
            if self.selected_significance not in _valid_sig:
                raise ConfigError(
                    f"selected_significance must be one of {_valid_sig} or None, "
                    f"got '{self.selected_significance}'"
                )

        # Check conflicting specifications
        if self.stereotype_column is not None and self.stereotype_keywords is not None:
            raise ConfigError(
                "Cannot specify both stereotype_column and stereotype_keywords. "
                "Use stereotype_column for metadata or stereotype_keywords for text relevance."
            )
        
        # Validate target
        if isinstance(self.stereotype_target, str):
            if self.stereotype_target not in ['min', 'max']:
                raise ConfigError(
                    f"stereotype_target must be 'min', 'max', or numeric value, "
                    f"got: '{self.stereotype_target}'"
                )

    def _compute_stereotypical_rank(
        self, 
        X_scaled: ArrayLike,
        index: pd.Index,
        stereotype_source: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Compute stereotypical ranking based on configuration.
        
        Parameters
        ----------
        X_scaled : ArrayLike
            Scaled feature matrix (for fallback to extremeness)
        index : pd.Index
            Row index
        stereotype_source : pd.Series, optional
            Pre-computed values to rank against (from df_original, metadata, or topology)
            
        Returns
        -------
        stereotype_rank : np.ndarray
            Scores in [0, 1] where 1 = closest to stereotype target

        Notes
        -----
        When ``stereotype_source`` is supplied, the returned rank is a monotone
        rescaling of the distance from the target on that one column and nothing
        else (Methods 2.3.1, eq. 26). Do not place an outcome-derived quantity in
        ``stereotype_column`` and then evaluate the rank against that outcome on
        held-out rows: the correlation is guaranteed by construction.
        """
        if stereotype_source is None:
            # BACKWARD COMPATIBLE: use extremeness
            X_dense = X_scaled.toarray() if (sp is not None and sp.isspmatrix(X_scaled)) else X_scaled
            s = np.max(np.abs(X_dense - 0.5), axis=1) * 2.0
            s_min, s_max = float(s.min()), float(s.max())
            if (s_max - s_min) > 1e-12:
                return (s - s_min) / (s_max - s_min)
            else:
                return np.zeros_like(s)
        
        # USER-DIRECTED: Rank toward specific target
        values = _stereotype_values_as_float(stereotype_source, self.stereotype_column)
        
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            warnings.warn("All stereotype values are NaN, using zeros")
            return np.zeros(len(values))
        
        # Compute target value
        if isinstance(self.stereotype_target, str):
            if self.stereotype_target == "min":
                target = np.nanmin(values)
            elif self.stereotype_target == "max":
                target = np.nanmax(values)
            else:
                raise ValueError(
                    f"stereotype_target must be 'min', 'max', or numeric value, "
                    f"got '{self.stereotype_target}'"
                )
        else:
            target = float(self.stereotype_target)
        
        # Rank by distance to target (inverted: 1 = closest, 0 = furthest)
        distances = np.abs(values - target)
        max_dist = np.nanmax(distances)
        
        # v0.8.1: this used to test max_dist against an ABSOLUTE 1e-12. A rank
        # is scale-free, but the stereotype column is in the user's own units,
        # so a column of concentrations or mole fractions could have perfect
        # structure and still fall under the threshold, collapsing every rank
        # to 1.0 with no warning. The threshold is now relative to the column's
        # own magnitude, and saying so is no longer optional.
        _finite = distances[np.isfinite(distances)]
        _scale = float(np.max(np.abs(_finite))) if _finite.size else 0.0
        _resolution = np.finfo(np.float64).eps * max(_scale, 1.0) * 8.0
        if max_dist > _resolution:
            stereotype_rank = 1.0 - (distances / max_dist)
        else:
            warnings.warn(
                "every value in stereotype_column %r lies at the target to "
                "within float64 resolution (spread %.3g), so stereotypical_rank "
                "is 1.0 for every sample and carries no information. If the "
                "column has real structure at this magnitude, rescale it before "
                "fitting." % (self.stereotype_column, max_dist),
                RuntimeWarning, stacklevel=3)
            stereotype_rank = np.ones_like(distances, dtype=np.float64)
        
        # Handle NaN entries
        stereotype_rank[~valid_mask] = 0.0
        
        return np.clip(stereotype_rank, 0.0, 1.0)


    def _get_stereotype_source_table(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract stereotype values from tabular data."""
        if self.stereotype_column is None:
            return None
        
        # Check if column exists in df (features or labels)
        if self.stereotype_column not in df.columns:
            raise ValueError(
                f"stereotype_column '{self.stereotype_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # v0.8.0: fail here, naming the column, rather than several frames deep
        # inside pandas once the fit is already under way.
        source = df[self.stereotype_column]
        _stereotype_values_as_float(source, self.stereotype_column)
        return source

    def _get_stereotype_source_text(self) -> Optional[pd.Series]:
        """Extract stereotype values from text metadata or keywords."""
        
        # Priority 1: User-specified column from metadata
        if self.stereotype_column is not None:
            if self.text_metadata_ is None:
                raise ValueError(
                    "stereotype_column specified but no text_metadata provided. "
                    "Pass text_metadata to fit_text() or use stereotype_keywords."
                )
            
            if self.stereotype_column not in self.text_metadata_.columns:
                raise ValueError(
                    f"stereotype_column '{self.stereotype_column}' not found in text_metadata. "
                    f"Available columns: {list(self.text_metadata_.columns)}"
                )
            
            return self.text_metadata_[self.stereotype_column]
        
        # Priority 2: Keyword-based scores
        if self.stereotype_keyword_scores_ is not None:
            return pd.Series(self.stereotype_keyword_scores_)
        
        # No stereotype specified
        return None

    def _compute_keyword_scores(
        self, 
        X_tfidf: "sp.spmatrix",
        corpus: List[str],
        keywords: List[str]
    ) -> np.ndarray:
        """
        Compute relevance scores for documents based on keyword TF-IDF sum.
        
        Parameters
        ----------
        X_tfidf : sparse matrix
            TF-IDF matrix (n_docs, n_vocab)
        corpus : List[str]
            Original documents (for fallback if keywords not in vocab)
        keywords : List[str]
            Keywords to compute relevance for
            
        Returns
        -------
        scores : np.ndarray
            Relevance score per document (n_docs,)
        """
        vocab = self.vectorizer_.vocabulary_
        
        # Find indices of keywords in vocabulary
        keyword_indices = []
        missing_keywords = []
        
        for kw in keywords:
            if kw in vocab:
                keyword_indices.append(vocab[kw])
            else:
                missing_keywords.append(kw)
        
        if missing_keywords:
            warnings.warn(
                f"Keywords not found in vocabulary: {missing_keywords}. "
                f"These will be ignored in stereotype computation."
            )
        
        if not keyword_indices:
            warnings.warn(
                "No stereotype keywords found in vocabulary. "
                "Using zero scores (equivalent to no stereotype)."
            )
            return np.zeros(X_tfidf.shape[0])
        
        # Sum TF-IDF scores for keyword columns
        keyword_indices = np.array(keyword_indices)
        X_keywords = X_tfidf[:, keyword_indices]
        scores = np.asarray(X_keywords.sum(axis=1)).ravel()
        
        return scores

    # ============================================================
    # Internals - Tables (numeric-only features)
    # ============================================================
    def _select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select numeric feature columns, excluding ID-like and non-informative columns.
        
        FIXED: Uniqueness heuristic only applied to integer/categorical columns.
        Continuous float columns are never excluded on uniqueness grounds, as high
        uniqueness is expected and meaningful for continuous measurements (e.g. LogP).
        """
        numeric_df = df.select_dtypes(include=[np.number])
        n = len(df)
        to_drop = set()
        
        for col in numeric_df.columns:
            series = numeric_df[col]
            nunique = series.nunique()
            
            # Exclude columns with ID-like names
            if isinstance(col, str):
                col_lower = col.lower()
                if (col_lower == 'id' or
                    col_lower.endswith('_id') or
                    col_lower.startswith('id_')):
                    to_drop.add(col)
                    if self.verbose:
                        print(f"  Dropping ID-like column: '{col}'")
                    continue
            
            # Exclude strictly monotonic columns (likely row indices)
            if nunique == n:
                vals = series.dropna().values
                if len(vals) == n:
                    if np.all(np.diff(vals) > 0) or np.all(np.diff(vals) < 0):
                        to_drop.add(col)
                        if self.verbose:
                            print(f"  Dropping strictly monotonic column: '{col}'")
                        continue
            
            # FIXED: Only apply uniqueness heuristic to integer columns
            # Float columns are exempt - high uniqueness is expected for continuous
            # measurements (e.g. LogP, molecular weight, solubility) and is NOT
            # indicative of an identifier
            is_float_col = np.issubdtype(series.dtype, np.floating)
            
            if not is_float_col and nunique >= 0.8 * n:
                to_drop.add(col)
                if self.verbose:
                    print(f"  Dropping high-uniqueness non-float column: '{col}'")
                continue
        
        feature_cols = [c for c in numeric_df.columns if c not in to_drop]
        
        if not feature_cols:
            raise DataTypicalError(
                "No numeric feature columns remain after preprocessing. "
                "Check label_columns and input data."
            )
        
        return numeric_df[feature_cols]

    def _preprocess_table_fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, ArrayLike]:
        # Store original df for stereotype computation
        self._df_original_fit = df.copy()
        
        # Separate label columns (preserve but don't use in NMF)
        if self.label_columns is not None:
            label_cols_present = [c for c in self.label_columns if c in df.columns]
            missing_labels = [c for c in self.label_columns if c not in df.columns]
            
            if missing_labels:
                warnings.warn(f"Label columns not found: {missing_labels}")
            
            if label_cols_present:
                self.label_df_ = df[label_cols_present].copy()
                df_for_features = df.drop(columns=label_cols_present)
            else:
                self.label_df_ = None
                df_for_features = df
        else:
            self.label_df_ = None
            df_for_features = df
        
        # Pick numeric features only
        feat_df = self._select_numeric_features(df_for_features)
        self.feature_columns_ = list(feat_df.columns)

        # v0.8.0: read and scale in float64, then cast the SCALED result to the
        # working dtype below. Casting the raw data first destroyed it whenever
        # the input was not already of order 1:
        #   * values above ~3.4e38 became inf, and MinMaxScaler then raised a
        #     bare "Input X contains infinity" from inside sklearn;
        #   * values below ~1.2e-38 underflowed to zero, every column read as
        #     constant, and the fit died with "No numeric feature columns
        #     remain", which points at the wrong thing entirely;
        #   * an informative feature carrying its signal in the 7th significant
        #     digit (say a measurement offset by 1000) lost that signal outright
        #     and was silently dropped as constant.
        # The scaled matrix lives in [0, 1], so storing THAT in float32 gives the
        # memory saving dtype is for, with none of the damage.
        X = feat_df.to_numpy(dtype=np.float64, copy=True)

        # Missingness report on features
        miss_frac = np.mean(pd.isna(feat_df), axis=0).to_numpy()
        self.missingness_ = {name: float(frac) for name, frac in zip(self.feature_columns_, miss_frac)}
        worst = np.max(miss_frac) if miss_frac.size else 0.0
        if worst > self.max_missing_frac:
            raise DataTypicalError(
                f"Missingness too high (max frac={worst:.3f} > threshold={self.max_missing_frac})."
            )

        # Deterministic imputer: per-feature median
        med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
        self.impute_median_ = med

        # v0.8.0: warn when a few extreme values own a feature's range.
        #
        # MinMax scaling maps the extremes to 0 and 1 and compresses everything
        # else towards one end. The origin is itself a corner of the unit
        # hypercube, so a compressed bulk sits *in* the corner region that the
        # archetypal corner term rewards, and the term stops discriminating.
        # Measured on synthetic data, an outlier at 3 to 5 standard deviations
        # still ranks above the cloud median on 6 of 6 seeds, but one at 30
        # standard deviations ranks *below* it on 0 of 6: the more extreme the
        # point, the less archetypal it is judged. The middle-98% share of the
        # full range separates the two regimes cleanly (about 0.75 for normal
        # data, 0.14 at 30 sd), so that is what is reported here.
        #
        # This is a property of the published measure, not a coding error, so
        # nothing about the arithmetic changes. It is worth saying out loud
        # because the regime is common in assay and biomarker data.
        if X.shape[0] >= 20 and X.shape[1] >= 1:
            _lo = np.nanpercentile(X, 1, axis=0)
            _hi = np.nanpercentile(X, 99, axis=0)
            _full = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
            _safe = np.where(_full > 0, _full, 1.0)
            _share = (_hi - _lo) / _safe
            _cramped = np.where(np.isfinite(_share) & (_share < 0.2))[0]
            if _cramped.size:
                _names = [self.feature_columns_[i] for i in _cramped[:5]]
                warnings.warn(
                    f"{_cramped.size} feature(s) have their range dominated by a "
                    f"few extreme values; the middle 98% of "
                    f"{_names}{' and more' if _cramped.size > 5 else ''} spans "
                    f"under a fifth of the full range. After MinMax scaling the "
                    "bulk of the rows is compressed against one end, which is "
                    "itself a corner of the unit cube, so the archetypal corner "
                    "term loses resolution there. In practice archetypal_rank "
                    "becomes sensitive to whether those columns are transformed: "
                    "on a real assay cohort, log-transforming them changed 12 of "
                    "the 20 most archetypal instances. Where one such feature "
                    "dominates the whole frame the ordering can invert outright, "
                    "with the most extreme rows scoring below the median. Fit "
                    "both ways and compare, or apply a log or rank transform to "
                    "those columns before fitting.",
                    RuntimeWarning, stacklevel=3
                )

        # Scale to [0,1]
        self.scaler_ = MinMaxScaler(copy=True, clip=True)
        X_scaled_full = self.scaler_.fit_transform(X).astype(self.dtype, copy=False)

        # Drop constant columns
        var = X_scaled_full.var(axis=0)
        keep_mask = var > 0.0
        self.keep_mask_ = keep_mask
        if not np.all(keep_mask):
            self.dropped_columns_ = [c for c, k in zip(self.feature_columns_, keep_mask) if not k]

            # v0.8.0: separate genuinely constant columns from ones that merely
            # fell below the scaler's resolution. MinMaxScaler treats any feature
            # whose range is under about 2e-15 as constant and leaves it at zero,
            # so a column with real structure at that magnitude is dropped as if
            # it carried no information. Previously both cases were silent unless
            # verbose was on, and when every column went this way the fit died
            # further downstream complaining about nmf_rank, which points at the
            # wrong thing entirely.
            raw_range = np.ptp(X, axis=0)
            truly_constant, below_resolution = [], []
            for name, keep, rng in zip(self.feature_columns_, keep_mask, raw_range):
                if keep:
                    continue
                (truly_constant if rng == 0.0 else below_resolution).append(name)

            if truly_constant:
                warnings.warn(
                    f"Dropped constant feature columns: {truly_constant}"
                )
            if below_resolution:
                warnings.warn(
                    f"Dropped feature columns whose values vary, but by less than "
                    f"the scaler can resolve (a range under ~2e-15): "
                    f"{below_resolution}. Their structure is real but is being "
                    f"discarded. Rescale them into a normal range, for example "
                    f"df[col] = df[col] / df[col].abs().max(), before fitting."
                )

            if not np.any(keep_mask):
                raise DataTypicalError(
                    "Every feature column was dropped as constant after scaling, "
                    f"so there is nothing left to fit. Truly constant: "
                    f"{truly_constant or 'none'}. Varying but below the scaler's "
                    f"resolution: {below_resolution or 'none'}. If the second list "
                    "is non-empty, the data carries real structure at a magnitude "
                    "MinMaxScaler treats as zero; rescale those columns into a "
                    "normal range and refit."
                )
        X_scaled = X_scaled_full[:, keep_mask]

        # Optional feature weights (length must match number of original numeric features)
        if self.feature_weights is not None:
            w = np.asarray(self.feature_weights, dtype=np.float64).ravel()
            if w.shape[0] != len(self.feature_columns_):
                warnings.warn("feature_weights length mismatch -- ignoring weights.")
            else:
                X_scaled = (X_scaled * w[keep_mask]).astype(self.dtype, copy=False)

        # L2 copy
        X_l2 = _l2_normalize_rows_dense(X_scaled.astype(np.float64))
        return X_scaled, X_l2

    def _preprocess_table_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, ArrayLike]:
        if any(v is None for v in (self.feature_columns_, self.impute_median_, self.keep_mask_, self.scaler_)):
            raise RuntimeError("Model not fitted.")
        # Align by feature column NAMES (order enforced from training)
        missing = [c for c in self.feature_columns_ if c not in df.columns]
        if missing:
            raise DataTypicalError(f"Missing required feature columns at transform: {missing}")
        feat_df = df[self.feature_columns_]
        # Ensure numeric
        if not all(np.issubdtype(t, np.number) for t in feat_df.dtypes):
            raise DataTypicalError("Non-numeric values present in feature columns at transform.")
        # v0.8.0: float64 here for the same reason as in _preprocess_table_fit.
        # The cast to the working dtype happens after scaling.
        X = feat_df.to_numpy(dtype=np.float64, copy=True)

        # Impute with training medians
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = np.take(self.impute_median_, inds[1])

        # v0.8.0: the fitted scaler clips to the training range, so anything
        # outside it is pinned to 0 or 1. Two samples far beyond the training
        # maximum, and far apart from each other, then receive identical ranks.
        # For a library whose purpose is identifying extremes that is exactly
        # the wrong silent failure, so say when it happens. The clipping itself
        # is kept: unclipped values break the [0, 1] geometry the archetypal
        # scores assume.
        _lo = np.asarray(self.scaler_.data_min_, dtype=np.float64)
        _hi = np.asarray(self.scaler_.data_max_, dtype=np.float64)
        _span = np.where((_hi - _lo) > 0, _hi - _lo, 1.0)
        _below = (X < _lo) & ~np.isnan(X)
        _above = (X > _hi) & ~np.isnan(X)
        _n_clipped = int(np.count_nonzero(_below | _above))
        if _n_clipped:
            _excess = np.maximum(
                np.max((_lo - X) / _span, axis=0, initial=0.0),
                np.max((X - _hi) / _span, axis=0, initial=0.0),
            )
            _worst_col = int(np.argmax(_excess))
            _rows = int(np.count_nonzero((_below | _above).any(axis=1)))
            warnings.warn(
                f"{_n_clipped} value(s) across {_rows} of {X.shape[0]} row(s) fall "
                f"outside the range seen during fit and are being clipped to it. "
                f"The worst is column "
                f"'{self.feature_columns_[_worst_col]}', which extends "
                f"{_excess[_worst_col]:.3g} training ranges beyond the training "
                f"limit. Clipped rows are indistinguishable from one another at "
                f"the boundary, so their ranks understate how extreme they are. "
                f"Refit on data covering this range if these rows matter.",
                RuntimeWarning, stacklevel=3
            )

        # Scale using fitted scaler; then drop constants via keep_mask_
        X_scaled_full = self.scaler_.transform(X).astype(self.dtype, copy=False)
        X_scaled = X_scaled_full[:, self.keep_mask_]

        # Optional weights
        if self.feature_weights is not None and len(self.feature_columns_) == np.asarray(self.feature_weights).shape[0]:
            X_scaled = (X_scaled * np.asarray(self.feature_weights)[self.keep_mask_]).astype(self.dtype, copy=False)

        X_l2 = _l2_normalize_rows_dense(X_scaled.astype(np.float64))
        return X_scaled, X_l2

    # ============================================================
    # Internals - Text
    # ============================================================
    def _preprocess_text_fit(
        self, 
        corpus: Iterable[str], 
        vectorizer: str,
        text_metadata: Optional[pd.DataFrame] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Preprocess text with optional metadata for stereotypes.
        
        Parameters
        ----------
        corpus : Iterable[str]
            Text documents
        vectorizer : str
            Vectorization method
        text_metadata : pd.DataFrame, optional
            External document-level properties (e.g., relevance scores, timestamps)
        """
        if vectorizer != "tfidf":
            raise NotImplementedError("Only TF-IDF supported in v0.4.")
        if sp is None:
            raise ImportError("scipy is required for text path.")
        
        corpus_list = list(corpus)
        n_docs = len(corpus_list)
        
        # Store metadata if provided
        if text_metadata is not None:
            if len(text_metadata) != n_docs:
                raise ValueError(
                    f"text_metadata length ({len(text_metadata)}) must match "
                    f"corpus length ({n_docs})"
                )
            self.text_metadata_ = text_metadata.copy()
        else:
            self.text_metadata_ = None
        
        # Fit TF-IDF vectorizer
        self.vectorizer_ = TfidfVectorizer()
        X_tfidf = self.vectorizer_.fit_transform(corpus_list)
        
        # Compute keyword-based stereotype if specified
        if self.stereotype_keywords is not None:
            self.stereotype_keyword_scores_ = self._compute_keyword_scores(
                X_tfidf, corpus_list, self.stereotype_keywords
            )
        else:
            self.stereotype_keyword_scores_ = None
        
        X_scaled_sp = _sparse_minmax_0_1_nonneg(X_tfidf)
        X_l2 = _sparse_l2_normalize_rows(X_scaled_sp)
        
        return X_scaled_sp, X_l2

    def _preprocess_text_transform(self, corpus: Iterable[str]) -> Tuple[ArrayLike, ArrayLike]:
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit_text first.")
        if sp is None:
            raise ImportError("scipy is required for text path.")
        X_tfidf = self.vectorizer_.transform(list(corpus))
        X_scaled_sp = _sparse_minmax_0_1_nonneg(X_tfidf)
        X_l2 = _sparse_l2_normalize_rows(X_scaled_sp)
        return X_scaled_sp, X_l2


    # ============================================================
    # Archetypal Analysis Methods (NEW in v0.7)
    # ============================================================
    
    def _fit_archetypal_aa(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        True archetypal analysis via PCHA.

        Backend policy (changed in v0.8.0)
        ----------------------------------
        ``archetypal_method='aa'``
            PCHA is required. If ``py_pcha`` is not installed, or the data cannot
            support it, or PCHA fails, this raises :class:`ConfigError`. Through
            v0.7.7 the method fell silently through to ConvexHull and then to NMF,
            with the only notice behind ``verbose``, so a fit reported as
            archetypal analysis could return NMF output with nothing to
            distinguish it. Any ``'aa'`` result produced by v0.7.7 or earlier in an
            environment without ``py_pcha`` is an NMF approximation and should be
            re-run.
        ``archetypal_method='auto'``
            The permissive cascade PCHA -> ConvexHull -> NMF, emitting a
            ``RuntimeWarning`` at each downgrade.
        ``archetypal_method='nmf'``
            Does not reach this method at all.

        The backend that actually ran is recorded in ``self.archetypal_backend_``
        (``'pcha'``, ``'convexhull'`` or ``'nmf'``) and in ``self.settings_``.

        MEMORY OPTIMIZED: Respects configured dtype while preserving input dtype
        when needed.
        """
        n_samples, n_features = X_scaled.shape
        strict = (self.archetypal_method == 'aa')

        # OPTIMIZED: Use configured dtype, but respect input if it's float64
        input_dtype = X_scaled.dtype
        if input_dtype == np.float64:
            target_dtype = np.float64  # Preserve float64 if input is float64
        elif self.dtype == 'float32':
            target_dtype = np.float32
        else:
            target_dtype = np.float64

        # Determine effective k
        k_max = min(n_samples, n_features)
        k_eff = min(self.nmf_rank, k_max)

        # ---- PCHA: the method the manuscript names as primary ----
        if PCHA is None:
            if strict:
                raise ConfigError(
                    "archetypal_method='aa' requires the py_pcha package, which is "
                    "not installed, and v0.8.0 no longer substitutes another method "
                    "silently. Install it with: pip install py_pcha. Or pass "
                    "archetypal_method='auto' to allow the ConvexHull/NMF cascade, "
                    "or archetypal_method='nmf' to ask for NMF directly."
                )
            warnings.warn(
                "py_pcha is not installed, so archetypal_method='auto' cannot run "
                "PCHA and will fall back to ConvexHull or NMF. The result will be an "
                "approximation of archetypal analysis, not archetypal analysis. "
                "Check archetypal_backend_ after fitting.",
                RuntimeWarning, stacklevel=2
            )
        elif k_eff < 2:
            message = (
                f"PCHA needs at least 2 archetypes, but nmf_rank={self.nmf_rank} with "
                f"data shape {X_scaled.shape} gives k_eff={k_eff}. Increase nmf_rank, "
                "supply more samples or features, or pass archetypal_method='auto' "
                "or 'nmf'."
            )
            if strict:
                raise ConfigError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            try:
                # PCHA requires float64 internally
                X_T = X_scaled.astype(np.float64).T.copy()
                X_min = X_T.min()
                if X_min < 0:
                    X_T = X_T - X_min + 1e-10

                if self.verbose:
                    print(f"  Computing {k_eff} archetypes using PCHA (stable in {n_features}D)...")

                XC, S, C, SSE, varexpl = PCHA(X_T, noc=k_eff, delta=0.0)

                if self.verbose:
                    print(f"  PCHA converged, variance explained: {varexpl:.2%}")

                # Convert to ndarray (PCHA returns matrix objects)
                W = np.asarray(S.T, dtype=target_dtype)
                H = np.asarray(XC.T, dtype=target_dtype)

                # Validate dimensions with detailed error messages
                if W.shape != (n_samples, k_eff):
                    raise ValueError(f"PCHA W shape error: got {W.shape}, expected ({n_samples}, {k_eff})")
                if H.shape != (k_eff, n_features):
                    raise ValueError(f"PCHA H shape error: got {H.shape}, expected ({k_eff}, {n_features})")

                self.nmf_model_ = None
                self.reconstruction_error_ = float(SSE)
                self.n_archetypes_ = k_eff
                self.archetypal_backend_ = 'pcha'
                return W, H

            except Exception as e:
                if strict:
                    raise ConfigError(
                        f"archetypal_method='aa' was requested but PCHA failed: {e}. "
                        "v0.8.0 no longer substitutes another method silently. Pass "
                        "archetypal_method='auto' to fall back to ConvexHull or NMF, "
                        "or archetypal_method='nmf' to ask for NMF directly."
                    ) from e
                if self.verbose:
                    print(f"  PCHA failed ({e}), trying ConvexHull")
                warnings.warn(
                    f"PCHA failed ({e}); archetypal_method='auto' is falling back to "
                    "ConvexHull or NMF. Check archetypal_backend_ after fitting.",
                    RuntimeWarning, stacklevel=2
                )

        # ---- ConvexHull fallback (permissive mode only, low dimensions) ----
        if ConvexHull is not None and cdist is not None and n_features <= 20:
            try:
                # ConvexHull needs float64
                X_hull = X_scaled.astype(np.float64)
                hull = ConvexHull(X_hull)
                boundary_indices = np.unique(hull.simplices.ravel())
                n_archetypes = len(boundary_indices)

                if self.verbose:
                    print(f"  Found {n_archetypes} archetypes on convex hull")

                W = np.zeros((n_samples, n_archetypes), dtype=target_dtype)
                for i in range(n_samples):
                    point = X_hull[i:i+1]
                    boundary_points = X_hull[boundary_indices]
                    distances = cdist(point, boundary_points).ravel()
                    weights = 1.0 / (distances + 1e-6)
                    W[i, :] = weights / weights.sum()

                H = np.asarray(X_scaled[boundary_indices], dtype=target_dtype)
                self.nmf_model_ = None
                self.reconstruction_error_ = None
                self.n_archetypes_ = n_archetypes
                self.archetypal_backend_ = 'convexhull'
                return W, H
            except Exception as e:
                if self.verbose:
                    print(f"  ConvexHull failed ({e}), using NMF")

        # ---- Final fallback: NMF ----
        # Loud on purpose: what comes back is not archetypal analysis.
        warnings.warn(
            f"archetypal_method='{self.archetypal_method}' fell back to NMF. The "
            "returned archetypes are an NMF approximation, not archetypal analysis; "
            "archetypal_backend_ is set to 'nmf'.",
            RuntimeWarning, stacklevel=2
        )
        if self.verbose:
            print(f"  Using NMF fallback")
        return self._fit_archetypal_nmf(X_scaled)


    def _fit_archetypal_nmf(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast NMF-based approximation of archetypes.
        
        MEMORY OPTIMIZED: Respects configured dtype.
        """
        # Ensure non-negative (NMF requirement)
        X_nonneg = np.maximum(X_scaled.astype(np.float64), 0)
        
        # Determine effective rank
        k_eff = min(self.nmf_rank, X_nonneg.shape[0], X_nonneg.shape[1])
        if k_eff < 1:
            raise DataTypicalError(
                f"NMF requires at least 1 component, but data shape {X_nonneg.shape} "
                f"and nmf_rank={self.nmf_rank} yield k_eff={k_eff}. "
                "Provide more samples or reduce nmf_rank."
            )

        # OPTIMIZED: Determine target dtype
        input_dtype = X_scaled.dtype
        if input_dtype == np.float64:
            target_dtype = np.float64
        elif self.dtype == 'float32':
            target_dtype = np.float32
        else:
            target_dtype = np.float64
        
        if self.verbose:
            print(f"\nFitting archetypes: NMF (k={k_eff})")
        
        # Fit NMF with convergence warning suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            nmf = NMF(
                n_components=k_eff,
                init='nndsvd',
                max_iter=self.max_iter_nmf,
                tol=self.tol_nmf,
                random_state=self.random_state
            )
            W = nmf.fit_transform(X_nonneg)
            H = nmf.components_
        
        # Store model and metadata
        self.nmf_model_ = nmf
        self.reconstruction_error_ = float(nmf.reconstruction_err_)
        self.n_archetypes_ = k_eff
        self.archetypal_backend_ = 'nmf'
        
        # OPTIMIZED: Ensure output matches target dtype
        W = W.astype(target_dtype, copy=False)
        H = H.astype(target_dtype, copy=False)
        
        return W, H

    # ============================================================
    # Internals - Fit Components (NMF + Prototypes)
    # ============================================================
    def _fit_components(self, X_scaled: ArrayLike, X_l2: ArrayLike, index: pd.Index) -> None:
        """
        Fit archetypal and prototypical components.
        
        MEMORY OPTIMIZED: Explicit cleanup of large temporaries.
        
        Parameters
        ----------
        X_scaled : ArrayLike
            Scaled feature matrix [0, 1]
        X_l2 : ArrayLike
            L2-normalized feature matrix
        index : pd.Index
            Sample index
        """
        run_arch = self.selected_significance is None or self.selected_significance == "archetypal"
        run_proto = self.selected_significance is None or self.selected_significance == "prototypical"

        # ---- ARCHETYPAL ANALYSIS (NMF or AA)
        if run_arch:
            X_euc_fit = X_scaled.toarray().astype(np.float64, copy=False) \
                if (sp is not None and sp.isspmatrix(X_scaled)) else np.asarray(X_scaled, dtype=np.float64)

            if self.verbose:
                method_name = {
                    'aa': "Archetypal Analysis (PCHA, required)",
                    'auto': "Archetypal Analysis (PCHA -> ConvexHull -> NMF)",
                }.get(self.archetypal_method, "NMF Approximation")
                print(f"\nFitting archetypal: {method_name}")

            # Call appropriate method
            if self.archetypal_method in ('aa', 'auto'):
                W, H = self._fit_archetypal_aa(X_euc_fit)
            else:  # 'nmf'
                W, H = self._fit_archetypal_nmf(X_euc_fit)

            # Store with validation and correct dtype
            input_dtype = X_euc_fit.dtype
            if input_dtype == np.float64:
                target_dtype = np.float64
            elif self.dtype == 'float32':
                target_dtype = np.float32
            else:
                target_dtype = np.float64

            self.W_ = W.astype(target_dtype, copy=False)
            self.H_ = H.astype(target_dtype, copy=False)
            self.n_archetypes_ = self.H_.shape[0]

            # Final validation
            n_samples, n_features = X_euc_fit.shape
            assert self.W_.shape == (n_samples, self.n_archetypes_), \
                f"W_ dimension mismatch: {self.W_.shape} vs ({n_samples}, {self.n_archetypes_})"
            assert self.H_.shape == (self.n_archetypes_, n_features), \
                f"H_ dimension mismatch: {self.H_.shape} vs ({self.n_archetypes_}, {n_features})"

            # MEMORY CLEANUP: Free temporaries (stored in self.W_, self.H_)
            _cleanup_memory(W, H, X_euc_fit)

            if self.verbose:
                print(f"  Stored: W ={self.W_.shape}, H ={self.H_.shape}, n_archetypes ={self.n_archetypes_}")
        else:
            self.W_ = None
            self.H_ = None
            self.n_archetypes_ = 0
            if self.verbose:
                print(f"\nSkipping archetypal (selected_significance='{self.selected_significance}')")
        
        # ---- Prepare scaled dense & L2 copies
        X_euc = X_scaled.toarray().astype(np.float64, copy=False) \
            if (sp is not None and sp.isspmatrix(X_scaled)) else np.asarray(X_scaled, dtype=np.float64)
        Xl2 = X_l2.toarray().astype(np.float64, copy=False) \
            if (sp is not None and sp.isspmatrix(X_l2)) else np.asarray(X_l2, dtype=np.float64)
        n = X_euc.shape[0]
        
        # ---- Helper: archetypal "cornerness" score
        def _corner_scores(Xe: np.ndarray) -> np.ndarray:
            eps = 1e-12
            col_min, col_max = Xe.min(axis=0), Xe.max(axis=0)
            hits_edge = (col_min <= eps) & (col_max >= 1.0 - eps)
            idxs = np.where(hits_edge)[0]
            if idxs.size >= 2:
                var = Xe[:, idxs].var(axis=0)
                take = idxs[np.argsort(-var)[:2]]
            else:
                var = Xe.var(axis=0)
                take = np.argsort(-var)[:2] if Xe.shape[1] >= 2 else np.array([0])
            X2 = Xe[:, take] if take.size else Xe[:, :1]
            m = np.minimum(X2, 1.0 - X2)
            dmin = np.sqrt(np.sum(m * m, axis=1))
            denom = math.sqrt(X2.shape[1]) if X2.shape[1] >= 1 else 1.0
            return 1.0 - np.clip(dmin / denom, 0.0, 1.0)
        
        # ---- Helper: kNN density (cosine)
        def _knn_density_cosine(Xl2_arr: np.ndarray, k: int = 10, clip_neg: bool = True) -> np.ndarray:
            k = max(1, min(k, max(1, n - 1)))
            # Chunked row-wise dot product: caps memory at chunk_size x n instead of n x n.
            # Each block is at most ~256 MB regardless of n.
            chunk_size = max(1, min(n, int(256 * 1024 * 1024 // max(1, 8 * n))))
            top_k_means = np.zeros(n)
            for row_start in range(0, n, chunk_size):
                row_end = min(n, row_start + chunk_size)
                block = Xl2_arr[row_start:row_end] @ Xl2_arr.T  # (chunk, n)
                if clip_neg:
                    np.maximum(block, 0.0, out=block)
                diag_r = np.arange(row_end - row_start)
                block[diag_r, diag_r + row_start] = 0.0
                topk_block = np.partition(block, -k, axis=1)[:, -k:]
                top_k_means[row_start:row_end] = topk_block.mean(axis=1)
            m = top_k_means.mean()
            return top_k_means / m if m > 0 else np.ones_like(top_k_means)
        
        # ---- Build forbidden set from top archetypal (if enabled)
        disallow_overlap = bool(getattr(self, "disallow_overlap", False))
        overlap_alpha = float(getattr(self, "overlap_alpha", 0.0))
        forbidden = set()
        if run_arch and disallow_overlap and overlap_alpha > 0.0:
            corner = _corner_scores(X_euc)
            m = max(1, min(n - 1, int(math.ceil(overlap_alpha * n))))
            order = np.argsort(-corner)
            forbidden = set(order[:m])

        if run_proto:
            # ---- Compute kNN density for prototype selection
            dens = _knn_density_cosine(Xl2, k=10)

            # ---- Prototypes via CELF with optional density weighting
            if self.verbose:
                print(f"\nFitting prototypes: Facility Location (k={self.n_prototypes})")

            # Determine if density weighting is enabled
            density_weighted_fl = bool(getattr(self, "density_weighted_fl", False))
            density_k = int(getattr(self, "density_k", 10))
            density_clip_neg = bool(getattr(self, "density_clip_neg", True))
            weights = dens if density_weighted_fl else None

            # Run facility location selector (it handles similarity matrix internally)
            selector = FacilityLocationSelector(
                n_prototypes=self.n_prototypes,
                deterministic=self.deterministic,
                speed_mode=self.speed_mode,
                verbose=self.verbose
            )
            P_idx, mg = selector.select(Xl2, weights=weights, forbidden=forbidden)

            # Optional auto-k (Kneedle)
            knee = None
            if self.auto_n_prototypes == "kneedle" and mg.size >= 2:
                requested = mg.size
                knee = self._kneedle(mg)
                if knee is not None and knee > 0:
                    P_idx = P_idx[:knee]
                    mg = mg[:knee]

                    # v0.8.0: facility-location gains decay steeply, so the
                    # first prototype often carries most of the coverage and
                    # Kneedle lands on 1. Truncating a requested set of 15 down
                    # to a single prototype changes every prototypical result,
                    # and it used to happen without a word.
                    if knee < 2 or knee < 0.25 * requested:
                        warnings.warn(
                            f"auto_n_prototypes='kneedle' cut the prototype set "
                            f"from {requested} to {knee}. Facility-location "
                            "gains fall away quickly, so the knee often sits at "
                            "the first prototype. If that is too few, drop "
                            "auto_n_prototypes and set n_prototypes yourself.",
                            RuntimeWarning, stacklevel=3
                        )

            self.prototype_indices_ = P_idx
            self.prototype_rows_ = index.to_numpy()[P_idx]
            self.prototype_features_ = X_euc[P_idx].copy()
            self.prototype_features_l2_ = Xl2[P_idx].copy()
            self.marginal_gains_ = mg

            # v0.8.0: knee_ used to be assigned the Kneedle result and then
            # overwritten unconditionally by the second-difference heuristic
            # below, so the value that actually truncated the selection was
            # never reported. When Kneedle ran, knee_ now reports what it chose,
            # which is also len(prototype_indices_). Otherwise it keeps the
            # second-difference diagnostic it has always carried.
            if knee is not None:
                self.knee_ = knee
            elif len(mg) > 2:
                diffs = np.diff(mg)
                if len(diffs) > 1:
                    diffs2 = np.diff(diffs)
                    self.knee_ = int(np.argmax(np.abs(diffs2)) + 1)
                else:
                    self.knee_ = 1
            else:
                self.knee_ = len(mg)

            # Training-time assignments & coverage
            best_cos, proto_label = self._assignments_cosine(Xl2, P_idx)
            self.assignments_ = proto_label
            self.coverage_ = best_cos

            if self.verbose:
                print(f"  Selected {len(P_idx)} prototypes, knee at {self.knee_}")
        else:
            self.prototype_indices_ = None
            self.prototype_rows_ = None
            self.prototype_features_ = None
            self.prototype_features_l2_ = None
            self.marginal_gains_ = np.array([])
            self.assignments_ = np.zeros(n, dtype=int)
            self.coverage_ = np.zeros(n)
            self.knee_ = 0
            if self.verbose:
                print(f"\nSkipping prototypical (selected_significance='{self.selected_significance}')")

        # ---- Stereotypes (verbose output)
        if self.verbose:
            if self.stereotype_column is not None:
                target_str = f"'{self.stereotype_target}'" if isinstance(self.stereotype_target, str) else f"{self.stereotype_target}"
                print(f"\nStereotypical configuration:")
                print(f"  Target column: '{self.stereotype_column}'")
                print(f"  Target value: {target_str}")
                
                # Show target distribution if we have the data
                # v0.8.0: _df_original_fit is None for text and graph fits, so the
                # hasattr check alone let this verbose branch dereference None.
                if (getattr(self, '_df_original_fit', None) is not None
                        and self.stereotype_column in self._df_original_fit.columns):
                    stereo_vals = self._df_original_fit[self.stereotype_column]
                    print(f"  Column range: [{stereo_vals.min():.2f}, {stereo_vals.max():.2f}]")
                    
                    if isinstance(self.stereotype_target, str):
                        if self.stereotype_target == 'max':
                            print(f"  Targeting samples with maximum {self.stereotype_column}")
                        elif self.stereotype_target == 'min':
                            print(f"  Targeting samples with minimum {self.stereotype_column}")
                    else:
                        distance_to_target = abs(stereo_vals - self.stereotype_target).mean()
                        print(f"  Mean distance to target: {distance_to_target:.2f}")
            else:
                print(f"\nStereotypical: Not configured (using feature extremeness)")

    # ============================================================
    # Internals - Scoring with fitted artifacts
    # ============================================================
    def _score_with_fitted(
        self,
        X_scaled: ArrayLike,
        X_l2: ArrayLike,
        index: pd.Index,
        stereotype_source: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Score data with fitted artifacts.
        
        CRITICAL: This method must handle dimension matching correctly for transform.
        MEMORY OPTIMIZED: Cleanup large temporaries during transform.
        """
        run_arch  = self.selected_significance is None or self.selected_significance == "archetypal"
        run_proto = self.selected_significance is None or self.selected_significance == "prototypical"
        run_stereo = self.selected_significance is None or self.selected_significance == "stereotypical"

        if run_arch and (self.W_ is None or self.H_ is None):
            raise RuntimeError(
                "Archetypal scores requested but archetypes were not fitted. "
                "If fit() was called with selected_significance='prototypical' or "
                "'stereotypical', call fit() again with selected_significance=None or "
                "'archetypal' before requesting archetypal scores."
            )
        if run_proto and self.prototype_indices_ is None:
            raise RuntimeError(
                "Prototypical scores requested but prototypes were not fitted. "
                "If fit() was called with selected_significance='archetypal' or "
                "'stereotypical', call fit() again with selected_significance=None or "
                "'prototypical' before requesting prototypical scores."
            )

        # ---- Archetypal projections
        X_for_transform = X_scaled.astype(np.float64) if (sp is not None and sp.isspmatrix(X_scaled)) \
            else np.asarray(X_scaled, dtype=np.float64)

        n_samples_transform = X_for_transform.shape[0]

        if run_arch:
            n_archetypes = self.n_archetypes_
            n_features_model = self.H_.shape[1]
            n_features_transform = X_for_transform.shape[1]
            if n_features_transform != n_features_model:
                raise ValueError(
                    f"Feature dimension mismatch: transform data has {n_features_transform} features, "
                    f"but model was trained with {n_features_model} features"
                )
        else:
            n_archetypes = 0
        
        W = None
        if run_arch:
            if self.nmf_model_ is not None:
                # NMF method: use fitted model to transform
                W = self.nmf_model_.transform(X_for_transform)
            else:
                # AA method: compute weights from H using least squares
                H = self.H_

                assert H.shape == (n_archetypes, n_features_model), \
                    f"H dimension error: {H.shape} vs ({n_archetypes}, {n_features_model})"

                HHT = H @ H.T
                assert HHT.shape == (n_archetypes, n_archetypes), \
                    f"HHT dimension error: {HHT.shape} vs ({n_archetypes}, {n_archetypes})"

                HHT_inv = np.linalg.pinv(HHT + 1e-6 * np.eye(HHT.shape[0]))
                W = X_for_transform @ H.T @ HHT_inv
                _cleanup_memory(HHT, HHT_inv)
                W = np.maximum(W, 0)

            assert W.shape == (n_samples_transform, n_archetypes), \
                f"W dimension error: {W.shape} vs ({n_samples_transform}, {n_archetypes})"

            W_row_sum = W.sum(axis=1, keepdims=True)

            # v0.8.0: a row whose archetype weights sum to zero has no defined
            # membership. After MinMax scaling that is exactly the row sitting
            # at the origin, which is the row that was the minimum in every
            # retained feature. Both NMF and the least-squares AA route return
            # W = 0 there, the guard below turns it into an all-zero W_norm, and
            # arch_wmax becomes 0. The consequence is the opposite of what the
            # measure intends: the most extreme row in the low direction lands
            # at the very bottom of archetypal_rank, below every other row, and
            # only the 0.3-weighted corner term keeps its score off the floor.
            #
            # The arithmetic is left exactly as it was, because changing it
            # would move every archetypal rank ever computed. What is new is
            # that it no longer happens in silence.
            _degenerate = (W_row_sum.ravel() == 0.0)
            if _degenerate.any():
                _n_bad = int(np.count_nonzero(_degenerate))
                warnings.warn(
                    f"{_n_bad} row(s) have zero archetype membership, which "
                    "happens when a row is the minimum in every retained "
                    "feature and therefore sits at the origin of the scaled "
                    "space. Their archetypal_rank falls to the bottom of the "
                    "column even though they are extreme points, because the "
                    "membership term cannot be defined for them. Treat those "
                    "rows' archetypal scores as undefined rather than low.",
                    RuntimeWarning, stacklevel=3
                )

            W_row_sum[W_row_sum == 0.0] = 1.0
            W_norm = W / W_row_sum
            arch_wmax = W_norm.max(axis=1)
            _cleanup_memory(W_norm, W_row_sum)
        
        X_euc = None
        if run_arch or run_proto:
            X_euc = X_scaled.toarray().astype(np.float64, copy=False) \
                if (sp is not None and sp.isspmatrix(X_scaled)) else np.asarray(X_scaled, dtype=np.float64)

        if run_proto:
            # ---- Prototypes: cosine assignment
            P_idx = self.prototype_indices_
            best_cos, proto_label = self._assignments_cosine(X_l2, P_idx)

            # Use stored prototype feature vectors so transform on new data is correct
            if self.prototype_features_ is not None:
                P_mat = self.prototype_features_
            else:
                P_mat = X_euc[P_idx]

            best_euc = _euclidean_min_to_set_dense(X_euc, P_mat, max_memory_mb=self.max_memory_mb)
            _cleanup_memory(P_mat)

            norm95 = np.percentile(best_euc, 95) or 1.0
            proto_d_norm95 = np.clip(best_euc / norm95, 0.0, 1.0)

        # ---- Compute ranks
        if run_arch:
            eps = 1e-12
            col_min = X_euc.min(axis=0)
            col_max = X_euc.max(axis=0)
            hits_edge = (col_min <= eps) & (col_max >= 1.0 - eps)
            idxs = np.where(hits_edge)[0]
            if idxs.size >= 2:
                var = X_euc[:, idxs].var(axis=0)
                take = idxs[np.argsort(-var)[:2]]
            else:
                var = X_euc.var(axis=0)
                take = np.argsort(-var)[:2] if X_euc.shape[1] >= 2 else np.array([0])
            X2 = X_euc[:, take] if take.size else X_euc[:, :1]
            m = np.minimum(X2, 1.0 - X2)
            dmin = np.sqrt(np.sum(m * m, axis=1))
            # NOTE (v0.8.0): m is min(x, 1-x) and so is at most 0.5, which makes
            # the largest possible dmin 0.5*sqrt(d). Dividing by sqrt(d) rather
            # than 0.5*sqrt(d) therefore confines corner_score to [0.5, 1.0]
            # instead of [0, 1], and the 0.3-weighted term contributes between
            # 0.15 and 0.30 rather than between 0 and 0.30. Left as published,
            # since changing the normaliser would move every archetypal rank.
            denom = math.sqrt(X2.shape[1]) if X2.shape[1] >= 1 else 1.0
            corner_score = 1.0 - np.clip(dmin / denom, 0.0, 1.0)
            archetypal_score = arch_wmax * 0.7 + corner_score * 0.3
            _cleanup_memory(X2, col_min, col_max, corner_score)
        else:
            archetypal_score = np.full(n_samples_transform, np.nan)

        prototypical_score = (1.0 - proto_d_norm95) * 0.5 + best_cos * 0.5 \
            if run_proto else np.full(n_samples_transform, np.nan)

        stereotypical_scores = self._compute_stereotypical_rank(X_scaled, index, stereotype_source) \
            if run_stereo else np.full(n_samples_transform, np.nan)
        
        # ---- Build output DataFrame (only keep rank columns)
        out = pd.DataFrame(
            {
                "archetypal_rank": np.round(archetypal_score, 10),
                "prototypical_rank": np.round(prototypical_score, 10),
                "stereotypical_rank": np.round(stereotypical_scores, 10),
            },
            index=index,
        )
        
        # MEMORY CLEANUP: Force GC before returning (transform often called repeatedly)
        _cleanup_memory(X_for_transform, X_euc, W, force_gc=True)
        
        return out

    # ------------------------------------------------------------
    def _assignments_cosine(
        self,
        X_l2: ArrayLike,
        prototype_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity assignments to prototypes.

        OPTIMIZED: Uses JIT-compiled function for 2-3x speedup.
        Uses stored prototype L2 rows so transform on new data is correct.
        """
        # Convert to dense if needed
        Xl2_dense = X_l2.toarray() if (sp is not None and sp.isspmatrix(X_l2)) else np.asarray(X_l2, dtype=np.float64)
        # Use stored L2 prototype rows (fitted training rows); fall back to indexing
        # current data only when prototype_features_l2_ is not available (e.g. legacy).
        if self.prototype_features_l2_ is not None:
            P_l2 = self.prototype_features_l2_
        else:
            P_l2 = Xl2_dense[prototype_indices]
        
        n_samples = Xl2_dense.shape[0]
        n_protos = len(prototype_indices)
        
        # OPTIMIZED: Use JIT for small to medium datasets
        if n_samples * n_protos < 1000000:
            sims = _cosine_similarity_jit(Xl2_dense, P_l2)
        else:
            # For very large datasets, use numpy (better for huge matrices)
            sims = Xl2_dense @ P_l2.T
            np.maximum(sims, 0.0, out=sims)
        
        best_idx = sims.argmax(axis=1).astype(int)
        best_sim = sims[np.arange(len(sims)), best_idx]
        
        return best_sim, best_idx

    def _kneedle(self, gains: np.ndarray) -> Optional[int]:
        U = np.cumsum(gains)
        if U[-1] == 0.0:
            return None
        U_norm = U / U[-1]
        k = gains.size
        x = np.linspace(1 / k, 1.0, k)
        diff = U_norm - x
        return int(np.argmax(diff)) + 1

    def _record_settings(self, tc: _ThreadControl):
        self.settings_ = {
            "deterministic": bool(self.deterministic),
            "speed_mode": bool(self.speed_mode),
            "thread_limit": tc.effective_limit,
            "random_state": int(self.random_state),
            "dtype": str(self.dtype),
            "max_memory_mb": int(self.max_memory_mb),
            # v0.8.0: serialise the requested method alongside the backend that
            # actually ran, so an archived fit can be audited after the fact.
            "archetypal_method": self.archetypal_method,
            "archetypal_backend": self.archetypal_backend_,
        }


__all__ = [
    "DataTypical",
    "exact_formative_archetypal",
    "exact_formative_prototypical",
    "exact_formative_stereotypical",
    "FacilityLocationSelector",
    "DataTypicalError",
    "ConfigError",
    "MemoryBudgetError",
]
