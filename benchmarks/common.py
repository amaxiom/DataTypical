"""
common.py -- Shared utilities for DataTypical structural-significance
benchmarks (v0.7.7).

Why this module exists
----------------------
The original Appendix A/C experiments compared DataTypical's *model-free,
label-free* formative score against significance methods that require a model
and/or labels (Isolation Forest, influence functions, KNN-Shapley), scored on
geometric metrics -- one of which (ARE) was built from DataTypical's own
archetypes. That stacks the deck. This module re-does the comparison cleanly:

  * Like-for-like comparator: coreset / facility-location (the only other model-free,
    geometric, label-free significance score). This is the like-for-like test.
  * Reference methods: Isolation Forest, influence functions, KNN-Shapley are
    kept, but reported SEPARATELY and clearly labelled as different-objective
    (model/label dependent) -- they answer "significant for predicting y under
    model M?", not "structurally load-bearing in X?".
  * Metrics are model-free: FLC (coverage) and kNN reconstruction (fidelity).
    The archetype-based ARE is intentionally dropped (it is not independent of
    DataTypical).
  * A controlled synthetic dataset with KNOWN structure-defining points lets us
    validate formative recovery against ground truth and gives a regime where
    the effect is actually detectable (unlike the saturated large UCI sets).
  * Effect size is quantified as a degradation AUC with mean +/- std over seeds,
    not eyeballed from near-flat curves.

Plot styling (applied everywhere): viridis colours, legend outside the axes,
standard (unrotated) tick labels, autoscaled axes (log where values span
orders of magnitude).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# [1] Plot styling
# ============================================================

def apply_rcparams():
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.weight":      "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.size":        12,
        "axes.titlesize":   12,
        "axes.labelsize":   12,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "figure.dpi":       130,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "image.cmap":       "viridis",
    })


def viridis_colors(n, lo=0.05, hi=0.92):
    """n evenly spaced colours sampled from viridis (avoids the dark/yellow ends)."""
    if n <= 1:
        return [plt.cm.viridis(0.5)]
    return [plt.cm.viridis(x) for x in np.linspace(lo, hi, n)]


def style_axes(ax, xlabel=None, ylabel=None, title=None, rotate_x=False):
    """Apply shared axis styling. Autoscales (no forced limits).

    rotate_x defaults to False: keep the standard horizontal orientation. Only
    pass rotate_x=True when tick labels are long/dense enough to overlap; in that
    case they are rotated 90 degrees and centred below their tick. For long
    categorical labels prefer a horizontal bar chart (no rotation needed at all).
    """
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    if title:
        ax.set_title(title, pad=6)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    if rotate_x:
        # 90 degrees, centred horizontally under the tick, anchored below the axis
        ax.tick_params(axis="x", which="both", pad=3)
        for lab in ax.get_xticklabels():
            lab.set_rotation(90)
            lab.set_ha("center")
            lab.set_va("top")
            lab.set_rotation_mode("anchor")


def legend_outside(ax, ncol=1, **kw):
    """Place the legend to the right of the axes so it never obscures the data."""
    return ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                     frameon=False, ncol=ncol, **kw)


# ============================================================
# [2] Controlled synthetic dataset with known structure
# ============================================================

def make_synthetic_archetypal(n=1000, d=8, k=6, n_corner_per=3,
                              noise=0.02, dirichlet_alpha=0.4, seed=0,
                              n_distractor_clusters=2, n_distractor_per=25):
    """
    Generate data with KNOWN structure-defining points and KNOWN distractors.

    k true archetypes are placed at distinct axis extremes (guaranteed convex-hull
    vertices). A small set of 'corner' points sits next to each archetype -- these
    are the ground-truth structure-defining instances. The remaining points are
    convex (Dirichlet) mixtures of the archetypes -- redundant interior mass.

    Because the corners are few and non-redundant, removing them measurably
    degrades coverage/fidelity, so structural significance is detectable here (unlike
    the saturated large UCI sets).

    Distractor outlier clusters (anomalous but individually redundant)
    -------------------------------------------------------------------
    n_distractor_clusters tight clusters of n_distractor_per points each are
    placed OFF the archetype simplex (at 0.65*(A_i + A_j), so their coordinate
    sum exceeds the simplex total -- a clear density anomaly that Isolation
    Forest flags). But because n_distractor_per points share each location, any
    single member is expendable: its cluster mates pin down the same geometry,
    so NO individual distractor is structurally significant. They separate
    "anomalous" from "load-bearing": an anomaly detector should rank them at
    the top, while a per-instance structural-significance score should not.

    Returns
    -------
    df : DataFrame with feature columns f0..f{d-1} and an integer 'label' column
         (the dominant archetype, a structure-linked label for reference methods).
    is_structural : bool ndarray (n,)  True for the ground-truth corner points.
    is_distractor : bool ndarray (n,)  True for the distractor-cluster points.
    """
    rng = np.random.default_rng(seed)

    # k archetypes at distinct extremes (scaled axis vertices + a couple of random
    # extreme combinations if k > d).
    A = np.zeros((k, d), dtype=np.float64)
    scale = 6.0
    for i in range(k):
        if i < d:
            A[i, i] = scale
        else:
            j = rng.choice(d, size=2, replace=False)
            A[i, j] = scale
    A += rng.normal(0.0, 0.05, size=A.shape)

    n_corner = k * n_corner_per
    n_corner = min(n_corner, n)
    n_distr = n_distractor_clusters * n_distractor_per
    n_distr = min(n_distr, max(0, n - n_corner))
    n_interior = n - n_corner - n_distr

    # Corner (structure-defining) points: tight cluster at each archetype.
    corner_X, corner_lab = [], []
    for i in range(k):
        m = n_corner_per if (i + 1) * n_corner_per <= n_corner else max(0, n_corner - i * n_corner_per)
        if m <= 0:
            break
        corner_X.append(A[i] + rng.normal(0.0, noise * scale, size=(m, d)))
        corner_lab.append(np.full(m, i))
    corner_X = np.vstack(corner_X)
    corner_lab = np.concatenate(corner_lab)

    # Distractor clusters: off-simplex anomalies made of mutually redundant points.
    distr_X, distr_lab = [], []
    if n_distr > 0 and n_distractor_clusters > 0:
        all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        sel = rng.choice(len(all_pairs), size=min(n_distractor_clusters, len(all_pairs)),
                         replace=False)
        remaining = n_distr
        for c, p in enumerate(sel):
            i, j = all_pairs[p]
            m = min(n_distractor_per, remaining)
            if m <= 0:
                break
            centre = 0.65 * (A[i] + A[j])
            distr_X.append(centre + rng.normal(0.0, noise * scale, size=(m, d)))
            distr_lab.append(np.full(m, i))
            remaining -= m
    if distr_X:
        distr_X = np.vstack(distr_X)
        distr_lab = np.concatenate(distr_lab)
    else:
        distr_X = np.empty((0, d))
        distr_lab = np.empty((0,), dtype=int)

    # Interior (redundant) points: convex mixtures of archetypes.
    W = rng.dirichlet(np.full(k, dirichlet_alpha), size=n_interior)  # (n_int, k)
    interior_X = W @ A + rng.normal(0.0, noise * scale, size=(n_interior, d))
    interior_lab = W.argmax(axis=1)

    X = np.vstack([corner_X, distr_X, interior_X])
    lab = np.concatenate([corner_lab, distr_lab, interior_lab])
    is_structural = np.zeros(len(X), dtype=bool)
    is_structural[:len(corner_X)] = True
    is_distractor = np.zeros(len(X), dtype=bool)
    is_distractor[len(corner_X):len(corner_X) + len(distr_X)] = True

    # Shuffle so position carries no information.
    perm = rng.permutation(len(X))
    X, lab = X[perm], lab[perm]
    is_structural, is_distractor = is_structural[perm], is_distractor[perm]

    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(d)])
    df["label"] = lab.astype(int)
    return df, is_structural, is_distractor


# ============================================================
# [3] Model-free structural metrics
# ============================================================

def _l2_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms


def flc_coverage(X_l2_full, retained_idx, query_idx=None, batch=1000):
    """
    Facility Location Coverage: mean over query points of the max cosine similarity
    to the nearest retained point. Model-free. Higher = better coverage.

    query_idx restricts the averaging to a subset of points (an unbiased Monte-Carlo
    estimate of the mean); used to keep cost manageable at large n. None = all points.
    """
    X_q = X_l2_full if query_idx is None else X_l2_full[query_idx]
    X_ret = X_l2_full[retained_idx]
    nq = X_q.shape[0]
    out = np.empty(nq, dtype=np.float64)
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        sims = X_q[s:e] @ X_ret.T
        out[s:e] = sims.max(axis=1)
    return float(out.mean())


def knn_recon_mse(X_full, retained_idx, kth=5, query_idx=None, batch=500):
    """
    kNN reconstruction error (model-free fidelity): reconstruct each query point as
    the mean of its k nearest neighbours within the retained set; return mean squared
    error. Lower = the retained set reconstructs the data manifold better. Does NOT
    use archetypes, so it is independent of DataTypical.

    query_idx restricts the averaging to a subset of points (unbiased estimate of the
    mean); used to keep cost manageable at large n. None = all points.
    """
    X_q = X_full if query_idx is None else X_full[query_idx]
    X_ret = X_full[retained_idx].astype(np.float64)
    sq_ret = np.sum(X_ret ** 2, axis=1)
    nq = X_q.shape[0]
    kk = min(kth, len(retained_idx))
    total = 0.0
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        Xb = X_q[s:e].astype(np.float64)
        sqb = np.sum(Xb ** 2, axis=1, keepdims=True)
        d2 = sqb + sq_ret[np.newaxis, :] - 2.0 * (Xb @ X_ret.T)
        np.clip(d2, 0.0, None, out=d2)
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]      # (B, kk)
        recon = X_ret[nn].mean(axis=1)                         # (B, d)
        total += float(((Xb - recon) ** 2).sum(axis=1).sum())
    return total / nq


# ============================================================
# [4] Significance scorers
# ============================================================
# Convention: higher score = more structurally significant.

def datatypical_formative_ranks(df, label_columns, M=50, seed=42, n_jobs=1):
    """DataTypical formative-Shapley scores (model-free, label-free).

    A single fit yields both formative axes:
      'archetypal'   -- instances that create the archetypal (extent/boundary)
                        structure; expected to align with fidelity metrics.
      'prototypical' -- instances that create the prototypical (coverage/density)
                        structure; expected to align with coverage metrics.

    Uses shapley_top_n=1 so the per-sample explanations (unused here) are skipped;
    the formative rankings are always computed on the full dataset, so this is
    identical to the full-fidelity scores, just faster.
    """
    from datatypical import DataTypical
    dt = DataTypical(
        label_columns=label_columns,
        shapley_mode=True, fast_mode=False, archetypal_method="aa",
        nmf_rank=8, n_prototypes=20,
        shapley_n_permutations=M, shapley_compute_formative=True,
        shapley_top_n=1,
        shapley_early_stopping_patience=99_999,
        random_state=seed, deterministic=True, n_jobs=n_jobs, verbose=False,
    )
    res = dt.fit_transform(df)
    return {
        "archetypal":   res["archetypal_shapley_rank"].to_numpy(np.float64),
        "prototypical": res["prototypical_shapley_rank"].to_numpy(np.float64),
    }


def coreset_scores(X_l2, batch=512):
    """Facility-location COVERAGE significance. Model-free, label-free. This is the
    like-for-like comparator to the formative score.

    significance[i] = coverage lost from the facility-location objective
        F(S) = sum_j max_{p in S, p != j} sim(j, p)
    if point i is removed = sum over points j whose nearest OTHER point is i of
    (sim to nearest - sim to second nearest). High = i uniquely covers a region
    (no good substitute), so it is structurally significant for coverage.

    NOTE: this is the proper marginal-coverage criterion, NOT a column-sum
    centrality/representativeness proxy (which rewards dense interior points and
    is anti-correlated with extent-defining structure).
    """
    n = X_l2.shape[0]
    imp = np.zeros(n, dtype=np.float64)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sims = X_l2[s:e] @ X_l2.T            # (b, n)
        rows = np.arange(e - s)
        sims[rows, s + rows] = -np.inf       # exclude self-coverage
        nn1 = np.argmax(sims, axis=1)
        s1 = sims[rows, nn1]
        sims[rows, nn1] = -np.inf
        nn2 = np.argmax(sims, axis=1)
        s2 = sims[rows, nn2]
        np.add.at(imp, nn1, s1 - s2)
    return imp


# ---- reference methods (model/label dependent -- reported separately) ----

def isolation_forest_scores(X, seed=42):
    """Isolation Forest anomaly score (unsupervised MODEL). Higher = more anomalous
    => treated as more 'significant' here for the high/low sweep."""
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination="auto", random_state=seed, n_jobs=1)
    clf.fit(X)
    return -clf.decision_function(X).astype(np.float64)  # higher = more anomalous


def influence_scores(X, y):
    """Self-influence via a logistic model (Koh & Liang). MODEL + LABEL dependent."""
    from sklearn.linear_model import LogisticRegression
    n, d = X.shape
    Xa = np.column_stack([X.astype(np.float64), np.ones(n)])
    dp1 = d + 1
    le = {v: i for i, v in enumerate(sorted(set(y)))}
    y_enc = np.array([le[v] for v in y])
    classes = sorted(le.values())
    scores = np.zeros(n)
    reg = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             fit_intercept=False)
    reg.fit(Xa, y_enc)
    if len(classes) == 2:
        w = reg.coef_[0]
        p = 1.0 / (1.0 + np.exp(-Xa @ w))
        G = (p - y_enc)[:, None] * Xa
        H = (Xa.T * (p * (1 - p))) @ Xa / n + 1e-4 * np.eye(dp1)
        Hinv = np.linalg.inv(H)
        scores = np.einsum("ij,ij->i", G, (Hinv @ G.T).T)
    else:
        P = reg.predict_proba(Xa)
        for c in range(P.shape[1]):
            ind = (y_enc == c).astype(float)
            Gc = (P[:, c] - ind)[:, None] * Xa
            Hc = (Xa.T * (P[:, c] * (1 - P[:, c]))) @ Xa / n + 1e-4 * np.eye(dp1)
            scores += np.einsum("ij,ij->i", Gc, (np.linalg.inv(Hc) @ Gc.T).T)
    return scores  # higher self-influence = more 'significant'


def knn_shapley_scores(X, y, K=10, n_ref=1500, seed=42):
    """KNN-Shapley data value (Jia et al. 2019). MODEL + LABEL dependent.
    Higher value = more 'significant'. Classification variant."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    le = {v: i for i, v in enumerate(sorted(set(y)))}
    y_enc = np.array([le[v] for v in y])
    classes, counts = np.unique(y_enc, return_counts=True)
    per = max(1, min(n_ref, n) // len(classes))
    ref = np.concatenate([rng.choice(np.where(y_enc == c)[0],
                                     min(per, cnt), replace=False)
                          for c, cnt in zip(classes, counts)])[:min(n_ref, n)]
    Xt = X.astype(np.float32)
    sq = np.sum(Xt ** 2, axis=1)
    j = np.arange(n, dtype=np.float64)
    w = np.minimum(K, j + 1.0) / (K * (j + 1.0))
    phi = np.zeros(n)
    for t in ref:
        d2 = sq[t] + sq - 2.0 * (Xt @ Xt[t])
        order = np.argsort(d2)
        s = (y_enc[order] == y_enc[t]).astype(np.float64)
        diffs = s[:-1] - s[1:]
        suffix = np.cumsum((diffs * w[:-1])[::-1])[::-1]
        ps = np.empty(n)
        ps[-1] = s[-1] / (n * K)
        ps[:-1] = ps[-1] + suffix
        phi[order] += ps
    return phi / len(ref)  # higher = more valuable


# ============================================================
# [5] Ablation runner and effect size
# ============================================================

def _retention_curve(metric_fn, order, fractions, n, higher_is_better, full_value):
    """Remove `order[:n_remove]`; evaluate metric on the retained tail; return
    retention normalised to [0, 1] against the full-dataset value.

    Bounded-ratio normalisation (no arbitrary "floor" reference, so retention can
    never go below 0):
      * FLC coverage (higher is better, non-negative): retention = v / full.
        Removing points can only lower coverage, so v <= full and retention in [0, 1].
      * kNN reconstruction error (lower is better): retention = full / v.
        Removing points can only raise the error, so v >= full and retention in (0, 1].
    """
    rets = []
    for f in fractions:
        n_rem = int(round(f * n))
        v = metric_fn(order[n_rem:])
        if higher_is_better:
            r = (v / full_value) if full_value > 0 else 1.0
        else:
            r = (full_value / v) if v > 0 else 1.0
        rets.append(min(1.0, max(0.0, r)))   # clamp for floating-point safety
    return np.array(rets, dtype=np.float64)


def degradation_auc(fractions, retention):
    """Area under the degradation curve (1 - retention) vs fraction, by trapezoid.
    0 = no degradation; larger = structure lost faster."""
    deg = 1.0 - np.asarray(retention, dtype=np.float64)
    return float(np.trapz(deg, np.asarray(fractions, dtype=np.float64)))


def evaluate_scores(scores, X_full, X_l2, fractions, kth=5, query_cap=3000, seed=12345):
    """
    For one significance score, sweep removal in both directions on both model-free
    metrics and return degradation AUCs.

    Returns dict with, for metric in {flc, knn}:
        auc_high  (remove most-significant first  -> should be LARGE)
        auc_low   (remove least-significant first -> should be SMALL)
        signal = auc_high - auc_low   (discrimination of structural significance)
    plus the raw retention curves for plotting.
    """
    n = X_full.shape[0]
    order_high = np.argsort(-scores)   # most significant removed first
    order_low = np.argsort(scores)     # least significant removed first

    # Fixed query subsample (same set for the full-dataset reference and every
    # fraction, so the ratio normalisation is consistent). Keeps O(n^2) metric
    # cost bounded at large n; an unbiased estimate of the mean metric.
    query_idx = None
    if n > query_cap:
        query_idx = np.random.default_rng(seed).choice(n, query_cap, replace=False)

    flc_full = flc_coverage(X_l2, np.arange(n), query_idx=query_idx)
    knn_full = knn_recon_mse(X_full, np.arange(n), kth=kth, query_idx=query_idx)

    def flc_metric(idx): return flc_coverage(X_l2, idx, query_idx=query_idx)
    def knn_metric(idx): return knn_recon_mse(X_full, idx, kth=kth, query_idx=query_idx)

    out = {}
    for mname, mfn, full, hib in [
        ("flc", flc_metric, flc_full, True),
        ("knn", knn_metric, knn_full, False),
    ]:
        r_high = _retention_curve(mfn, order_high, fractions, n, hib, full)
        r_low = _retention_curve(mfn, order_low, fractions, n, hib, full)
        out[mname] = {
            "retention_high": r_high,
            "retention_low": r_low,
            "auc_high": degradation_auc(fractions, r_high),
            "auc_low": degradation_auc(fractions, r_low),
            "signal": degradation_auc(fractions, r_high) - degradation_auc(fractions, r_low),
        }
    return out


def preprocess(df, feature_cols):
    """MinMax scale features to [0,1] and return (X_scaled, X_l2)."""
    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(df[feature_cols].to_numpy(np.float64))
    return X.astype(np.float64), _l2_rows(X).astype(np.float64)
