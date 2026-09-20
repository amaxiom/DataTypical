"""
exp2_scaling.py -- DataTypical v0.7.7
=====================================
Focused scaling of the formative-Shapley computation, and the v0.7.7 win.

v0.7.7 replaced the per-prefix re-evaluation of each formative value function with
an exact streaming pass. This script measures the formative step directly (via the
Shapley engine on cached archetype geometry), comparing the streaming path against
the original generic path, as a function of:

  (a) dataset size n        (streaming vs generic)   -- the headline speedup
  (b) dimensionality d      (streaming)
  (c) permutations M        (streaming vs generic)

The generic path is O(M*n^2) (archetypal) / O(M*n^3) (prototypical); streaming is
O(M*n) / O(M*n^2). Both produce numerically identical Shapley values.

Output
------
  exp2_results.csv
  exp2_scaling.pdf/.png   1 x 3 panels (n, d, M), streaming vs generic

Styling: viridis, legend outside, standard (unrotated) tick labels, autoscaled
(log scales where values span orders of magnitude).
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as fc

import datatypical as dtmod
from datatypical import (ShapleySignificanceEngine,
                         formative_archetypal_pcha_cached)

warnings.filterwarnings("ignore")
fc.apply_rcparams()

D_DEFAULT, K, M_DEFAULT, SEED = 8, 8, 50, 42
N_GRID = [500, 1000, 2000, 4000]
D_GRID = [2, 4, 6, 8, 10, 16]
M_GRID = [10, 25, 50, 100, 200, 400]
# Smaller grids for a quick (e.g. notebook) run -- the generic path is slow.
N_GRID_QUICK = [500, 1000, 2000]
M_GRID_QUICK = [10, 50, 200]
_ORIG_REGISTRY = dict(dtmod._PREFIX_VALUE_FUNCS)

# Output mode (set by main() or a notebook before calling run()).
SAVE = True
SHOW = False


def _emit(fig, fname):
    if SAVE:
        fig.savefig(fname + ".pdf"); fig.savefig(fname + ".png")
        print(f"  figure: {fname}.pdf")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def _engine(M):
    return ShapleySignificanceEngine(n_permutations=M, random_state=SEED, n_jobs=1,
                                     early_stopping_patience=99_999,
                                     early_stopping_tolerance=0.01, verbose=False)


def _time_formative(X, archetypes, M, streaming):
    if streaming:
        dtmod._PREFIX_VALUE_FUNCS.clear(); dtmod._PREFIX_VALUE_FUNCS.update(_ORIG_REGISTRY)
    else:
        dtmod._PREFIX_VALUE_FUNCS.clear()
    try:
        ctx = {"archetypes": archetypes}
        t0 = time.perf_counter()
        _engine(M).compute_shapley_values(X, formative_archetypal_pcha_cached,
                                          "archetypal_formative", ctx)
        dt = time.perf_counter() - t0
    finally:
        dtmod._PREFIX_VALUE_FUNCS.clear(); dtmod._PREFIX_VALUE_FUNCS.update(_ORIG_REGISTRY)
    return dt


def _data(n, d, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    A = rng.normal(size=(K, d)).astype(np.float64)
    return X, A


def run(quick=False):
    """Time the formative step (streaming vs generic) across n, d, M. Honours
    module-level SAVE/SHOW. quick=True uses smaller grids for a fast run."""
    n_grid = N_GRID_QUICK if quick else N_GRID
    m_grid = M_GRID_QUICK if quick else M_GRID
    # warm up numba so JIT compile time is excluded
    Xw, Aw = _data(60, D_DEFAULT)
    _time_formative(Xw, Aw, 5, True); _time_formative(Xw, Aw, 5, False)

    rows = []
    print("vary n (d=%d, M=%d): streaming vs generic" % (D_DEFAULT, M_DEFAULT))
    for n in n_grid:
        X, A = _data(n, D_DEFAULT)
        ts = _time_formative(X, A, M_DEFAULT, True)
        tg = _time_formative(X, A, M_DEFAULT, False)
        rows.append({"axis": "n", "value": n, "streaming_s": ts, "generic_s": tg})
        print(f"  n={n:5d}  streaming={ts:7.3f}s  generic={tg:8.3f}s  ({tg/max(ts,1e-9):.0f}x)")

    print("vary d (n=1000, M=%d): streaming" % M_DEFAULT)
    for d in D_GRID:
        X, A = _data(1000, d)
        ts = _time_formative(X, A, M_DEFAULT, True)
        rows.append({"axis": "d", "value": d, "streaming_s": ts, "generic_s": np.nan})
        print(f"  d={d:3d}  streaming={ts:7.3f}s")

    print("vary M (n=1000, d=%d): streaming vs generic" % D_DEFAULT)
    for M in m_grid:
        X, A = _data(1000, D_DEFAULT)
        ts = _time_formative(X, A, M, True)
        tg = _time_formative(X, A, M, False)
        rows.append({"axis": "M", "value": M, "streaming_s": ts, "generic_s": tg})
        print(f"  M={M:4d}  streaming={ts:7.3f}s  generic={tg:8.3f}s")

    df = pd.DataFrame(rows)
    if SAVE:
        df.to_csv("exp2_results.csv", index=False)
        print("\nSaved exp2_results.csv")
    make_figure(df)


def make_figure(df):
    # ---- figure: 1 x 3 ; log-y where the two paths span orders of magnitude ----
    c_stream, c_generic = fc.viridis_colors(2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    panels = [
        # axis, xlabel, title, show_generic, log_y, log_x
        ("n", "Dataset size n", "(a) Runtime vs n  (d=%d, M=%d)" % (D_DEFAULT, M_DEFAULT), True, True, True),
        ("d", "Dimensionality d", "(b) Runtime vs d  (n=1000, M=%d)" % M_DEFAULT, False, False, False),
        ("M", "Shapley permutations M", "(c) Runtime vs M  (n=1000, d=%d)" % D_DEFAULT, True, True, True),
    ]
    import matplotlib.ticker as mticker
    for ax, (axis, xlabel, title, show_generic, log_y, log_x) in zip(axes, panels):
        sub = df[df.axis == axis].sort_values("value")
        ax.plot(sub["value"], sub["streaming_s"], color=c_stream, lw=2.0, marker="o",
                ms=6, markeredgecolor="white", markeredgewidth=0.6, label="streaming (v0.7.7)")
        if show_generic:
            ax.plot(sub["value"], sub["generic_s"], color=c_generic, lw=2.0, ls="--",
                    marker="s", ms=6, markeredgecolor="white", markeredgewidth=0.6,
                    label="generic (pre-v0.7.7)")
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(sub["value"].tolist())
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.minorticks_off()
        ylab = "Formative-step wall time (s)" + (", log scale" if log_y else "")
        xlab = xlabel + (", log scale" if log_x else "")
        fc.style_axes(ax, xlabel=xlab, ylabel=ylab, title=title)
    fc.legend_outside(axes[2])
    fig.suptitle("DataTypical v0.7.7 formative-Shapley scaling (archetypal value function, single-thread)\n"
                 "streaming and generic paths produce numerically identical Shapley values",
                 fontsize=10)
    _emit(fig, "exp2_scaling")


def main():
    global SAVE, SHOW
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grids for a fast run")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw from exp2_results.csv only")
    args = ap.parse_args()
    SAVE = not args.no_save
    SHOW = args.show
    if args.replot:
        make_figure(pd.read_csv("exp2_results.csv"))
    else:
        run(quick=args.quick)


if __name__ == "__main__":
    main()
