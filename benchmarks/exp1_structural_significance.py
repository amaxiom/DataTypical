"""
exp1_structural_significance.py -- DataTypical v0.7.7
=====================================================
Test of the central claim: ARE formative instances structurally significant?

Design (see common.py for the rationale):
  * Model-free, label-free scorers compared like-for-like:
        formative_arch   (DataTypical archetypal formative-Shapley)
        formative_proto  (DataTypical prototypical formative-Shapley)
        coreset          (facility-location coverage)
        random           (null baseline)
  * Reference scorers (MODEL/LABEL dependent -- different objective, reported
    in a separate panel, never mixed into the like-for-like comparison):
        isolation_forest, influence, knn_shapley
  * Model-free metrics:
        FLC  (coverage / density)
        kNN  (reconstruction fidelity / manifold extent)   <- discriminating metric
  * Effect size = degradation AUC; structural-significance signal = AUC(remove
    most-significant-first) - AUC(remove least-significant first), averaged over seeds with std,
    plus PAIRED per-seed differences (formative_arch minus each baseline) with a
    one-sided paired t-test.
  * Removal fractions include a FINE low end (0.2%..5%) so the few ground-truth
    corner points are resolved rather than diluted at large n.
  * Primary dataset is SYNTHETIC with KNOWN structure-defining points AND known
    redundant distractor-outlier clusters, so recovery is validated against ground
    truth (AUROC vs corners) and against distraction (AUROC vs distractors --
    anomaly detectors rank the distractors high, a per-instance structural score
    should not). Real UCI sets are included as a (saturated) reality check.

Outputs
-------
  exp1_results.csv                 long-form: dataset, seed, scorer, metric, auc_high/low, signal
  exp1_recovery.csv                synthetic recovery per scorer/seed:
                                   auroc_structure (vs corners), auroc_distractor (vs distractors)
  exp1_paired_stats.csv            paired per-seed signal differences + one-sided paired t-tests
  exp1_synthetic_signal.pdf/.png   headline: structural-significance signal (kNN), like-for-like vs reference
  exp1_synthetic_curves.pdf/.png   degradation curves (both metrics), model-free scorers
  exp1_recovery.pdf/.png           structure vs distractor AUROC per scorer
  exp1_realdata_signal.pdf/.png    signal across real datasets (shows saturation)

Usage
-----
  python exp1_structural_significance.py                 # synthetic only (fast)
  python exp1_structural_significance.py --real --data-dir <dir> --cap 4000
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as fc
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
fc.apply_rcparams()

# Removal grid with a fine low end: the ground-truth corners are a small fixed
# count (k * n_corner_per), so at large n all the discriminating information
# sits below a few percent removed. A grid starting at 5% would merge the
# corners into the first bulk step and mechanically dilute the signal.
FRACTIONS = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
                      0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
SEEDS = [0, 1, 2, 3, 4]

# Output mode (set by main() or by a notebook before calling run()):
#   SAVE=True  -> write .pdf/.png (and CSVs) to the current directory
#   SHOW=True  -> display figures inline (e.g. in a notebook)
# The notebook uses SAVE=False, SHOW=True; the scripts default to SAVE=True, SHOW=False.
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

LIKE_FOR_LIKE = ["formative_arch", "formative_proto", "coreset", "random"]
REFERENCE = ["isolation_forest", "influence", "knn_shapley"]
PRETTY = {
    "formative_arch": "Formative (archetypal)",
    "formative_proto": "Formative (prototypical)",
    "coreset": "Coreset (facility location)",
    "random": "Random",
    "isolation_forest": "Isolation Forest",
    "influence": "Influence functions",
    "knn_shapley": "KNN-Shapley",
}


# ============================================================
# scoring
# ============================================================

def all_scores(df, feature_cols, X, X_l2, label_col, seed, want_reference=True):
    """Return {scorer_name: scores}. Higher = more structurally significant."""
    form = fc.datatypical_formative_ranks(df, label_columns=[label_col], M=50, seed=42)
    scores = {
        "formative_arch":  form["archetypal"],
        "formative_proto": form["prototypical"],
        "coreset":         fc.coreset_scores(X_l2),
        "random":          np.random.default_rng(seed).normal(size=len(df)),
    }
    if want_reference:
        y = df[label_col].to_numpy()
        scores["isolation_forest"] = fc.isolation_forest_scores(X, seed=42)
        scores["influence"]        = fc.influence_scores(X, y)
        scores["knn_shapley"]      = fc.knn_shapley_scores(X, y, seed=42)
    return scores


# ============================================================
# run one dataset across seeds
# ============================================================

def run_dataset(name, make_df, has_ground_truth, seeds, want_reference=True):
    rows, recovery_rows = [], []
    curve_store = {}  # (scorer, metric) -> list of (retention_high, retention_low) per seed
    for seed in seeds:
        df, is_struct, is_distr, feature_cols, label_col = make_df(seed)
        X, X_l2 = fc.preprocess(df, feature_cols)
        scores = all_scores(df, feature_cols, X, X_l2, label_col, seed, want_reference)

        for scorer, sc in scores.items():
            if has_ground_truth and scorer != "random":
                recovery_rows.append({
                    "dataset": name, "seed": seed, "scorer": scorer,
                    "auroc_structure": float(roc_auc_score(is_struct, sc)),
                    "auroc_distractor": (float(roc_auc_score(is_distr, sc))
                                         if is_distr.any() else np.nan),
                })
            ev = fc.evaluate_scores(sc, X, X_l2, FRACTIONS)
            for metric in ("flc", "knn"):
                rows.append({
                    "dataset": name, "seed": seed, "scorer": scorer, "metric": metric,
                    "auc_high": ev[metric]["auc_high"],
                    "auc_low": ev[metric]["auc_low"],
                    "signal": ev[metric]["signal"],
                })
                curve_store.setdefault((scorer, metric), []).append(
                    (ev[metric]["retention_high"], ev[metric]["retention_low"]))
        print(f"  [{name}] seed {seed} done")
    return pd.DataFrame(rows), pd.DataFrame(recovery_rows), curve_store


# ============================================================
# figures
# ============================================================

def fig_signal(df_res, dataset, metric, scorers, fname, title):
    # horizontal bars: long scorer names sit on the y-axis (no rotation needed)
    sub = df_res[(df_res.dataset == dataset) & (df_res.metric == metric)]
    g = sub.groupby("scorer")["signal"].agg(["mean", "std"]).reindex(scorers)
    colors = fc.viridis_colors(len(scorers))
    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    y = np.arange(len(scorers))
    ax.barh(y, g["mean"], xerr=g["std"], color=colors, edgecolor="white",
            linewidth=0.6, capsize=4, zorder=3)
    ax.axvline(0.0, color="0.4", linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[s] for s in scorers])
    ax.invert_yaxis()  # first scorer at top
    fc.style_axes(
        ax,
        xlabel=("Structural-significance signal\n"
                "= AUC(remove most-significant first) - AUC(remove least-significant first)\n"
                "positive: removing the most-significant first degrades structure more"),
        title=title)
    ax.xaxis.label.set_fontsize(9)
    _emit(fig, fname)


def fig_curves(curve_store, dataset, scorers, fname):
    metrics = [("knn", "kNN reconstruction (fidelity)"), ("flc", "FLC coverage (density)")]
    colors = fc.viridis_colors(len(scorers))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, (metric, mtitle) in zip(axes, metrics):
        for sc, col in zip(scorers, colors):
            hi = np.array([c[0] for c in curve_store[(sc, metric)]])
            lo = np.array([c[1] for c in curve_store[(sc, metric)]])
            mh, sh = hi.mean(0), hi.std(0)
            ml = lo.mean(0)
            ax.plot(FRACTIONS * 100, mh, color=col, lw=2.0, marker="o", ms=5,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=f"{PRETTY[sc]}\n(remove most-significant first)")
            ax.fill_between(FRACTIONS * 100, mh - sh, mh + sh, color=col, alpha=0.15)
            ax.plot(FRACTIONS * 100, ml, color=col, lw=1.5, ls="--",
                    label=f"{PRETTY[sc]}\n(remove least-significant first)")
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50])
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:g}"))
        ax.minorticks_off()
        fc.style_axes(ax, xlabel="Fraction removed (%), log scale",
                      ylabel="Structural retention", title=mtitle)
    fc.legend_outside(axes[1], ncol=1)
    fig.suptitle(f"{dataset}: structural degradation under pruning (mean +/- std over seeds)",
                 fontsize=11)
    _emit(fig, fname)


def fig_recovery(df_rec, fname):
    gs = df_rec.groupby("scorer")["auroc_structure"].agg(["mean", "std"])
    gd = df_rec.groupby("scorer")["auroc_distractor"].agg(["mean", "std"])
    order = [s for s in ["formative_arch", "formative_proto", "coreset",
                         "isolation_forest", "influence", "knn_shapley"] if s in gs.index]
    gs, gd = gs.reindex(order), gd.reindex(order)
    c_struct, c_distr = fc.viridis_colors(2, lo=0.15, hi=0.75)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), layout="constrained")
    y = np.arange(len(order))
    h = 0.38
    ax.barh(y - h / 2, gs["mean"], height=h, xerr=gs["std"], color=c_struct,
            edgecolor="white", linewidth=0.6, capsize=3, zorder=3,
            label="structure-defining corners\n(load-bearing: want HIGH)")
    ax.barh(y + h / 2, gd["mean"], height=h, xerr=gd["std"], color=c_distr,
            edgecolor="white", linewidth=0.6, capsize=3, zorder=3,
            label="redundant distractor outliers\n(anomalous, expendable: want LOW)")
    ax.axvline(0.5, color="0.4", lw=0.8, ls="--", zorder=2, label="chance (0.5)")
    ax.set_yticks(y); ax.set_yticklabels([PRETTY[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    fc.style_axes(ax, xlabel="AUROC vs ground-truth instance class",
                  title="Structure vs distraction: does a high score mean load-bearing,\n"
                        "or merely anomalous? (synthetic, mean +/- std over seeds)")
    fc.legend_outside(ax)
    _emit(fig, fname)


# ============================================================
# main
# ============================================================

def run(n=1000, seeds=(0, 1, 2, 3, 4), n_features=8, real=False,
        data_dir=".", cap=4000):
    """Run the structural-significance experiment. Honours module-level SAVE/SHOW.
    Set SAVE/SHOW before calling (CLI sets them from flags; notebooks set directly)."""
    seeds = list(seeds)
    all_res, all_rec = [], []

    # ---- synthetic (primary, ground truth) ----
    print(f"Synthetic (controlled, known structure)  n={n}, seeds={seeds}:")
    def make_syn(seed):
        df, is_struct, is_distr = fc.make_synthetic_archetypal(
            n=n, d=n_features, k=6, n_corner_per=3, seed=seed,
            n_distractor_clusters=2, n_distractor_per=25)
        return df, is_struct, is_distr, [f"f{j}" for j in range(n_features)], "label"
    res, rec, curves = run_dataset("Synthetic", make_syn, has_ground_truth=True, seeds=seeds)
    all_res.append(res); all_rec.append(rec)

    fig_recovery(rec, "exp1_recovery")
    fig_signal(res, "Synthetic", "knn", LIKE_FOR_LIKE, "exp1_synthetic_signal",
               "Synthetic: structural-significance signal (kNN fidelity)\nmodel-free, like-for-like comparison")
    fig_signal(res, "Synthetic", "knn", REFERENCE, "exp1_reference_signal",
               "Reference methods (model/label dependent) - different objective")
    fig_curves(curves, "Synthetic", LIKE_FOR_LIKE, "exp1_synthetic_curves")

    # ---- real datasets (optional reality check) ----
    if real:
        from sklearn.model_selection import train_test_split
        def loader_factory(kind):
            def make(seed):
                df, label = _load_real(kind, data_dir)
                if cap and len(df) > cap:
                    strat = df[label] if df[label].dtype.kind != "f" else None
                    df, _ = train_test_split(df, train_size=cap, random_state=seed,
                                             stratify=strat)
                    df = df.reset_index(drop=True)
                feats = [c for c in df.columns if c != label]
                zeros = np.zeros(len(df), bool)
                return df, zeros, zeros, feats, label
            return make
        for kind, dname in [("magic", "MAGIC"), ("casp", "CASP"), ("shuttle", "Shuttle")]:
            print(f"{dname} (real, capped {cap}):")
            res2, _, _ = run_dataset(dname, loader_factory(kind),
                                     has_ground_truth=False, seeds=seeds, want_reference=True)
            all_res.append(res2)
        df_all = pd.concat(all_res, ignore_index=True)
        rdata = df_all[(df_all.dataset != "Synthetic") & (df_all.metric == "knn")]
        if len(rdata):
            piv = rdata.groupby(["dataset", "scorer"])["signal"].mean().unstack().reindex(columns=LIKE_FOR_LIKE)
            colors = fc.viridis_colors(len(LIKE_FOR_LIKE))
            fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
            xpos = np.arange(len(piv.index)); w = 0.8 / len(LIKE_FOR_LIKE)
            for i, sc in enumerate(LIKE_FOR_LIKE):
                ax.bar(xpos + i * w, piv[sc].values, w, color=colors[i],
                       edgecolor="white", linewidth=0.5, label=PRETTY[sc], zorder=3)
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set_xticks(xpos + 0.4 - w / 2); ax.set_xticklabels(piv.index)
            fc.style_axes(ax, ylabel="Signal (kNN)", xlabel="Dataset",
                          title="Real datasets: structural-significance signal (note saturation)")
            fc.legend_outside(ax)
            _emit(fig, "exp1_realdata_signal")

    df_all = pd.concat(all_res, ignore_index=True)
    df_rec_all = pd.concat(all_rec, ignore_index=True)
    df_paired = paired_signal_stats(df_all)
    if SAVE:
        df_all.to_csv("exp1_results.csv", index=False)
        df_rec_all.to_csv("exp1_recovery.csv", index=False)
        df_paired.to_csv("exp1_paired_stats.csv", index=False)
        print("\nSaved exp1_results.csv, exp1_recovery.csv, exp1_paired_stats.csv")

    # console summary
    print("\n=== Synthetic, kNN fidelity: mean signal +/- std over seeds ===")
    syn = df_all[(df_all.dataset == "Synthetic") & (df_all.metric == "knn")]
    g = syn.groupby("scorer")["signal"].agg(["mean", "std"])
    for s in LIKE_FOR_LIKE + REFERENCE:
        if s in g.index:
            tag = "like" if s in LIKE_FOR_LIKE else "ref "
            print(f"  [{tag}] {PRETTY[s]:26s} {g.loc[s,'mean']:+.4f} +/- {g.loc[s,'std']:.4f}")

    print("\n=== Paired per-seed differences in kNN signal (one-sided paired t-test) ===")
    for _, r in df_paired.iterrows():
        print(f"  {r['comparison']:38s} mean diff {r['mean_diff']:+.4f} "
              f"+/- {r['std_diff']:.4f}  (n={int(r['n_seeds'])}, t={r['t_stat']:.2f}, "
              f"p={r['p_one_sided']:.4f})")

    print("\n=== Structure vs distraction (AUROC, mean over seeds) ===")
    grec = df_rec_all.groupby("scorer")[["auroc_structure", "auroc_distractor"]].mean()
    for s in LIKE_FOR_LIKE + REFERENCE:
        if s in grec.index:
            print(f"  {PRETTY[s]:26s} structure {grec.loc[s,'auroc_structure']:.3f}   "
                  f"distractor {grec.loc[s,'auroc_distractor']:.3f}")


def paired_signal_stats(df_all, dataset="Synthetic", metric="knn",
                        target="formative_arch",
                        baselines=("random", "coreset", "formative_proto")):
    """Paired per-seed differences in structural-significance signal.

    The seed is the unit of replication: the same data draw feeds every scorer,
    so a paired comparison (target minus baseline within seed) removes the
    shared data variability and is the appropriate test. One-sided alternative:
    target's signal exceeds the baseline's.
    """
    from scipy import stats
    sub = df_all[(df_all.dataset == dataset) & (df_all.metric == metric)]
    piv = sub.pivot(index="seed", columns="scorer", values="signal")
    rows = []
    for base in baselines:
        if target not in piv.columns or base not in piv.columns:
            continue
        diff = (piv[target] - piv[base]).dropna()
        t = stats.ttest_rel(piv[target], piv[base], alternative="greater")
        rows.append({
            "comparison": f"{target} - {base}",
            "n_seeds": len(diff),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "t_stat": float(t.statistic),
            "p_one_sided": float(t.pvalue),
        })
    return pd.DataFrame(rows)


def _load_real(kind, data_dir):
    if kind == "magic":
        cols = ["fLength","fWidth","fSize","fConc","fConc1",
                "fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
        df = pd.read_csv(os.path.join(data_dir, "magic04.data"), header=None, names=cols)
        return df, "class"
    if kind == "casp":
        df = pd.read_csv(os.path.join(data_dir, "CASP.csv"))
        return df, "RMSD"
    if kind == "shuttle":
        cols = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","class"]
        trn_path = os.path.join(data_dir, "shuttle_trn")
        if not os.path.isfile(trn_path):
            import pathlib, unlzw3
            with open(trn_path, "wb") as fh:
                fh.write(unlzw3.unlzw(pathlib.Path(os.path.join(data_dir, "shuttle.trn.Z"))))
        trn = pd.read_csv(trn_path, header=None, names=cols, sep=r"\s+")
        tst = pd.read_csv(os.path.join(data_dir, "shuttle.tst"), header=None, names=cols, sep=r"\s+")
        return pd.concat([trn, tst], ignore_index=True), "class"
    raise ValueError(kind)


def main():
    global SAVE, SHOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also run MAGIC/CASP/Shuttle")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--cap", type=int, default=4000, help="subsample cap for real datasets")
    ap.add_argument("--n", type=int, default=1000, help="synthetic dataset size")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds")
    ap.add_argument("--show", action="store_true", help="display figures instead of/with saving")
    ap.add_argument("--no-save", action="store_true", help="do not write figure/CSV files")
    args = ap.parse_args()
    SAVE = not args.no_save
    SHOW = args.show
    run(n=args.n, seeds=tuple(range(args.seeds)), real=args.real,
        data_dir=args.data_dir, cap=args.cap)


if __name__ == "__main__":
    main()
