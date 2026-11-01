# Structural-significance benchmarks (DataTypical v0.8.2)

> **The figures stand for v0.8.3.** They were last run with v0.8.2. Version
> 0.8.3 changes only the archetypal and prototypical feature-level
> explanations, which these benchmarks do not read, and leaves every rank and
> formative value as it was, so a re-run would reproduce them.

Focused, corrected re-do of the Appendix A/C structural experiments. These
scripts answer one question cleanly:

> **Are formative instances structurally significant?**

## Why this exists (what was wrong before)

The original experiments were ambiguous because:

1. **Mismatched comparators.** DataTypical's formative score is **model-free and
   label-free**. It was benchmarked against methods that require a model and/or
   labels (Isolation Forest, influence functions, KNN-Shapley) on geometric
   metrics. Those answer a different question and don't belong on the same axis.
2. **Circular / saturated metrics.** One metric (ARE) was built from
   DataTypical's own archetypes; and the large UCI sets are so redundant that
   removing half the data barely moves any metric.
3. **Eyeballed curves.** No effect size or error bars.

## What these scripts do instead

- **Like-for-like comparator: `coreset`** (facility-location coverage),
  the only other model-free, geometric, label-free significance score.
  Isolation Forest / influence / KNN-Shapley are kept only as clearly-labelled
  **reference** methods (different objective).
- **Model-free metrics:** `FLC` (coverage/density) and `kNN reconstruction`
  (fidelity/extent). The archetype-based ARE is dropped.
- **Controlled synthetic dataset with known structure-defining points**, so
  formative recovery is validated against ground truth (AUROC) in a regime where
  the effect is detectable. Real UCI sets included as a (saturated) reality check.
- **Redundant distractor-outlier clusters** in the synthetic data: tight clusters
  placed off the archetype simplex, so every member is a textbook density anomaly,
  yet every member is individually expendable (its cluster mates pin down the same
  geometry). This separates "load-bearing" from "merely anomalous": an anomaly
  detector ranks the distractors high; a per-instance structural-significance
  score should not. Recovery is therefore reported as two AUROCs per scorer:
  vs the corners (want high) and vs the distractors (want low).
- **Effect size = degradation AUC**; structural-significance **signal** =
  AUC(remove most-significant first) - AUC(remove least-significant first), with mean +/- std
  over seeds, plus **paired per-seed differences** (formative minus each baseline)
  with a one-sided paired t-test (the seed is the unit of replication: every scorer
  sees the same data draw, so pairing removes the shared data variability).
- **Fine low end on the removal grid (0.2%..5%)**: the ground-truth corners are a
  small fixed count, so at large n all the discriminating information sits below a
  few percent removed; a grid starting at 5% would merge the corners into the first
  bulk step and mechanically dilute the signal. Degradation curves use a log x-axis.

## Files

| File | Purpose |
|------|---------|
| `common.py` | shared styling, synthetic-data generator (corners + distractor clusters), metrics, scorers, ablation runner |
| `exp1_structural_significance.py` | the structural-significance test + figures + paired stats |
| `exp2_scaling.py` | scaling of the sampled formative step, streaming against generic |
| `Benchmarks.ipynb` | runs both experiments and **shows figures inline (saves nothing)**, fast (n=1000) |

Experiment 1 writes `exp1_results.csv` (signals), `exp1_recovery.csv` (structure and
distractor AUROCs per scorer/seed) and `exp1_paired_stats.csv` (paired per-seed
differences with one-sided paired t-tests).

## Metrics (model-free, bounded)

Both metrics are normalised as a bounded ratio to the full-dataset value, so
retention always lies in [0, 1] (it can never go negative):

- **FLC coverage** = mean max cosine similarity of each point to the retained set;
  retention = `flc(retained) / flc(full)`.
- **kNN reconstruction** = mean squared error reconstructing each point from its k
  nearest retained neighbours; retention = `error(full) / error(retained)`.

On redundant data FLC coverage is largely **saturated** (insensitive to removal),
so kNN fidelity is the discriminating metric.

## How to read the curves (including inversions)

Each score is tested in **both removal directions**: remove its *most-significant*
instances first (solid) and its *least-significant* first (dashed). For a score
that truly captures structural significance, the solid curve degrades faster
(signal > 0). **An inverted pair, least-significant-first degrading more, is
not an artefact**: it means that score's ranking is *anti-aligned* with structural
significance, i.e. the structure-defining instances sit at the *bottom* of its
ranking. The ground-truth recovery figure confirms this one-to-one: scorers with
AUROC > 0.5 (formative-archetypal, Isolation Forest) degrade more under
most-significant-first removal; scorers with AUROC < 0.5 (influence functions at
~0.0, KNN-Shapley, prototypical/coreset) invert, because their "least significant"
end contains the geometric extremes the manifold depends on. The random baseline
shows no direction bias (signal ~ 0), verifying the pipeline itself is symmetric.

The recovery figure reports two AUROCs per scorer: **vs the structure-defining
corners (want high)** and **vs the redundant distractor outliers (want low)**.
Anomaly detection scores high on both (it cannot tell load-bearing from merely
anomalous); a per-instance structural-significance score separates them, because
Shapley credit for a location shared by many redundant points is split among
them, so no individual distractor is significant.

## Running (scripts, CLI)

```bash
# Experiment 1: synthetic (fast, has ground truth). --show to display, --no-save to skip files.
python exp1_structural_significance.py
python exp1_structural_significance.py --real --data-dir ../examples --cap 4000
python exp1_structural_significance.py --n 10000 --seeds 3      # publication-scale

# Experiment 2: scaling (a few minutes; the generic path is slow by design)
python exp2_scaling.py                 # full grids
python exp2_scaling.py --quick         # smaller grids, fast
python exp2_scaling.py --replot        # redraw figure from exp2_results.csv only
```

Scripts save outputs (CSVs + `.pdf/.png`) to the current directory by default;
pass `--show`/`--no-save` to change the output mode. The real datasets (MAGIC,
CASP, Shuttle) are the ones the example notebooks use, in `../examples`.

## Headline result

On the synthetic data with known structure, **formative (archetypal) instances
recover the ground-truth structure-defining points while down-ranking the
redundant distractor outliers**. Under the exact computation (the default since
v0.8.0), at n=1000 over 5 seeds, re-run with v0.8.2 on 2026-10-22 and
identical to the v0.8.0 figures:

| scorer | structure AUROC (want high) | distractor AUROC (want low) |
| --- | --- | --- |
| Formative (archetypal) | **0.923** | **0.039** |
| Isolation Forest | 0.940 | 0.861 |
| Coreset (facility location) | 0.477 | 0.415 |
| Formative (prototypical) | 0.083 | 0.282 |
| KNN-Shapley | 0.208 | 0.457 |
| Influence functions | 0.005 | 0.626 |

Isolation Forest matches it on the corners and fails completely on the
distractors. That is the whole distinction: it detects *anomalies*, not
*load-bearing instances*.

> **The kNN fidelity signal metric does not agree, and the disagreement is
> informative.** Under the exact computation the archetypal formative signal is
> -0.021, while Isolation Forest scores +0.118, by far the largest. Isolation
> Forest is also the scorer that ranks distractors highest. That is not a
> coincidence: the distractors are tight redundant clusters, so removing one
> wholesale leaves its members with no near neighbours and kNN reconstruction
> error explodes. The signal metric therefore rewards ranking the distractors
> *highly*, which is the opposite of what it is supposed to measure whenever
> redundant clusters are present. The recovery AUROCs above have actual ground
> truth; the signal metric does not. Read them in that order.
>
> Through v0.7.7 this metric reported a positive signal for the formative score.
> That was computed from the Monte Carlo estimator, whose ordering at 100
> permutations is sampling noise: five seeds gave five different top formative
> prototypes on Wine. The sign of that number was not a finding.

## Scaling (experiment 2)

Experiment 2 times the **sampled** formative step, which since v0.8.0 runs only
when `formative_method='monte_carlo'` is asked for (the default is the exact
closed form, which samples nothing). It compares the streaming pass introduced
in v0.7.7 against the generic per-prefix re-evaluation it replaced, on the
archetypal value function, single-threaded, d = 8, 8 archetypes. The two give
numerically identical values.

Re-run with v0.8.2 on 2026-10-22, on an Intel Core Ultra 9 285K (24 threads,
64 GB) under Windows 11, Python 3.13.9, NumPy 2.3.5 and Numba 0.62.1:

| n (M = 50) | streaming | generic | ratio |
| --- | --- | --- | --- |
| 500 | 0.0016 s | 0.87 s | 533x |
| 1,000 | 0.0024 s | 2.84 s | 1,185x |
| 2,000 | 0.0047 s | 9.90 s | 2,124x |
| 4,000 | 0.0092 s | 120.2 s | 13,100x |

| M (n = 1,000) | streaming | generic |
| --- | --- | --- |
| 10 | 0.0013 s | 0.56 s |
| 50 | 0.0025 s | 3.02 s |
| 100 | 0.0043 s | 6.00 s |
| 400 | 0.0155 s | 24.29 s |

The streaming time grows linearly in n and M; across d = 2 to 16 it stays
between 0.002 and 0.004 seconds at n = 1,000. The generic path grows faster
than quadratically in n over this range. Timings are wall-clock and
machine-dependent; the ratios, and the linear growth, are the result.

## Style

All figures use **viridis**, with two-category comparisons in viridis purple
(#440154) and green (#32b67a); legends **outside** the axes and without a
frame; titles on one line; nothing bold; one font size per text role;
**standard (unrotated) tick labels** (long categorical labels go on the y-axis
via horizontal bars; rotation is used only if labels would otherwise overlap);
and **autoscaled** axes (log scales where values span orders of magnitude), via
the helpers in `common.py` (`apply_rcparams`, `viridis_colors`, `style_axes`,
`legend_outside`).
