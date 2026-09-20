# Structural-significance benchmarks (DataTypical v0.7.7)

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

- **Like-for-like comparator: `coreset`** (facility-location coverage) --
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
| `exp2_scaling.py` | formative-step scaling and the v0.7.7 streaming win |
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
(signal > 0). **An inverted pair -- least-significant-first degrading more -- is
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
# Experiment 1 -- synthetic (fast, has ground truth). --show to display, --no-save to skip files.
python exp1_structural_significance.py
python exp1_structural_significance.py --real --data-dir ../DRAFTS/benchmakrs_for_repo --cap 4000
python exp1_structural_significance.py --n 10000 --seeds 3      # publication-scale

# Experiment 2 -- scaling / v0.7.7 win (a few minutes; generic path is slow by design)
python exp2_scaling.py                 # full grids
python exp2_scaling.py --quick         # smaller grids, fast
python exp2_scaling.py --replot        # redraw figure from exp2_results.csv only
```

Scripts save outputs (CSVs + `.pdf/.png`) to the current directory by default;
pass `--show`/`--no-save` to change the output mode.

## Headline result

On the synthetic data with known structure, **formative (archetypal) instances
recover the ground-truth structure-defining points** (AUROC well above chance)
**while down-ranking the redundant distractor outliers** (AUROC below chance),
and are the **only model-free score with a positive structural-significance signal
on the fidelity metric** -- distinct from random and from coverage-based methods
(prototypical formative, coreset), which target density instead. Isolation Forest
also finds the corners, but it flags the distractors just as readily: it detects
*anomalies*, not *load-bearing instances*. The two roles are complementary, and
the metric must match the question.
