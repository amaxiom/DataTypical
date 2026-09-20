"""
The last reachable branches: alternate parallel backends, dtype fallbacks,
degenerate geometry, and the guards that only fire on hand-assembled state.

A handful of statements in datatypical.py remain uncovered by design; they are
defensive guards on conditions the surrounding code makes impossible (a
zero-width feature subset inside the explanation closures, a stereotype target
that is a string other than 'min'/'max' after validation has already run, and a
knee branch whose condition contradicts its enclosing test). They are listed in
the release notes rather than papered over here.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import datatypical as dtmod
from datatypical import ConfigError, DataTypical, DataTypicalError


# ---------------------------------------------------------------------------
# Shapley engine: both parallel backends
# ---------------------------------------------------------------------------
class TestEngineBackends:
    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_verbose_early_stop_on_every_backend(self, n_jobs, capsys):
        X = np.random.default_rng(0).normal(size=(6, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=400, random_state=0, verbose=True, n_jobs=n_jobs,
            early_stopping_patience=1, early_stopping_tolerance=1e9,
        )
        engine.compute_shapley_values(X, lambda a, b, c=None: 1.0, "const")
        assert "Early stop" in capsys.readouterr().out

    def test_verbose_early_stop_on_the_parallel_branch(self, capsys):
        """Parallelism needs n_jobs != 1 AND at least 20 samples."""
        X = np.random.default_rng(0).normal(size=(24, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=400, random_state=0, verbose=True, n_jobs=2,
            early_stopping_patience=1, early_stopping_tolerance=1e9,
        )
        engine.compute_shapley_values(X, lambda a, b, c=None: 1.0, "const")
        assert "Early stop" in capsys.readouterr().out

    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_feature_level_on_every_backend(self, n_jobs):
        X = np.random.default_rng(0).normal(size=(5, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=4, random_state=0, n_jobs=n_jobs
        )
        phi, _ = engine.compute_feature_shapley_values(
            X, lambda a, b, c=None: float(np.mean(a)), "mean"
        )
        assert phi.shape == (5, 3)


# ---------------------------------------------------------------------------
# Convex hull volume with too few distinct points
# ---------------------------------------------------------------------------
class TestConvexHullDegenerate:
    def test_duplicate_points_use_the_range_fallback(self):
        """Five identical rows in 2D cannot span a hull."""
        X = np.tile(np.array([[1.0, 2.0]]), (5, 1))
        value = dtmod.formative_archetypal_convex_hull(X, np.arange(5))
        assert value == pytest.approx(1e-20, abs=1e-15)

    def test_nearly_duplicate_points_use_the_range_fallback(self):
        X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        assert np.isfinite(dtmod.formative_archetypal_convex_hull(X, np.arange(4)))


# ---------------------------------------------------------------------------
# Archetypal dtype and PCHA guards, driven directly
# ---------------------------------------------------------------------------
class TestArchetypalDirect:
    def test_nmf_dtype_falls_back_to_float64(self):
        """float32 input with dtype='float64' must widen, not narrow."""
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, dtype="float64")
        W, H = dt._fit_archetypal_nmf(np.abs(np.ones((6, 4), dtype=np.float32)))
        assert W.dtype == np.float64

    def test_aa_dtype_falls_back_to_float64(self):
        pytest.importorskip("py_pcha")
        rng = np.random.default_rng(0)
        X = np.abs(rng.normal(size=(20, 5))).astype(np.float32)
        dt = DataTypical(archetypal_method="auto", nmf_rank=3, dtype="float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W, _ = dt._fit_archetypal_aa(X)
        assert W.dtype == np.float64

    def test_negative_values_are_shifted_before_pcha(self):
        """PCHA needs a non-negative matrix; negatives are translated first."""
        pytest.importorskip("py_pcha")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 6)) - 5.0
        assert X.min() < 0
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        W, H = dt._fit_archetypal_aa(X)
        assert dt.archetypal_backend_ == "pcha"
        assert np.isfinite(W).all()

    def test_pcha_h_shape_mismatch_is_caught(self, monkeypatch):
        """W is the right shape, H is not."""
        rng = np.random.default_rng(0)
        X = np.abs(rng.normal(size=(20, 6)))

        def wrong_h(Xt, noc, delta=0.0):
            n_samples = Xt.shape[1]
            S = np.ones((noc, n_samples))       # S.T -> (n_samples, noc), correct
            XC = np.ones((2, noc))              # XC.T -> (noc, 2), wrong width
            return XC, S, None, 1.0, 0.5

        monkeypatch.setattr(dtmod, "PCHA", wrong_h)
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        with pytest.raises(ConfigError, match="H shape error"):
            dt._fit_archetypal_aa(X)


# ---------------------------------------------------------------------------
# Preprocessing warnings and transform guards
# ---------------------------------------------------------------------------
class TestPreprocessingGuards:
    def test_constant_columns_are_dropped_with_a_warning(self, df_small):
        df = df_small.copy()
        df["flat"] = 3.0
        # The warning itself is gated behind verbose; the drop is not.
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, verbose=True, random_state=0
        )
        with pytest.warns(UserWarning, match="Dropped constant feature columns"):
            dt.fit(df)
        assert "flat" in dt.dropped_columns_

    def test_feature_weights_length_mismatch_warns(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            feature_weights=np.ones(2),
        )
        with pytest.warns(UserWarning, match="feature_weights length mismatch"):
            dt.fit(df_small)

    def test_non_numeric_values_at_transform_are_rejected(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        corrupt = df_small.copy()
        corrupt["f0"] = corrupt["f0"].astype(object)
        corrupt.iloc[0, corrupt.columns.get_loc("f0")] = "not a number"
        with pytest.raises(DataTypicalError, match="Non-numeric values present"):
            dt.transform(corrupt)

    def test_stereotype_source_table_is_none_when_unconfigured(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        assert dt._get_stereotype_source_table(df_small) is None


# ---------------------------------------------------------------------------
# Text path without scipy
# ---------------------------------------------------------------------------
class TestTextWithoutScipy:
    def test_fit_requires_scipy(self, corpus, monkeypatch):
        monkeypatch.setattr(dtmod, "sp", None)
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        with pytest.raises(ImportError, match="scipy is required"):
            dt.fit_text(corpus)

    def test_transform_requires_scipy(self, corpus, monkeypatch):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit_text(corpus)
        monkeypatch.setattr(dtmod, "sp", None)
        with pytest.raises(ImportError, match="scipy is required"):
            dt.transform_text(corpus)


# ---------------------------------------------------------------------------
# Single-feature geometry, which takes the corner-score fallback
# ---------------------------------------------------------------------------
def test_corner_scores_with_a_single_feature():
    """One column means fewer than two edge-hitting columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"only": rng.normal(size=25)})
    dt = DataTypical(
        archetypal_method="nmf", nmf_rank=1, n_prototypes=4, random_state=0
    )
    dt.disallow_overlap = True
    dt.overlap_alpha = 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt.fit(df)
    assert dt.prototype_indices_ is not None


# ---------------------------------------------------------------------------
# Scoring guards driven directly
# ---------------------------------------------------------------------------
class TestScoringGuards:
    def test_feature_dimension_mismatch_is_reported(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        wrong_width = np.zeros((5, dt.H_.shape[1] + 3))
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            dt._score_with_fitted(
                wrong_width,
                dtmod._l2_normalize_rows_dense(wrong_width + 1.0),
                pd.RangeIndex(5),
                None,
            )

    def test_prototype_features_fallback(self, df_small):
        """Legacy fits have no stored prototype rows, so they are re-indexed."""
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=4, random_state=0
        )
        dt.fit(df_small)
        dt.prototype_features_ = None
        out = dt.transform(df_small)
        assert len(out) == len(df_small)

    def test_large_assignment_uses_the_numpy_path(self):
        """Above a million sample-prototype pairs the matmul beats the kernel."""
        rng = np.random.default_rng(0)
        X_l2 = dtmod._l2_normalize_rows_dense(rng.normal(size=(5200, 8)))
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=200)
        dt.prototype_features_l2_ = X_l2[:200]
        best_cos, labels = dt._assignments_cosine(X_l2, np.arange(200))
        assert len(labels) == 5200
        assert np.isfinite(best_cos).all()


# ---------------------------------------------------------------------------
# Explanation subsampling without a union core set
# ---------------------------------------------------------------------------
def test_explanations_with_subsample_but_no_union_core(df_small):
    dt = DataTypical(
        shapley_mode=True, shapley_n_permutations=4, nmf_rank=3, n_prototypes=4,
        archetypal_method="nmf", shapley_compute_formative=False, random_state=0,
    )
    dt.fit_transform(df_small)
    # A direct call with subsample_indices and no _union_core_samples: every
    # requested sample is then a core sample.
    if hasattr(dt, "_union_core_samples"):
        del dt._union_core_samples
    engine = dtmod.ShapleySignificanceEngine(n_permutations=4, random_state=0)
    X_dense = dt._df_original_fit[dt.feature_columns_].to_numpy(dtype=np.float64)
    X_dense = X_dense[:, dt.keep_mask_]
    dt._fit_shapley_explanations(
        X_dense, dt.train_index_, engine, subsample_indices=np.arange(5)
    )
    assert dt.Phi_archetypal_explanations_.shape[0] == len(df_small)
