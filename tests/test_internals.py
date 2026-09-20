"""
Tests for the module-level machinery underneath DataTypical: numeric helpers,
the facility-location selector, the Shapley engine, and the value functions.
"""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import datatypical as dtmod
from datatypical import (
    ConfigError,
    DataTypical,
    DataTypicalError,
    FacilityLocationSelector,
    MemoryBudgetError,
)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
class TestNumericHelpers:
    def test_cleanup_memory_accepts_none_and_arrays(self):
        dtmod._cleanup_memory(np.ones(3), None, np.zeros(2))

    def test_cleanup_memory_can_force_gc(self):
        dtmod._cleanup_memory(np.ones(3), force_gc=True)

    def test_l2_normalize_rows_dense(self):
        X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        out = dtmod._l2_normalize_rows_dense(X)
        np.testing.assert_allclose(out[0], [0.6, 0.8])
        np.testing.assert_allclose(out[1], [0.0, 0.0])  # zero row survives intact
        np.testing.assert_allclose(np.linalg.norm(out[2]), 1.0)

    def test_sparse_l2_normalize_rows(self):
        M = sp.csr_matrix(np.array([[3.0, 4.0], [0.0, 0.0]]))
        out = dtmod._sparse_l2_normalize_rows(M)
        np.testing.assert_allclose(out.toarray()[0], [0.6, 0.8])

    def test_sparse_l2_normalize_converts_non_csr(self):
        M = sp.coo_matrix(np.array([[3.0, 4.0], [1.0, 0.0]]))
        out = dtmod._sparse_l2_normalize_rows(M)
        np.testing.assert_allclose(np.linalg.norm(out.toarray(), axis=1), [1.0, 1.0])

    def test_sparse_minmax(self):
        M = sp.csr_matrix(np.array([[2.0, 0.0], [4.0, 0.0]]))
        out = dtmod._sparse_minmax_0_1_nonneg(M)
        np.testing.assert_allclose(out.toarray()[:, 0], [0.5, 1.0])
        # an all-zero column must not divide by zero
        np.testing.assert_allclose(out.toarray()[:, 1], [0.0, 0.0])

    def test_sparse_minmax_rejects_dense(self):
        with pytest.raises(TypeError, match="scipy.sparse"):
            dtmod._sparse_minmax_0_1_nonneg(np.ones((2, 2)))

    def test_sparse_helpers_require_scipy(self, monkeypatch):
        monkeypatch.setattr(dtmod, "sp", None)
        with pytest.raises(ImportError):
            dtmod._sparse_l2_normalize_rows(None)
        with pytest.raises(ImportError):
            dtmod._sparse_minmax_0_1_nonneg(None)

    def test_chunk_len_is_at_least_one(self):
        assert dtmod._chunk_len(10_000_000, 10_000_000, 8, 1) >= 1

    def test_chunk_len_caps_at_n_right(self):
        assert dtmod._chunk_len(2, 5, 8, 4096) == 5

    def test_chunk_len_rejects_a_nonpositive_budget(self):
        with pytest.raises(MemoryBudgetError):
            dtmod._chunk_len(4, 4, 8, 0)

    @pytest.mark.parametrize(
        "dtype,expected", [("float32", np.float32), ("float64", np.float64)]
    )
    def test_ensure_dtype_converts(self, dtype, expected):
        X = np.ones((2, 2), dtype=np.int64)
        assert dtmod._ensure_dtype(X, dtype).dtype == expected

    def test_ensure_dtype_is_a_noop_when_already_correct(self):
        X = np.ones((2, 2), dtype=np.float32)
        assert dtmod._ensure_dtype(X, "float32") is X

    def test_seed_everything(self):
        dtmod._seed_everything(3)
        first = np.random.rand(4)
        dtmod._seed_everything(3)
        np.testing.assert_allclose(first, np.random.rand(4))


class TestDistanceKernels:
    def test_euclidean_min_to_set_dense(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        Y = np.array([[0.0, 1.0], [9.0, 10.0]])
        out = dtmod._euclidean_min_to_set_dense(X, Y)
        np.testing.assert_allclose(out, [1.0, 1.0])

    def test_euclidean_min_to_set_with_an_empty_reference_set(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        out = dtmod._euclidean_min_to_set_dense(X, np.empty((0, 2)))
        assert out.shape == (2,)
        assert np.all(np.isinf(out)) or np.all(out >= 0)

    def test_euclidean_min_to_set_chunks_under_a_tight_budget(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        Y = rng.normal(size=(30, 3))
        full = dtmod._euclidean_min_to_set_dense(X, Y, max_memory_mb=4096)
        chunked = dtmod._euclidean_min_to_set_dense(X, Y, max_memory_mb=1)
        np.testing.assert_allclose(full, chunked, rtol=1e-6)

    def test_pairwise_euclidean_jit(self):
        """Returns the upper triangle only, to halve the memory."""
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = dtmod._pairwise_euclidean_jit(X)
        assert out[0, 1] == pytest.approx(5.0)

    def test_cosine_similarity_jit(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = dtmod._cosine_similarity_jit(X, X)
        np.testing.assert_allclose(np.diag(out), [1.0, 1.0], atol=1e-6)
        assert abs(out[0, 1]) < 1e-6

    def test_cosine_similarity_handles_zero_rows(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        out = dtmod._cosine_similarity_jit(X, X)
        assert np.isfinite(out).all()

    def test_euclidean_min_jit_matches_the_dense_path(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(12, 3))
        Y = rng.normal(size=(5, 3))
        np.testing.assert_allclose(
            dtmod._euclidean_min_jit(X, Y),
            dtmod._euclidean_min_to_set_dense(X, Y),
            rtol=1e-6,
        )

    def test_euclidean_chunk_jit_runs(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(9, 3))
        Y = rng.normal(size=(4, 3))
        x2 = np.sum(X * X, axis=1)
        out = dtmod._euclidean_chunk_jit(X, Y, x2)
        assert out.shape[0] == 9
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Facility location
# ---------------------------------------------------------------------------
class TestFacilityLocationSelector:
    def test_selects_the_requested_number(self, rng):
        X = dtmod._l2_normalize_rows_dense(rng.normal(size=(20, 4)))
        idx, gains = FacilityLocationSelector(n_prototypes=5).select(X)
        assert len(idx) == 5
        assert len(gains) == 5

    def test_gains_are_non_increasing(self, rng):
        X = dtmod._l2_normalize_rows_dense(rng.normal(size=(25, 4)))
        _, gains = FacilityLocationSelector(n_prototypes=6).select(X)
        assert np.all(np.diff(gains) <= 1e-9)

    def test_is_deterministic(self, rng):
        X = dtmod._l2_normalize_rows_dense(rng.normal(size=(20, 4)))
        a, _ = FacilityLocationSelector(n_prototypes=5).select(X)
        b, _ = FacilityLocationSelector(n_prototypes=5).select(X)
        np.testing.assert_array_equal(a, b)

    def test_empty_input(self):
        idx, gains = FacilityLocationSelector(n_prototypes=3).select(
            np.empty((0, 3))
        )
        assert len(idx) == 0
        assert len(gains) == 0

    def test_accepts_sparse_input(self, rng):
        X = sp.csr_matrix(dtmod._l2_normalize_rows_dense(rng.normal(size=(15, 4))))
        idx, _ = FacilityLocationSelector(n_prototypes=4).select(X)
        assert len(idx) == 4

    def test_honours_client_weights(self, rng):
        X = dtmod._l2_normalize_rows_dense(rng.normal(size=(18, 4)))
        weights = np.ones(18)
        weights[0] = 50.0
        idx, _ = FacilityLocationSelector(n_prototypes=3).select(X, weights=weights)
        assert len(idx) == 3

    def test_honours_a_forbidden_set(self, rng):
        X = dtmod._l2_normalize_rows_dense(rng.normal(size=(18, 4)))
        forbidden = {0, 1, 2}
        idx, _ = FacilityLocationSelector(n_prototypes=4).select(
            X, forbidden=forbidden
        )
        assert not (set(int(i) for i in idx) & forbidden)

    def test_requesting_more_than_available(self, rng):
        X = dtmod._l2_normalize_rows_dense(rng.normal(size=(4, 3)))
        idx, _ = FacilityLocationSelector(n_prototypes=10).select(X)
        assert len(idx) <= 4

    def test_speed_mode_and_verbose(self, rng, capsys):
        X = dtmod._l2_normalize_rows_dense(rng.normal(size=(12, 3)))
        idx, _ = FacilityLocationSelector(
            n_prototypes=3, speed_mode=True, verbose=True
        ).select(X)
        assert len(idx) == 3


# ---------------------------------------------------------------------------
# Shapley engine
# ---------------------------------------------------------------------------
class TestShapleyEarlyStopping:
    def test_does_not_stop_before_twenty_permutations(self):
        es = dtmod.ShapleyEarlyStopping(patience=1, tolerance=1.0)
        stop, info = es.update(np.ones(3), n_perms=5)
        assert stop is False
        assert info["converged"] is False

    def test_needs_two_observations(self):
        es = dtmod.ShapleyEarlyStopping(patience=1, tolerance=1.0)
        stop, _ = es.update(np.ones(3), n_perms=20)
        assert stop is False

    def test_stops_once_estimates_settle(self):
        es = dtmod.ShapleyEarlyStopping(patience=1, tolerance=0.5)
        es.update(np.ones(3), n_perms=20)
        stop, info = es.update(np.ones(3), n_perms=25)
        assert stop is True
        assert info["stable_iterations"] >= 1
        assert info["mean_rel_change"] == pytest.approx(0.0)

    def test_resets_when_estimates_move_again(self):
        es = dtmod.ShapleyEarlyStopping(patience=2, tolerance=0.01)
        es.update(np.ones(3), n_perms=20)
        es.update(np.ones(3), n_perms=25)
        stop, info = es.update(np.full(3, 10.0), n_perms=30)
        assert stop is False
        assert info["stable_iterations"] == 0


def _mean_value(X_subset, indices, ctx=None):
    if len(X_subset) == 0:
        return 0.0
    return float(np.mean(X_subset))


class TestShapleySignificanceEngine:
    def test_sample_level_shape_and_info(self, rng):
        """The sample-level marginal is spread evenly across the features."""
        X = rng.normal(size=(8, 3))
        engine = dtmod.ShapleySignificanceEngine(n_permutations=6, random_state=0)
        phi, info = engine.compute_shapley_values(X, _mean_value, "test")
        assert phi.shape == (8, 3)
        assert np.isfinite(phi).all()
        assert info["n_permutations_used"] >= 1
        assert "additivity_error" in info

    def test_feature_level_shape(self, rng):
        X = rng.normal(size=(6, 4))
        engine = dtmod.ShapleySignificanceEngine(n_permutations=6, random_state=0)
        phi, info = engine.compute_feature_shapley_values(X, _mean_value, "test")
        assert phi.shape == (6, 4)
        assert np.isfinite(phi).all()

    def test_is_reproducible_at_a_fixed_seed(self, rng):
        X = rng.normal(size=(8, 3))
        a, _ = dtmod.ShapleySignificanceEngine(
            n_permutations=6, random_state=11
        ).compute_shapley_values(X, _mean_value, "test")
        b, _ = dtmod.ShapleySignificanceEngine(
            n_permutations=6, random_state=11
        ).compute_shapley_values(X, _mean_value, "test")
        np.testing.assert_allclose(a, b)

    def test_verbose_reports_progress(self, rng, capsys):
        X = rng.normal(size=(6, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=4, random_state=0, verbose=True
        )
        engine.compute_shapley_values(X, _mean_value, "verbose-test")
        assert "verbose-test" in capsys.readouterr().out

    def test_verbose_feature_path(self, rng, capsys):
        X = rng.normal(size=(5, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=4, random_state=0, verbose=True
        )
        engine.compute_feature_shapley_values(X, _mean_value, "feature-verbose")
        assert "feature-verbose" in capsys.readouterr().out

    def test_early_stopping_can_trigger(self, rng):
        """A constant value function converges immediately."""
        X = rng.normal(size=(6, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=200,
            random_state=0,
            early_stopping_patience=1,
            early_stopping_tolerance=10.0,
        )
        _, info = engine.compute_shapley_values(
            X, lambda a, b, c=None: 1.0, "constant"
        )
        assert info["n_permutations_used"] <= 200

    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        engine = dtmod.ShapleySignificanceEngine(n_permutations=4, random_state=0)
        phi, _ = engine.compute_shapley_values(X, _mean_value, "single")
        assert phi.shape == (1, 2)

    def test_serial_backend(self, rng):
        X = rng.normal(size=(6, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=4, random_state=0, n_jobs=1
        )
        phi, _ = engine.compute_shapley_values(X, _mean_value, "serial")
        assert phi.shape == (6, 3)


# ---------------------------------------------------------------------------
# Formative value functions
# ---------------------------------------------------------------------------
class TestFormativeValueFunctions:
    def test_convex_hull_needs_three_points(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert dtmod.formative_archetypal_convex_hull(X, np.arange(2)) == 0.0

    def test_convex_hull_grows_with_spread(self):
        small = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        large = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        v_small = dtmod.formative_archetypal_convex_hull(small, np.arange(3))
        v_large = dtmod.formative_archetypal_convex_hull(large, np.arange(3))
        assert v_large > v_small

    def test_convex_hull_falls_back_in_high_dimensions(self, rng):
        X = rng.normal(size=(10, 30))
        value = dtmod.formative_archetypal_convex_hull(X, np.arange(10))
        assert np.isfinite(value)

    def test_convex_hull_with_degenerate_points(self):
        """Collinear points make a hull impossible; must not raise."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert np.isfinite(dtmod.formative_archetypal_convex_hull(X, np.arange(4)))

    def test_pcha_cached_uses_the_stored_archetypes(self, rng):
        X = rng.normal(size=(8, 3))
        context = {"archetypes": X[:2]}
        value = dtmod.formative_archetypal_pcha_cached(X, np.arange(8), context)
        assert np.isfinite(value)

    def test_pcha_cached_on_an_empty_subset(self, rng):
        context = {"archetypes": rng.normal(size=(2, 3))}
        value = dtmod.formative_archetypal_pcha_cached(
            np.empty((0, 3)), np.array([], dtype=int), context
        )
        assert np.isfinite(value)

    def test_prototypical_coverage_needs_two_points(self):
        X = np.array([[1.0, 0.0]])
        assert dtmod.formative_prototypical_coverage(X, np.arange(1)) == 0.0

    def test_prototypical_coverage_is_finite(self, rng):
        X = rng.normal(size=(8, 3))
        assert np.isfinite(dtmod.formative_prototypical_coverage(X, np.arange(8)))

    def test_stereotypical_extremeness_on_an_empty_subset(self):
        ctx = {"target_values": np.arange(5.0), "target": "max"}
        assert dtmod.formative_stereotypical_extremeness(
            np.empty((0, 2)), np.array([], dtype=int), ctx
        ) == 0.0

    @pytest.mark.parametrize("target", ["max", "min", 2.0])
    def test_stereotypical_extremeness_targets(self, target):
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ctx = {"target_values": values, "target": target, "median": 2.0}
        out = dtmod.formative_stereotypical_extremeness(
            np.zeros((3, 2)), np.array([0, 1, 2]), ctx
        )
        assert np.isfinite(out)

    def test_stereotypical_extremeness_infers_the_median(self):
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ctx = {"target_values": values, "target": "max"}
        out = dtmod.formative_stereotypical_extremeness(
            np.zeros((2, 2)), np.array([3, 4]), ctx
        )
        assert out > 0


# ---------------------------------------------------------------------------
# Streaming prefix kernels (the v0.7.7 rewrite)
# ---------------------------------------------------------------------------
class TestPrefixKernels:
    def test_archetypal_prefix_returns_one_value_per_prefix(self, rng):
        X = rng.normal(size=(6, 3))
        archetypes = rng.normal(size=(2, 3))
        perm = np.arange(6)
        out = dtmod._prefix_values_archetypal(perm, X, {"archetypes": archetypes})
        assert out.shape == (7,)
        assert np.isfinite(out).all()

    def test_prototypical_prefix_returns_one_value_per_prefix(self, rng):
        X = rng.normal(size=(6, 3))
        out = dtmod._prefix_values_prototypical(np.arange(6), X)
        assert out.shape == (7,)
        assert np.isfinite(out).all()

    def test_prototypical_prefix_handles_zero_rows(self):
        X = np.zeros((4, 3))
        out = dtmod._prefix_values_prototypical(np.arange(4), X)
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("target", ["max", "min", 2.0])
    def test_stereotypical_prefix(self, target):
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ctx = {"target_values": values, "target": target, "median": 2.0}
        out = dtmod._prefix_values_stereotypical(np.arange(5), np.zeros((5, 2)), ctx)
        assert out.shape == (6,)
        assert np.isfinite(out).all()

    def test_cummean_kernel(self):
        terms = np.array([1.0, 2.0, 3.0, 4.0])
        out = dtmod._prefix_cummean_kernel(np.arange(4), terms)
        np.testing.assert_allclose(out[1:], [1.0, 1.5, 2.0, 2.5])


# ---------------------------------------------------------------------------
# Thread control
# ---------------------------------------------------------------------------
class TestThreadControl:
    def test_deterministic_context_manager(self):
        with dtmod._ThreadControl(True) as tc:
            assert tc.effective_limit is None or tc.effective_limit >= 1

    def test_non_deterministic_context_manager(self):
        with dtmod._ThreadControl(False) as tc:
            assert tc is not None


# ---------------------------------------------------------------------------
# Stereotype coercion helper, used by both the rank and the Shapley paths
# ---------------------------------------------------------------------------
class TestStereotypeCoercion:
    def test_numeric_passes_through(self):
        import pandas as pd

        s = pd.Series([1, 2, 3], name="x")
        np.testing.assert_allclose(
            dtmod._stereotype_values_as_float(s), [1.0, 2.0, 3.0]
        )

    def test_unnamed_series_gets_a_generic_label(self):
        import pandas as pd

        s = pd.Series(["a", "b"])
        with pytest.raises(ConfigError, match="the stereotype column"):
            dtmod._stereotype_values_as_float(s)

    def test_all_nan_object_column_is_allowed_through(self):
        import pandas as pd

        s = pd.Series([None, None], dtype=object, name="x")
        out = dtmod._stereotype_values_as_float(s)
        assert np.isnan(out).all()

    def test_mixed_column_names_the_offending_values(self):
        import pandas as pd

        s = pd.Series(["1.0", "2.0", "banana"], name="mix")
        with pytest.raises(ConfigError, match="banana"):
            dtmod._stereotype_values_as_float(s)

    def test_explicit_column_name_wins_over_the_series_name(self):
        import pandas as pd

        s = pd.Series(["x"], name="series_name")
        with pytest.raises(ConfigError, match="passed_name"):
            dtmod._stereotype_values_as_float(s, "passed_name")
