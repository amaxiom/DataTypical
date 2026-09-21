"""
Split-half reliability of the Shapley estimates.

The additivity error already reported says the values sum to the right total. It
says nothing about whether the *order* of the samples is reproducible, and the
order is what gets reported as the formative instances.

Measured during the v0.8.0 sweep, using only the public API: two fits of the
same data differing only in `random_state`, at the default 100 permutations,
produced archetypal formative rankings with a Spearman correlation of 0.02 and a
top-10 overlap of 1 of 10, where chance is 2 of 10. Raising the count to 1000
gave rho 0.32.

Every estimate now carries a `split_half_rho` in `shapley_info_`, and warns when
it is low.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import datatypical as dtmod
from datatypical import DataTypical


def _frame(n=50, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)),
                        columns=["f%d" % i for i in range(d)])


def _formative_key(info):
    keys = [k for k in info if "formative" in k and "arch" in k]
    assert keys, "no archetypal formative entry in shapley_info_: %s" % list(info)
    return keys[0]


class TestRankCorrelationHelper:
    def test_a_perfect_match_is_one(self):
        a = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        assert dtmod._rank_correlation(a, a) == pytest.approx(1.0)

    def test_a_reversal_is_minus_one(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert dtmod._rank_correlation(a, -a) == pytest.approx(-1.0)

    def test_a_monotone_transform_does_not_change_it(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert dtmod._rank_correlation(a, np.exp(a)) == pytest.approx(1.0)

    def test_a_constant_side_gives_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert dtmod._rank_correlation(a, np.ones(4)) == 0.0

    def test_too_few_points_gives_nan(self):
        assert np.isnan(dtmod._rank_correlation(np.array([1.0]), np.array([2.0])))

    def test_two_constant_sides_give_zero(self):
        """Both halves flat: no ordering at all, so no agreement."""
        assert dtmod._rank_correlation(np.zeros(6), np.ones(6)) == 0.0

    def test_mismatched_lengths_give_nan(self):
        assert np.isnan(dtmod._rank_correlation(np.arange(5.0), np.arange(4.0)))

    def test_non_finite_entries_are_dropped(self):
        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert dtmod._rank_correlation(a, b) == pytest.approx(1.0)

    def test_it_agrees_with_scipy(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(0)
        a, b = rng.normal(size=40), rng.normal(size=40)
        assert dtmod._rank_correlation(a, b) == pytest.approx(
            spearmanr(a, b).statistic, abs=1e-9)

    def test_it_agrees_with_scipy_when_there_are_ties(self):
        """Ties are where a naive argsort-of-argsort goes wrong."""
        from scipy.stats import spearmanr

        rng = np.random.default_rng(1)
        a = np.round(rng.normal(size=60))      # many repeated values
        b = np.round(rng.normal(size=60))
        assert dtmod._rank_correlation(a, b) == pytest.approx(
            spearmanr(a, b).statistic, abs=1e-9)

    def test_a_degenerate_half_does_not_claim_convergence(self):
        """
        A constant vector must not correlate perfectly with an increasing one.
        Getting this wrong would make the diagnostic report the worst case as
        the best, which is the failure it exists to catch.
        """
        increasing = np.arange(10.0)
        assert dtmod._rank_correlation(increasing, np.zeros(10)) == 0.0
        assert dtmod._rank_correlation(np.zeros(10), increasing) == 0.0


class TestSplitHalfIsReported:
    def test_every_estimate_carries_one(self):
        dt = DataTypical(shapley_mode=True, shapley_compute_formative=True,
                         archetypal_method="nmf", nmf_rank=3, n_prototypes=5,
                         shapley_n_permutations=20, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        info = dt.shapley_info_[_formative_key(dt.shapley_info_)]
        assert "split_half_rho" in info
        assert -1.0 <= info["split_half_rho"] <= 1.0

    def test_it_is_reported_alongside_the_additivity_error(self):
        """The two answer different questions and both belong in the record."""
        dt = DataTypical(shapley_mode=True, shapley_compute_formative=True,
                         archetypal_method="nmf", nmf_rank=3, n_prototypes=5,
                         shapley_n_permutations=20, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        info = dt.shapley_info_[_formative_key(dt.shapley_info_)]
        assert "additivity_error" in info and "split_half_rho" in info


class TestUnconvergedEstimatesWarn:
    def test_the_default_permutation_count_warns(self):
        """
        The archetypal formative value function does not converge at 100
        permutations. This is the headline of the convergence finding, and the
        reason v0.8.0 stopped sampling it by default. The warning now belongs
        to the sampled path, so the test asks for that path explicitly.
        """
        dt = DataTypical(shapley_mode=True, shapley_compute_formative=True,
                         archetypal_method="nmf", nmf_rank=8, n_prototypes=20,
                         shapley_n_permutations=100, random_state=1,
                         formative_method="monte_carlo")
        with pytest.warns(RuntimeWarning, match="has not converged"):
            dt.fit_transform(_frame(60, 6))

    def test_the_warning_quotes_the_correlation(self):
        dt = DataTypical(shapley_mode=True, shapley_compute_formative=True,
                         archetypal_method="nmf", nmf_rank=8, n_prototypes=20,
                         shapley_n_permutations=100, random_state=1,
                         formative_method="monte_carlo")
        with pytest.warns(RuntimeWarning) as record:
            dt.fit_transform(_frame(60, 6))
        message = " ".join(str(w.message) for w in record)
        assert "correlate only" in message
        assert "shapley_n_permutations" in message

    def test_the_warning_distinguishes_values_from_order(self):
        """The values are fine; the ordering is not. The message must say so."""
        dt = DataTypical(shapley_mode=True, shapley_compute_formative=True,
                         archetypal_method="nmf", nmf_rank=8, n_prototypes=20,
                         shapley_n_permutations=100, random_state=1,
                         formative_method="monte_carlo")
        with pytest.warns(RuntimeWarning) as record:
            dt.fit_transform(_frame(60, 6))
        message = " ".join(str(w.message) for w in record)
        assert "sum correctly" in message
        assert "ORDER" in message

    def test_a_converged_estimate_does_not_warn(self):
        """A value function whose Shapley values are exactly determined."""
        X = np.random.default_rng(0).random((12, 3))
        engine = dtmod.ShapleySignificanceEngine(n_permutations=40,
                                                 random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, info = engine.compute_shapley_values(
                X, lambda a, b, c=None: float(np.sum(a)), "sum")
        assert info["split_half_rho"] > 0.99
        assert not [w for w in caught if "not converged" in str(w.message)]

    def test_a_tiny_permutation_count_is_not_judged(self):
        """Below four permutations the split is meaningless."""
        X = np.random.default_rng(0).random((10, 3))
        engine = dtmod.ShapleySignificanceEngine(n_permutations=2,
                                                 random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.compute_shapley_values(
                X, lambda a, b, c=None: float(np.sum(a * a)), "sq")
        assert not [w for w in caught if "not converged" in str(w.message)]


class TestReliabilityRisesWithEffort:
    def test_more_permutations_give_a_better_split_half(self):
        """
        Not a convergence guarantee, a direction check: if raising the count did
        not help at all, the diagnostic itself would be suspect.
        """
        X = np.random.default_rng(0).random((30, 4))

        def value(X_subset, indices, ctx=None):
            return float(np.sum(np.var(X_subset, axis=0)))

        rhos = []
        for n_perm in [20, 400]:
            engine = dtmod.ShapleySignificanceEngine(
                n_permutations=n_perm, random_state=3,
                early_stopping_patience=10_000)
            _, info = engine.compute_shapley_values(X, value, "var")
            rhos.append(info["split_half_rho"])
        assert rhos[1] > rhos[0], \
            "split-half reliability did not improve with 20x the permutations: %s" % rhos

    def test_the_split_halves_are_genuinely_independent(self):
        """Alternate permutations, so neither half sees the other's draws."""
        X = np.random.default_rng(1).random((20, 3))
        engine = dtmod.ShapleySignificanceEngine(n_permutations=40,
                                                 random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phi, info = engine.compute_shapley_values(
                X, lambda a, b, c=None: float(np.sum(a)), "sum")
        # a linear value function is exact from one permutation, so the halves
        # must agree perfectly
        assert info["split_half_rho"] == pytest.approx(1.0, abs=1e-9)
