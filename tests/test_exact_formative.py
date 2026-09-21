"""
The closed-form archetypal formative values.

The value function is a mean of "minimum" games, one per archetype, and Shapley
is linear in the value function, so the whole thing has a closed form. The
Monte Carlo estimator is unbiased but converges so slowly on this game that at
the default 100 permutations the ranking is chance-level. The exact computation
removes the sampling entirely, costs O(n log n) per archetype, and is opt-in so
no existing result moves.
"""
import itertools
import math
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import datatypical as dtmod
from datatypical import (
    ConfigError, DataTypical, exact_formative_archetypal,
    exact_formative_stereotypical,
)

KW = dict(shapley_mode=True, shapley_compute_formative=True,
          archetypal_method="nmf", nmf_rank=8, n_prototypes=20,
          shapley_n_permutations=100)


def _frame(n=60, d=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)),
                        columns=["f%d" % i for i in range(d)])


def _brute_force(X, archetypes):
    """Enumerate every coalition. Only viable for a handful of samples."""
    n = X.shape[0]
    ctx = {"archetypes": archetypes}

    def v(S):
        if not S:
            return 0.0
        return dtmod.formative_archetypal_pcha_cached(
            X[list(S)], np.asarray(S), ctx)

    phi = np.zeros(n)
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for size in range(len(rest) + 1):
            w = (math.factorial(size) * math.factorial(n - size - 1)
                 / math.factorial(n))
            for S in itertools.combinations(rest, size):
                phi[i] += w * (v(tuple(S) + (i,)) - v(S))
    return phi


class TestAgainstEnumeration:
    """The only way to know a Shapley formula is right is to enumerate."""

    @pytest.mark.parametrize("n_samples,n_arch", [(5, 1), (6, 2), (7, 3), (8, 4)])
    def test_it_matches_full_enumeration(self, n_samples, n_arch):
        rng = np.random.default_rng(n_samples * 10 + n_arch)
        X = rng.random((n_samples, 3))
        archetypes = rng.random((n_arch, 3))
        np.testing.assert_allclose(
            exact_formative_archetypal(X, archetypes),
            _brute_force(X, archetypes),
            atol=1e-12,
        )

    def test_it_matches_with_duplicate_distances(self):
        """Ties in the sort are where a rank-based formula goes wrong."""
        X = np.array([[0.0], [1.0], [1.0], [2.0], [2.0], [3.0]])
        archetypes = np.array([[0.0]])
        np.testing.assert_allclose(
            exact_formative_archetypal(X, archetypes),
            _brute_force(X, archetypes),
            atol=1e-12,
        )

    def test_the_values_satisfy_efficiency(self):
        """They must sum to v(grand coalition) minus v(empty)."""
        rng = np.random.default_rng(3)
        X = rng.random((25, 4))
        archetypes = rng.random((5, 4))
        total = dtmod.formative_archetypal_pcha_cached(
            X, np.arange(len(X)), {"archetypes": archetypes})
        assert exact_formative_archetypal(X, archetypes).sum() == \
            pytest.approx(total, rel=1e-9)

    def test_identical_samples_get_identical_values(self):
        rng = np.random.default_rng(4)
        X = rng.random((12, 3))
        X[5] = X[0]
        archetypes = rng.random((3, 3))
        phi = exact_formative_archetypal(X, archetypes)
        assert phi[0] == pytest.approx(phi[5])


class TestEdgeCases:
    def test_no_samples(self):
        out = exact_formative_archetypal(np.zeros((0, 3)), np.ones((2, 3)))
        assert out.shape == (0,)

    def test_one_sample_carries_the_whole_value(self):
        X = np.zeros((1, 3))
        archetypes = np.ones((2, 3))
        total = dtmod.formative_archetypal_pcha_cached(
            X, np.array([0]), {"archetypes": archetypes})
        assert exact_formative_archetypal(X, archetypes)[0] == \
            pytest.approx(total)

    def test_a_single_archetype(self):
        rng = np.random.default_rng(5)
        X = rng.random((10, 2))
        archetypes = rng.random((1, 2))
        np.testing.assert_allclose(
            exact_formative_archetypal(X, archetypes),
            _brute_force(X, archetypes), atol=1e-12)

    def test_it_is_fast_enough_to_be_the_default_choice(self):
        rng = np.random.default_rng(6)
        X = rng.random((5000, 10))
        archetypes = rng.random((8, 10))
        start = time.perf_counter()
        exact_formative_archetypal(X, archetypes)
        assert time.perf_counter() - start < 5.0


class TestMonteCarloConvergesToIt:
    def test_more_permutations_move_the_estimate_towards_the_exact_answer(self):
        rng = np.random.default_rng(0)
        X = rng.random((40, 5))
        archetypes = rng.random((6, 5))
        truth = exact_formative_archetypal(X, archetypes)
        ctx = {"archetypes": archetypes}

        rhos = []
        for n_perm in [100, 4000]:
            engine = dtmod.ShapleySignificanceEngine(
                n_permutations=n_perm, random_state=3,
                early_stopping_patience=10 ** 9)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi, _ = engine.compute_shapley_values(
                    X, dtmod.formative_archetypal_pcha_cached, "f", ctx)
            rhos.append(dtmod._rank_correlation(phi.sum(axis=1), truth))
        assert rhos[1] > rhos[0], \
            "the sampler did not move towards the exact answer: %s" % rhos
        assert rhos[1] > 0.8, \
            "even 4000 permutations only reached rho %.3f" % rhos[1]


class TestTheOptIn:
    def test_the_default_is_exact(self):
        """Changed in v0.8.0: sampling did not converge, so it is no longer the default."""
        dt = DataTypical(random_state=1, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        assert dt.formative_method == "exact"
        assert dt.shapley_info_["archetypal_formative"]["method"] == "exact"

    def test_monte_carlo_is_still_reachable(self):
        """The old behaviour has to stay available to reproduce a pre-0.8.0 result."""
        dt = DataTypical(formative_method="monte_carlo", random_state=1, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        assert dt.shapley_info_["archetypal_formative"]["method"] == "monte_carlo"

    def test_exact_is_reproducible_across_seeds(self):
        """The whole point: no sampling, so the seed cannot matter."""
        df = _frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = DataTypical(formative_method="exact", random_state=1,
                            **KW).fit_transform(df)["archetypal_shapley_rank"]
            b = DataTypical(formative_method="exact", random_state=2,
                            **KW).fit_transform(df)["archetypal_shapley_rank"]
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_exact_records_what_it_did(self):
        dt = DataTypical(formative_method="exact", random_state=1, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        info = dt.shapley_info_["archetypal_formative"]
        assert info["method"] == "exact"
        assert info["split_half_rho"] == 1.0
        assert info["n_permutations_used"] == 0

    def test_exact_does_not_warn_about_convergence(self):
        dt = DataTypical(formative_method="exact", random_state=1, **KW)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.fit_transform(_frame())
        assert not [w for w in caught
                    if "has not converged" in str(w.message)
                    and "Archetypal" in str(w.message)]

    def test_the_output_shape_is_unchanged(self):
        """Downstream consumers must not be able to tell the difference."""
        df = _frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mc = DataTypical(random_state=1, **KW)
            mc.fit_transform(df)
            ex = DataTypical(formative_method="exact", random_state=1, **KW)
            ex.fit_transform(df)
        assert (ex.Phi_archetypal_formative_.shape
                == mc.Phi_archetypal_formative_.shape)

    def test_the_exact_row_sums_match_the_closed_form(self):
        df = _frame(40, 5)
        dt = DataTypical(formative_method="exact", random_state=1, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        scaled = dt.scaler_.transform(
            dt._df_original_fit[dt.feature_columns_].to_numpy(dtype=np.float64)
        )[:, dt.keep_mask_].astype(np.float64)
        expected = exact_formative_archetypal(scaled, dt.H_.astype(np.float64))
        # The fit works in the float32 working dtype while this reconstruction
        # is float64, so agreement is to float32 precision, not to the last bit.
        np.testing.assert_allclose(
            dt.Phi_archetypal_formative_.sum(axis=1), expected, atol=1e-6)

    def test_all_three_games_are_exact_under_the_default(self):
        """
        The archetypal and stereotypical games reduce outright. The
        prototypical one does not reduce but it decomposes, which is enough.
        """
        dt = DataTypical(formative_method="exact", random_state=1, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        assert dt.Phi_prototypical_formative_ is not None
        for game in ["archetypal_formative", "prototypical_formative"]:
            assert dt.shapley_info_[game]["method"] == "exact", game

    @pytest.mark.parametrize("bad", ["wibble", "exactly", "MC", ""])
    def test_an_unknown_method_is_rejected(self, bad):
        with pytest.raises(ConfigError, match="formative_method"):
            DataTypical(formative_method=bad, **KW).fit(_frame())

    def test_verbose_says_which_route_it_took(self, capsys):
        dt = DataTypical(formative_method="exact", random_state=1,
                         verbose=True, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        assert "exact closed form" in capsys.readouterr().out

    def test_it_survives_a_config_round_trip(self):
        dt = DataTypical(formative_method="exact", **KW)
        assert DataTypical.from_config(dt.to_config()).formative_method == "exact"


def _brute_stereo(vals, target, median):
    n = len(vals)
    ctx = {"target_values": np.asarray(vals, dtype=np.float64),
           "target": target, "median": median}

    def v(S):
        if not S:
            return 0.0
        return dtmod.formative_stereotypical_extremeness(
            np.zeros((len(S), 2)), np.asarray(S), ctx)

    phi = np.zeros(n)
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for size in range(len(rest) + 1):
            w = (math.factorial(size) * math.factorial(n - size - 1)
                 / math.factorial(n))
            for S in itertools.combinations(rest, size):
                phi[i] += w * (v(tuple(S) + (i,)) - v(S))
    return phi


class TestExactStereotypical:
    """The stereotypical formative game is a plain mean game."""

    @pytest.mark.parametrize("n", [5, 6, 7, 8])
    @pytest.mark.parametrize("target", ["max", "min", 2.5])
    def test_it_matches_full_enumeration(self, n, target):
        rng = np.random.default_rng(n)
        vals = rng.normal(size=n) * 10
        median = float(np.median(vals))
        np.testing.assert_allclose(
            exact_formative_stereotypical(vals, target, median),
            _brute_stereo(vals, target, median),
            atol=1e-10,
        )

    @pytest.mark.parametrize("target", ["max", "min", 1.0])
    def test_the_values_satisfy_efficiency(self, target):
        rng = np.random.default_rng(0)
        vals = rng.normal(size=30) * 5
        median = float(np.median(vals))
        total = dtmod.formative_stereotypical_extremeness(
            np.zeros((30, 2)), np.arange(30),
            {"target_values": vals, "target": target, "median": median})
        assert exact_formative_stereotypical(vals, target, median).sum() ==             pytest.approx(total, rel=1e-9)

    def test_it_is_affine_in_the_deviation(self):
        """
        The consequence worth knowing: this ranking is a monotone transform of
        the stereotype column and carries no information beyond it.
        """
        rng = np.random.default_rng(0)
        vals = rng.normal(size=200) * 10
        median = float(np.median(vals))
        phi = exact_formative_stereotypical(vals, "max", median)
        deviation = np.maximum(vals - median, 0.0)
        assert abs(np.corrcoef(phi, deviation)[0, 1]) == pytest.approx(1.0)

    def test_the_median_is_computed_when_not_given(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(
            exact_formative_stereotypical(vals, "max"),
            exact_formative_stereotypical(vals, "max", float(np.median(vals))))

    def test_no_samples(self):
        assert exact_formative_stereotypical(np.zeros(0), "max").shape == (0,)

    def test_one_sample_carries_the_whole_value(self):
        assert exact_formative_stereotypical(
            np.array([5.0]), "max", 0.0)[0] == pytest.approx(5.0)

    def test_an_invalid_target_is_rejected(self):
        with pytest.raises(ConfigError, match="target must be"):
            exact_formative_stereotypical(np.arange(5.0), "middling")

    def test_the_fit_uses_it_and_is_seed_independent(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(50, 5)),
                          columns=["f%d" % i for i in range(5)])
        df["Age"] = rng.integers(30, 85, 50).astype(float)
        kw = dict(KW)
        kw.update(stereotype_column="Age", stereotype_target="max",
                  nmf_rank=4, n_prototypes=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = DataTypical(formative_method="exact", random_state=1, **kw)
            ra = a.fit_transform(df)["stereotypical_shapley_rank"].to_numpy()
            b = DataTypical(formative_method="exact", random_state=2, **kw)
            rb = b.fit_transform(df)["stereotypical_shapley_rank"].to_numpy()
        np.testing.assert_array_equal(ra, rb)
        assert a.shapley_info_["stereotypical_formative"]["method"] == "exact"

    def test_verbose_says_which_route_it_took(self, capsys):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
        df["Age"] = rng.integers(30, 85, 40).astype(float)
        kw = dict(KW)
        kw.update(stereotype_column="Age", nmf_rank=4, n_prototypes=10)
        dt = DataTypical(formative_method="exact", random_state=1,
                         verbose=True, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        assert "Stereotypical formative: exact closed" in capsys.readouterr().out

    def test_monte_carlo_still_samples_when_asked(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
        df["Age"] = rng.integers(30, 85, 40).astype(float)
        kw = dict(KW)
        kw.update(stereotype_column="Age", nmf_rank=4, n_prototypes=10)
        dt = DataTypical(formative_method="monte_carlo", random_state=1, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        assert dt.shapley_info_["stereotypical_formative"]["method"] == "monte_carlo"

    def test_the_stereotypical_default_is_exact(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
        df["Age"] = rng.integers(30, 85, 40).astype(float)
        kw = dict(KW)
        kw.update(stereotype_column="Age", nmf_rank=4, n_prototypes=10)
        dt = DataTypical(random_state=1, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        assert dt.shapley_info_["stereotypical_formative"]["method"] == "exact"

    def test_the_prototypical_game_is_exact_too(self):
        """
        Its value is a mean of per-member maxima, which does not reduce to an
        order statistic the way a mean of minimum games does. It decomposes
        instead: see tests/test_exact_prototypical.py for the enumeration
        check. Here we only pin that the estimator uses it.
        """
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
        dt = DataTypical(formative_method="exact", random_state=1, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        assert dt.Phi_prototypical_formative_ is not None
        assert dt.shapley_info_["prototypical_formative"]["method"] == "exact"
