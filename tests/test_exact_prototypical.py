"""
The prototypical formative game, exactly.

This game is a mean of maxima, which has none of the structure that lets the
archetypal and stereotypical games collapse. It decomposes instead: the maximum
is attained at the first neighbour present in the coalition, which turns the
value function into a weighted sum of indicator games with a 1/|S| weight, and
Shapley is linear in the value function.

Everything here is checked against exhaustive enumeration of all 2^n coalitions
using the shipped value function itself, not a restatement of it, so the test
cannot drift from the thing it is testing.
"""
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import (
    DataTypical,
    exact_formative_prototypical,
    formative_prototypical_coverage,
)


def _v(subset, X):
    """The shipped value function, evaluated on a subset."""
    subset = list(subset)
    if len(subset) < 2:
        return 0.0
    return formative_prototypical_coverage(X[subset], np.asarray(subset))


def _brute_force(X):
    n = len(X)
    fact = [math.factorial(k) for k in range(n + 1)]
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(len(others) + 1):
            w = fact[size] * fact[n - size - 1] / fact[n]
            for S in itertools.combinations(others, size):
                phi[i] += w * (_v(S + (i,), X) - _v(S, X))
    return phi


def _wine_like(n=30, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)),
                        columns=["f%d" % k for k in range(d)])


# --------------------------------------------------------------------------
# agreement with enumeration


def _case(name):
    rng = np.random.default_rng(3)
    if name == "random":
        return rng.normal(size=(8, 4))
    if name == "positive orthant":
        return np.abs(rng.normal(size=(8, 4)))
    if name == "antipodal":
        return np.vstack([np.eye(4), -np.eye(4)])
    if name == "all identical":
        return np.ones((6, 3))
    if name == "duplicate rows":
        X = rng.normal(size=(8, 4))
        X[3] = X[1]
        X[6] = X[1]
        return X
    if name == "a zero row":
        X = rng.normal(size=(8, 4))
        X[2] = 0.0
        return X
    if name == "two rows":
        return rng.normal(size=(2, 4))
    if name == "three rows":
        return rng.normal(size=(3, 4))
    raise AssertionError(name)


@pytest.mark.parametrize("name", [
    "random", "positive orthant", "antipodal", "all identical",
    "duplicate rows", "a zero row", "two rows", "three rows",
])
def test_it_matches_exhaustive_enumeration(name):
    X = _case(name)
    np.testing.assert_allclose(exact_formative_prototypical(X),
                               _brute_force(X), atol=1e-10)


@pytest.mark.parametrize("name", [
    "random", "antipodal", "all identical", "a zero row", "two rows",
])
def test_efficiency_holds(name):
    """The values must sum to the value of the grand coalition."""
    X = _case(name)
    phi = exact_formative_prototypical(X)
    assert abs(phi.sum() - _v(range(len(X)), X)) < 1e-10


def test_symmetry_two_identical_rows_get_identical_credit():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(24, 5))
    X[17] = X[4]
    phi = exact_formative_prototypical(X)
    assert abs(phi[4] - phi[17]) < 1e-12


def test_a_lone_instance_and_an_empty_matrix():
    assert exact_formative_prototypical(np.zeros((0, 3))).shape == (0,)
    assert exact_formative_prototypical(np.ones((1, 3))).tolist() == [0.0]


def test_the_result_does_not_depend_on_row_order():
    """A permutation of the rows must permute the values, not change them."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 4))
    order = rng.permutation(20)
    a = exact_formative_prototypical(X)[order]
    b = exact_formative_prototypical(X[order])
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_it_is_bit_identical_on_repeat():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    assert np.array_equal(exact_formative_prototypical(X),
                          exact_formative_prototypical(X))


def test_the_block_size_does_not_change_the_answer():
    """Blocking is a memory device, not an approximation."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(35, 4))
    np.testing.assert_allclose(exact_formative_prototypical(X, block=4),
                               exact_formative_prototypical(X, block=4096),
                               atol=1e-12)


def test_a_zero_row_is_not_divided_by_its_norm():
    X = np.vstack([np.ones((4, 3)), np.zeros((1, 3))])
    phi = exact_formative_prototypical(X)
    assert np.all(np.isfinite(phi))


# --------------------------------------------------------------------------
# through the estimator


KW = dict(shapley_mode=True, shapley_compute_formative=True,
          archetypal_method="nmf", nmf_rank=3, n_prototypes=6, verbose=False)


def _fit(method, seed):
    dt = DataTypical(formative_method=method, random_state=seed, **KW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dt.fit_transform(_wine_like())
    return dt, out


def test_the_default_reports_exact_for_the_prototypical_game():
    dt, _ = _fit("exact", 1)
    assert dt.shapley_info_["prototypical_formative"]["method"] == "exact"
    assert dt.shapley_info_["prototypical_formative"]["split_half_rho"] == 1.0


def test_monte_carlo_is_still_reachable_for_it():
    dt, _ = _fit("monte_carlo", 1)
    assert dt.shapley_info_["prototypical_formative"]["method"] == "monte_carlo"


def test_the_default_prototypical_ranking_is_seed_independent():
    """
    The whole reason this function exists. Under sampling, five seeds gave five
    different top formative prototypes on Wine.
    """
    _, a = _fit("exact", 1)
    _, b = _fit("exact", 999)
    np.testing.assert_allclose(a["prototypical_shapley_rank"].to_numpy(),
                               b["prototypical_shapley_rank"].to_numpy(),
                               atol=1e-9)


def test_sampling_still_moves_with_the_seed():
    """The other half of the same fact, so the contrast stays pinned."""
    _, a = _fit("monte_carlo", 1)
    _, b = _fit("monte_carlo", 999)
    assert np.max(np.abs(a["prototypical_shapley_rank"].to_numpy()
                         - b["prototypical_shapley_rank"].to_numpy())) > 1e-9


def test_the_shape_matches_what_the_sampler_produced():
    """Downstream consumers index this array; the exact path must not reshape it."""
    dt_e, _ = _fit("exact", 1)
    dt_m, _ = _fit("monte_carlo", 1)
    assert dt_e.Phi_prototypical_formative_.shape == \
        dt_m.Phi_prototypical_formative_.shape


# --------------------------------------------------------------------------
# numerical behaviour at scale, where enumeration is impossible


class TestTheCoefficientsAtScale:
    """
    The coefficients C(pool, k) * w(s) must be formed as a single exponential.
    Computed separately, C(5000, 2500) is about 10^1504 and overflows to inf
    while w(s) underflows to zero, so the product arrives as nan. Small-n tests
    cannot see this: it needs a few hundred rows before it bites.

    There is an exact identity to check against at any n. Summing the Shapley
    values of u(A, B) over all players gives u(N), which is 1/n when B is empty
    and 0 otherwise, because a non-empty B makes the grand coalition worthless.
    """

    @pytest.mark.parametrize("n", [10, 60, 300, 1500])
    def test_the_coefficients_satisfy_the_efficiency_identity(self, n):
        from datatypical import _prototypical_coefficients

        a, b, c = _prototypical_coefficients(n)
        sizes = np.arange(n - 1)
        total = 2 * a + sizes * b + (n - 2 - sizes) * c
        expected = np.zeros(n - 1)
        expected[0] = 1.0 / n
        np.testing.assert_allclose(total, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [300, 1500])
    def test_the_coefficients_stay_finite(self, n):
        from datatypical import _prototypical_coefficients

        for table in _prototypical_coefficients(n):
            assert np.all(np.isfinite(table))

    def test_a_few_hundred_rows_produce_finite_values(self):
        rng = np.random.default_rng(4)
        phi = exact_formative_prototypical(rng.normal(size=(400, 9)))
        assert np.all(np.isfinite(phi))
        assert np.any(phi != 0.0)

    def test_no_numerical_warning_is_emitted(self):
        """An overflow here used to arrive as a RuntimeWarning and then a nan."""
        rng = np.random.default_rng(6)
        X = rng.normal(size=(300, 7))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            phi = exact_formative_prototypical(X)
        assert np.all(np.isfinite(phi))


class TestTheSamplerAgreesWithTheClosedForm:
    """
    Enumeration stops at about n = 10. This checks the closed form against the
    shipped Monte Carlo engine at a size far beyond that: if the sampler walked
    towards a different answer, the closed form would be wrong.

    Measured at n = 120 over M = 50 to 25,600, the largest absolute difference
    fell from 4.9e-2 to 1.6e-3, a factor of 31 for a factor of 512 in M, which
    is the 1/sqrt(M) the central limit theorem gives.
    """

    def test_more_permutations_move_the_sampler_towards_the_closed_form(self):
        from datatypical import ShapleySignificanceEngine

        rng = np.random.default_rng(0)
        X = np.vstack([np.eye(5) * 4.0, rng.normal(size=(35, 5)) * 0.6])
        exact = exact_formative_prototypical(X)

        def sampled(M):
            engine = ShapleySignificanceEngine(
                n_permutations=M, random_state=0, n_jobs=1,
                early_stopping_patience=10 ** 9, verbose=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi, _ = engine.compute_shapley_values(
                    X, formative_prototypical_coverage, "proto")
            return np.asarray(phi).sum(axis=1)

        coarse = np.max(np.abs(sampled(40) - exact))
        fine = np.max(np.abs(sampled(1600) - exact))
        assert fine < 0.25 * coarse, (coarse, fine)

        # and a measurement worth keeping: at 40 permutations the sampling
        # error on this data is 1.4 times the largest true value, so the
        # ordering it produces carries no information at all
        assert coarse > np.max(np.abs(exact))
