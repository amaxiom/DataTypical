"""
Regression tests for the three defects fixed in v0.8.1.

The first of them shipped in 0.8.0 and made archetypal analysis, the library's
headline method, fail on any fresh install. The suite had 738 tests and none of
them caught it, because every test asserts that a fit SUCCEEDS and none assert
WHICH BACKEND RAN. That blind spot is the reason a published paper contains
results computed on a substituted backend, so it is pinned here first.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import datatypical
from datatypical import (
    DataTypical,
    DataTypicalError,
    exact_formative_prototypical,
)


def _frame(n=40, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)),
                        columns=["f%d" % k for k in range(d)])


# ---------------------------------------------------------------------------
# 1. NumPy 2 removed np.mat, which py_pcha calls


class TestArchetypalAnalysisActuallyRuns:
    """
    The gap this closes: asserting a fit succeeded says nothing about which
    backend produced it. `archetypal_method='aa'` promises PCHA specifically.
    """

    def test_the_numpy_alias_py_pcha_needs_is_present(self):
        """
        np.mat was removed in NumPy 2.0. py_pcha calls it in both of its
        modules, so without the alias every PCHA fit raises.
        """
        assert hasattr(np, "mat"), (
            "np.mat is absent, so py_pcha cannot run; the compatibility shim "
            "in datatypical.py did not take effect"
        )

    def test_aa_reports_the_backend_it_actually_used(self):
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, n_prototypes=5,
                         random_state=0, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        assert dt.archetypal_backend_ == "pcha", (
            "archetypal_method='aa' silently ran %r instead of PCHA"
            % dt.archetypal_backend_
        )

    def test_aa_leaves_the_pcha_signature_on_the_estimator(self):
        """
        Independent of archetypal_backend_, so a saved fit can be audited:
        nmf_model_ set means NMF, a numeric reconstruction_error_ means PCHA,
        both absent means ConvexHull.
        """
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, n_prototypes=5,
                         random_state=0, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(_frame())
        assert dt.nmf_model_ is None
        assert isinstance(dt.reconstruction_error_, float)

    def test_the_declared_backend_and_the_signature_agree(self):
        """They are written by different code paths and could disagree."""
        for method in ("aa", "nmf"):
            dt = DataTypical(archetypal_method=method, nmf_rank=3,
                             n_prototypes=5, random_state=0, verbose=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt.fit_transform(_frame())
            from_signature = ("nmf" if dt.nmf_model_ is not None else
                              ("pcha" if dt.reconstruction_error_ is not None
                               else "convexhull"))
            assert dt.archetypal_backend_ == from_signature, method


# ---------------------------------------------------------------------------
# 2. stereotypical_rank was scale-dependent


class TestStereotypicalRankIsScaleFree:
    """
    A rank is scale-free by definition, but the guard tested the spread of a
    user-supplied column against an ABSOLUTE 1e-12. A column of concentrations
    or mole fractions could have perfect structure and still collapse to
    all-1.0, silently. Same family as the scale-invariance defects in MissLearn.
    """

    @staticmethod
    def _fit_at(scale):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=60)
        df = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
        df["target"] = signal * scale
        dt = DataTypical(nmf_rank=3, archetypal_method="nmf",
                         stereotype_column="target", stereotype_target="max",
                         verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = dt.fit_transform(df)
        return out["stereotypical_rank"].to_numpy(), signal

    @pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12, 1e-13, 1e-15])
    def test_the_ranking_survives_rescaling_the_column(self, scale):
        ranks, signal = self._fit_at(scale)
        assert len(np.unique(ranks)) > 1, (
            "stereotypical_rank collapsed to a constant at scale %g" % scale)
        from scipy.stats import spearmanr
        assert abs(spearmanr(ranks, signal).statistic) > 0.99

    def test_the_ranking_is_identical_at_1_and_at_1e_minus_13(self):
        a, _ = self._fit_at(1.0)
        b, _ = self._fit_at(1e-13)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_genuine_float64_exhaustion_still_warns(self):
        """
        When the spread really is below float64 resolution the rank cannot be
        computed, and that must be said rather than returned as a confident 1.0.
        """
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
        df["target"] = rng.normal(size=30) * 1e-20
        dt = DataTypical(nmf_rank=3, archetypal_method="nmf",
                         stereotype_column="target", stereotype_target="max",
                         verbose=False)
        with pytest.warns(RuntimeWarning, match="carries no information"):
            dt.fit_transform(df)

    def test_a_genuinely_constant_column_warns_too(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
        df["target"] = 7.0
        dt = DataTypical(nmf_rank=3, archetypal_method="nmf",
                         stereotype_column="target", stereotype_target="max",
                         verbose=False)
        with pytest.warns(RuntimeWarning, match="carries no information"):
            dt.fit_transform(df)


# ---------------------------------------------------------------------------
# 3. exact_formative_prototypical absorbed NaN


class TestTheExactFunctionsAgreeAboutNaN:
    """
    A NaN made that row's norm NaN, and the `vals > 0.0` selection step compares
    False against NaN, so the affected similarities were DROPPED rather than
    propagated. The function returned an all-finite, fully rankable vector
    computed from an unknown subset of the data, and the NaN-bearing sample took
    an ordinary position in the ranking. Both siblings propagate NaN.
    """

    @pytest.mark.parametrize("placement", ["cell", "row", "column", "all"])
    def test_non_finite_input_is_refused(self, placement):
        rng = np.random.default_rng(11)
        X = rng.random((12, 4)) + 0.5
        if placement == "cell":
            X[3, 1] = np.nan
        elif placement == "row":
            X[3] = np.nan
        elif placement == "column":
            X[:, 1] = np.nan
        else:
            X[:] = np.nan
        with pytest.raises(DataTypicalError, match="non-finite"):
            exact_formative_prototypical(X)

    def test_infinity_is_refused_as_well(self):
        rng = np.random.default_rng(2)
        X = rng.random((8, 3)) + 0.5
        X[5, 0] = np.inf
        with pytest.raises(DataTypicalError, match="non-finite"):
            exact_formative_prototypical(X)

    def test_the_message_names_the_count(self):
        rng = np.random.default_rng(3)
        X = rng.random((8, 3)) + 0.5
        X[1, 1] = np.nan
        X[4, 2] = np.nan
        with pytest.raises(DataTypicalError, match="2 non-finite"):
            exact_formative_prototypical(X)

    def test_clean_input_is_unaffected(self):
        """The guard must not change any answer that was already correct."""
        rng = np.random.default_rng(7)
        X = rng.random((20, 4)) + 0.5
        phi = exact_formative_prototypical(X)
        assert np.all(np.isfinite(phi))
        assert len(np.unique(np.round(phi, 12))) > 1
