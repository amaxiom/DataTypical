"""
Knee detection and non-finite formative values.

Two defects from the v0.8.0 sweep:

* `auto_n_prototypes='kneedle'` cut the prototype set to a single prototype on
  every dataset tried, without a word. Facility-location gains fall away
  steeply, so the first prototype carries most of the coverage and the knee sits
  at index 1.
* A non-finite formative value landed in the equal-values branch of the rank
  normaliser, because every comparison against NaN is False, and every sample
  was then reported as 0.5. A confident mid-rank for the whole dataset stood in
  for "could not be computed".
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import DataTypical


def _frame(n=80, d=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)),
                        columns=["f%d" % i for i in range(d)])


class TestKneedle:
    def test_a_drastic_truncation_is_announced(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=15,
                         auto_n_prototypes="kneedle", random_state=0)
        with pytest.warns(RuntimeWarning, match="kneedle"):
            dt.fit(_frame())

    def test_the_warning_names_both_counts(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=15,
                         auto_n_prototypes="kneedle", random_state=0)
        with pytest.warns(RuntimeWarning) as record:
            dt.fit(_frame())
        message = " ".join(str(w.message) for w in record)
        assert "from 15 to" in message

    def test_knee_reports_what_actually_truncated(self):
        """
        The Kneedle value was assigned and then overwritten by a second,
        different heuristic, so the number that shaped the result was lost.
        """
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=15,
                         auto_n_prototypes="kneedle", random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(_frame())
        assert dt.knee_ == len(dt.prototype_indices_)

    def test_marginal_gains_match_the_prototypes_kept(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=15,
                         auto_n_prototypes="kneedle", random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(_frame())
        assert len(dt.marginal_gains_) == len(dt.prototype_indices_)

    def test_without_kneedle_knee_keeps_its_diagnostic_meaning(self):
        """The second-difference heuristic is unchanged when Kneedle is off."""
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=8,
                         random_state=0)
        dt.fit(_frame())
        mg = dt.marginal_gains_
        assert len(mg) > 2
        diffs2 = np.diff(np.diff(mg))
        assert dt.knee_ == int(np.argmax(np.abs(diffs2)) + 1)

    def test_no_warning_when_kneedle_is_not_requested(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=15,
                         random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.fit(_frame())
        assert not [w for w in caught if "kneedle" in str(w.message)]

    @pytest.mark.parametrize("gains", [
        np.array([10.0, 5.0, 1.0, 0.5, 0.1]),
        np.array([1.0, 1.0, 1.0]),
        np.linspace(10.0, 1.0, 20),
    ])
    def test_kneedle_returns_an_index_inside_the_array(self, gains):
        k = DataTypical()._kneedle(gains)
        assert k is None or 1 <= k <= gains.size

    def test_kneedle_on_all_zero_gains(self):
        assert DataTypical()._kneedle(np.zeros(5)) is None

    def test_a_very_short_gain_list_reports_its_own_length(self):
        """Fewer than three gains leaves no second difference to take."""
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=2,
                         random_state=0)
        dt.fit(_frame(30, 4, seed=3))
        assert dt.knee_ == len(dt.marginal_gains_)


class TestNonFiniteFormativeRanks:
    def _fitted(self, seed=0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
        dt = DataTypical(shapley_mode=True, shapley_n_permutations=4,
                         shapley_compute_formative=True,
                         archetypal_method="nmf", nmf_rank=2, n_prototypes=4,
                         random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        return dt

    def test_an_all_nan_matrix_gives_nan_ranks_not_a_confident_half(self):
        dt = self._fitted()
        dt.Phi_archetypal_formative_ = np.full((20, 2), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranks = dt._compute_shapley_formative_ranks()
        values = ranks["archetypal_shapley_rank"].to_numpy()
        assert np.isnan(values).all(), \
            "non-finite formative values were reported as %r" % np.unique(values)

    def test_it_warns_and_names_the_significance_type(self):
        dt = self._fitted()
        dt.Phi_prototypical_formative_ = np.full((20, 2), np.nan)
        with pytest.warns(RuntimeWarning, match="prototypical"):
            dt._compute_shapley_formative_ranks()

    def test_it_counts_the_affected_rows(self):
        dt = self._fitted()
        phi = dt.Phi_archetypal_formative_.copy()
        phi[:3] = np.nan
        dt.Phi_archetypal_formative_ = phi
        with pytest.warns(RuntimeWarning, match="3 of 20"):
            dt._compute_shapley_formative_ranks()

    def test_the_finite_rows_are_still_ranked(self):
        """A few bad rows must not discard the rest."""
        dt = self._fitted()
        phi = dt.Phi_archetypal_formative_.copy()
        phi[:2] = np.nan
        dt.Phi_archetypal_formative_ = phi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranks = dt._compute_shapley_formative_ranks()
        values = ranks["archetypal_shapley_rank"].to_numpy()
        assert np.isnan(values[:2]).all()
        assert np.isfinite(values[2:]).all()
        assert values[2:].min() >= 0.0 and values[2:].max() <= 1.0

    def test_an_infinite_value_is_treated_the_same_as_nan(self):
        dt = self._fitted()
        phi = dt.Phi_archetypal_formative_.copy()
        phi[0, 0] = np.inf
        dt.Phi_archetypal_formative_ = phi
        with pytest.warns(RuntimeWarning, match="finite"):
            ranks = dt._compute_shapley_formative_ranks()
        assert np.isnan(ranks["archetypal_shapley_rank"].to_numpy()[0])

    def test_genuinely_constant_values_still_report_a_half(self):
        """All equally formative is a real answer, not a failure."""
        dt = self._fitted()
        dt.Phi_archetypal_formative_ = np.ones((20, 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ranks = dt._compute_shapley_formative_ranks()
        assert not [w for w in caught if "finite" in str(w.message)]
        assert (ranks["archetypal_shapley_rank"].to_numpy() == 0.5).all()

    def test_surviving_values_that_are_all_equal_report_a_half(self):
        dt = self._fitted()
        phi = np.ones((20, 2))
        phi[:3] = np.nan
        dt.Phi_archetypal_formative_ = phi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranks = dt._compute_shapley_formative_ranks()
        values = ranks["archetypal_shapley_rank"].to_numpy()
        assert np.isnan(values[:3]).all()
        assert (values[3:] == 0.5).all()

    def test_a_single_surviving_value_reports_a_half(self):
        dt = self._fitted()
        phi = np.full((20, 2), np.nan)
        phi[7] = 1.0
        dt.Phi_archetypal_formative_ = phi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranks = dt._compute_shapley_formative_ranks()
        values = ranks["archetypal_shapley_rank"].to_numpy()
        assert values[7] == 0.5
        assert np.isnan(np.delete(values, 7)).all()

    def test_an_ordinary_fit_does_not_warn(self):
        dt = self._fitted(seed=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt._compute_shapley_formative_ranks()
        assert not [w for w in caught if "finite" in str(w.message)]


class TestParallelDeterminism:
    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_n_jobs_does_not_change_a_fit(self, n_jobs):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.normal(size=(40, 5)),
                          columns=["f%d" % i for i in range(5)])
        common = dict(shapley_mode=True, shapley_n_permutations=8,
                      shapley_compute_formative=True, archetypal_method="nmf",
                      nmf_rank=3, n_prototypes=5, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = DataTypical(n_jobs=1, **common).fit_transform(df)
            got = DataTypical(n_jobs=n_jobs, **common).fit_transform(df)
        for col in [c for c in reference.columns if c.endswith("_rank")]:
            np.testing.assert_allclose(
                reference[col].fillna(0).to_numpy(),
                got[col].fillna(0).to_numpy(), atol=1e-9,
                err_msg="%s changed with n_jobs=%s" % (col, n_jobs),
            )
