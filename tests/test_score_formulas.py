"""
The archetypal and prototypical score formulas, checked against data whose
answer is known by construction.

    archetypal_score   = 0.7 * arch_wmax + 0.3 * corner_score
    prototypical_score = 0.5 * (1 - proto_d_norm95) + 0.5 * best_cos

Two properties found during the v0.8.0 sweep are pinned here rather than
changed, because changing them would move every rank the library has ever
produced. Both are documented in the README under "Reading the ranks".
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import DataTypical

KW = dict(archetypal_method="nmf", nmf_rank=3, n_prototypes=5, random_state=0)


def _cloud(n=60, d=4, seed=0, spread=0.2):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) * spread


class TestFormulaWeights:
    """The published blend, pinned so a refactor cannot quietly reweight it."""

    def test_the_archetypal_blend_is_seven_three(self):
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "datatypical.py"
        assert "arch_wmax * 0.7 + corner_score * 0.3" in src.read_text(
            encoding="utf-8")

    def test_the_prototypical_blend_is_half_and_half(self):
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "datatypical.py"
        assert "(1.0 - proto_d_norm95) * 0.5 + best_cos * 0.5" in src.read_text(
            encoding="utf-8")


class TestKnownAnswers:
    def test_a_corner_of_the_unit_cube_maximises_the_corner_term(self):
        corner = np.array([0.0, 1.0])
        m = np.minimum(corner, 1.0 - corner)
        dmin = math.sqrt(float(np.sum(m * m)))
        assert 1.0 - min(dmin / math.sqrt(2), 1.0) == pytest.approx(1.0)

    def test_the_corner_term_is_confined_to_its_upper_half(self):
        """
        m is min(x, 1-x) and so is at most 0.5, making the largest dmin
        0.5*sqrt(d). Dividing by sqrt(d) confines the term to [0.5, 1].
        Documented, not changed: the normaliser sets every archetypal rank.
        """
        centre = np.full(2, 0.5)
        m = np.minimum(centre, 1.0 - centre)
        dmin = math.sqrt(float(np.sum(m * m)))
        assert 1.0 - min(dmin / math.sqrt(2), 1.0) == pytest.approx(0.5)

    def test_a_selected_prototype_scores_one(self):
        """Zero distance to itself and cosine 1 give exactly 1.0."""
        df = pd.DataFrame(_cloud(60, 4, seed=0, spread=1.0), columns=list("abcd"))
        dt = DataTypical(**KW)
        out = dt.fit_transform(df)
        assert out.iloc[dt.prototype_indices_]["prototypical_rank"].min() == \
            pytest.approx(1.0)

    def test_prototypes_occupy_the_top_of_the_column(self):
        df = pd.DataFrame(_cloud(60, 4, seed=0, spread=1.0), columns=list("abcd"))
        dt = DataTypical(**KW)
        out = dt.fit_transform(df)
        chosen = out.iloc[dt.prototype_indices_]["prototypical_rank"]
        others = out.drop(out.index[dt.prototype_indices_])["prototypical_rank"]
        assert chosen.min() >= others.max() - 1e-9

    def test_identical_rows_score_identically(self):
        X = _cloud(40, 4, seed=1, spread=1.0)
        X[5] = X[0]
        df = pd.DataFrame(X, columns=list("abcd"))
        out = DataTypical(**KW).fit_transform(df)
        for col in ["archetypal_rank", "prototypical_rank"]:
            assert out[col].iloc[0] == pytest.approx(out[col].iloc[5])

    def test_an_outlier_is_less_prototypical_than_the_cloud(self):
        X = _cloud(60, 4, seed=3)
        X[0] = np.full(4, 6.0)
        df = pd.DataFrame(X, columns=list("abcd"))
        out = DataTypical(**KW).fit_transform(df)
        assert out["prototypical_rank"].iloc[1:].median() > \
            out["prototypical_rank"].iloc[0]

    @pytest.mark.parametrize("seed", range(4))
    def test_a_moderate_outlier_is_more_archetypal_than_the_cloud(self, seed):
        """
        Holds while the outlier is a few standard deviations out. See
        TestOutlierInversion for what happens further out.
        """
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(60, 4))
        X[0] = np.full(4, 3.0 * X[1:].std())
        df = pd.DataFrame(X, columns=list("abcd"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = DataTypical(**KW).fit_transform(df)
        assert out["archetypal_rank"].iloc[0] > \
            out["archetypal_rank"].iloc[1:].median()


class TestBounds:
    @pytest.mark.parametrize("seed", range(4))
    def test_every_rank_stays_inside_zero_and_one(self, seed):
        df = pd.DataFrame(_cloud(80, 5, seed=seed, spread=1.0),
                          columns=["f%d" % i for i in range(5)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = DataTypical(**KW).fit_transform(df)
        for col in ["archetypal_rank", "prototypical_rank"]:
            values = out[col].to_numpy()
            assert values.min() >= -1e-9
            assert values.max() <= 1 + 1e-9

    def test_the_ranks_have_a_structural_floor_well_above_zero(self):
        """
        Documented in the README: these are scores, not percentiles. A value
        near the observed minimum is the floor of the measure, not a low score.
        """
        floors = []
        for seed in range(4):
            df = pd.DataFrame(_cloud(120, 5, seed=seed, spread=1.0),
                              columns=["f%d" % i for i in range(5)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = DataTypical(**KW).fit_transform(df)
            floors.append(float(out["archetypal_rank"].min()))
        assert min(floors) > 0.25, (
            "the floor moved below 0.25; the README documents roughly 0.4 at "
            "nmf_rank=3 and should be updated with it"
        )

    def test_the_membership_term_cannot_fall_below_one_over_nmf_rank(self):
        """arch_wmax is the max of a row-normalised vector over nmf_rank entries."""
        for rank in [2, 4, 8]:
            df = pd.DataFrame(_cloud(80, 8, seed=rank, spread=1.0),
                              columns=["f%d" % i for i in range(8)])
            dt = DataTypical(archetypal_method="nmf", nmf_rank=rank,
                             n_prototypes=5, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = dt.fit_transform(df)
            # 0.7 * (1/k) + 0.3 * 0.5 is the theoretical floor
            floor = 0.7 * (1.0 / dt.n_archetypes_) + 0.3 * 0.5
            assert out["archetypal_rank"].min() >= floor - 1e-6 or \
                out["archetypal_rank"].min() == pytest.approx(0.3, abs=1e-6), (
                    "nmf_rank=%d: minimum %.4f is below the floor %.4f and is "
                    "not the degenerate-membership value"
                    % (rank, out["archetypal_rank"].min(), floor))


class TestDegenerateMembership:
    """
    A row that is the minimum in every retained feature sits at the origin of
    the scaled space, where neither backend can define an archetype membership.
    Its archetypal_rank falls to the bottom of the column despite the row being
    an extreme point. The arithmetic is unchanged; the silence is not.
    """

    @staticmethod
    def _frame_with_an_all_minimum_row(seed=2):
        X = _cloud(60, 4, seed=seed)
        X[1] = np.full(4, -6.0)
        return pd.DataFrame(X, columns=list("abcd"))

    @pytest.mark.parametrize("method", ["nmf", "aa"])
    def test_it_warns(self, method):
        if method == "aa":
            pytest.importorskip("py_pcha")
        dt = DataTypical(archetypal_method=method, nmf_rank=3, n_prototypes=5,
                         random_state=0)
        with pytest.warns(RuntimeWarning, match="zero archetype membership"):
            dt.fit_transform(self._frame_with_an_all_minimum_row())

    def test_the_warning_counts_the_rows(self):
        dt = DataTypical(**KW)
        with pytest.warns(RuntimeWarning, match=r"1 row\(s\)"):
            dt.fit_transform(self._frame_with_an_all_minimum_row())

    def test_the_warning_says_to_treat_them_as_undefined(self):
        dt = DataTypical(**KW)
        with pytest.warns(RuntimeWarning) as record:
            dt.fit_transform(self._frame_with_an_all_minimum_row())
        message = " ".join(str(w.message) for w in record)
        assert "undefined rather than low" in message

    @pytest.mark.parametrize("method", ["nmf", "aa"])
    def test_the_score_is_unchanged_at_the_corner_term_alone(self, method):
        """0.7 * 0 + 0.3 * 1.0. Pinned so the arithmetic is not altered by accident."""
        if method == "aa":
            pytest.importorskip("py_pcha")
        dt = DataTypical(archetypal_method=method, nmf_rank=3, n_prototypes=5,
                         random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = dt.fit_transform(self._frame_with_an_all_minimum_row())
        assert out["archetypal_rank"].iloc[1] == pytest.approx(0.3)

    def test_ordinary_data_does_not_warn(self):
        df = pd.DataFrame(_cloud(60, 4, seed=7, spread=1.0), columns=list("abcd"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DataTypical(**KW).fit_transform(df)
        assert not [w for w in caught if "membership" in str(w.message)]


class TestOutlierInversion:
    """
    The archetypal measure inverts once a single row dominates a feature's
    range. MinMax scaling then compresses every other row against one end, and
    that end is itself a corner of the unit cube, so the corner term rewards the
    compressed bulk as much as the extreme point.

    Measured across six seeds: an outlier at 3 to 5 standard deviations ranks
    above the cloud median every time; one at 30 standard deviations ranks below
    it every time. This is a property of the published measure. The arithmetic
    is unchanged and these tests pin the behaviour so it cannot drift silently.
    """

    @staticmethod
    def _with_outlier(mult, seed):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(120, 5))
        X[0] = np.full(5, mult * X[1:].std())
        return pd.DataFrame(X, columns=["f%d" % i for i in range(5)])

    @pytest.mark.parametrize("mult", [3, 5])
    def test_a_moderate_outlier_ranks_above_the_median(self, mult):
        wins = 0
        for seed in range(6):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = DataTypical(**KW).fit_transform(self._with_outlier(mult, seed))
            wins += int(out["archetypal_rank"].iloc[0]
                        > out["archetypal_rank"].iloc[1:].median())
        assert wins >= 5, "%dx sd outlier ranked above the median on %d of 6" % (
            mult, wins)

    def test_a_very_extreme_outlier_ranks_below_the_median(self):
        """Pinned as a known property, not endorsed as desirable."""
        below = 0
        for seed in range(6):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = DataTypical(**KW).fit_transform(self._with_outlier(30, seed))
            below += int(out["archetypal_rank"].iloc[0]
                         < out["archetypal_rank"].iloc[1:].median())
        assert below >= 5, (
            "the 30x outlier no longer ranks below the median on %d of 6 seeds; "
            "if the measure was deliberately changed, update this test and the "
            "README section on reading the ranks" % below)

    @pytest.mark.parametrize("mult", [30, 100])
    def test_the_compressed_regime_is_announced(self, mult):
        with pytest.warns(RuntimeWarning, match="dominated by a few extreme"):
            DataTypical(**KW).fit_transform(self._with_outlier(mult, 0))

    @pytest.mark.parametrize("mult", [3, 5])
    def test_the_ordinary_regime_is_not_announced(self, mult):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DataTypical(**KW).fit_transform(self._with_outlier(mult, 0))
        assert not [w for w in caught if "dominated by a few extreme" in str(w.message)]

    def test_clean_data_is_not_announced(self):
        rng = np.random.default_rng(0)
        for data in [rng.normal(size=(200, 5)), rng.uniform(size=(200, 5)),
                     rng.lognormal(size=(200, 5))]:
            df = pd.DataFrame(data, columns=["f%d" % i for i in range(5)])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                DataTypical(**KW).fit_transform(df)
            assert not [w for w in caught
                        if "dominated by a few extreme" in str(w.message)]

    def test_the_warning_names_the_offending_columns(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(120, 4))
        X[0, 2] = 500.0 * X[1:, 2].std()      # only column index 2
        df = pd.DataFrame(X, columns=list("abcd"))
        with pytest.warns(RuntimeWarning) as record:
            DataTypical(**KW).fit_transform(df)
        message = " ".join(str(w.message) for w in record)
        assert "'c'" in message

    def test_a_small_frame_is_not_judged(self):
        """Percentiles are meaningless on a handful of rows."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(10, 3))
        X[0] = np.full(3, 100.0)
        df = pd.DataFrame(X, columns=list("abc"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                        random_state=0).fit_transform(df)
        assert not [w for w in caught
                    if "dominated by a few extreme" in str(w.message)]
