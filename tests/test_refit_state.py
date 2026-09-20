"""
State carried between fits on the same estimator.

Through v0.7.7 several attributes were written only on the path that produced
them and read back with `hasattr`, so they survived into the next fit. The
worst case was fitting text after tabular: `_df_original_fit`,
`feature_columns_` and `keep_mask_` still pointed at the tabular frame, so the
visualisations would have described the wrong data rather than refusing.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import DataTypical

KW = dict(archetypal_method="nmf", nmf_rank=3, n_prototypes=5, random_state=0)
RANKS = ["archetypal_rank", "prototypical_rank", "stereotypical_rank"]


def _frame(n=40, d=5, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, d)),
                      columns=["f%d" % i for i in range(d)])
    df["Age"] = rng.integers(30, 85, n).astype(float)
    return df


CORPUS = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon",
          "delta epsilon zeta", "epsilon zeta eta"]


class TestRefitting:
    def test_a_reused_instance_matches_a_fresh_one(self):
        first, second = _frame(40, 5, seed=0), _frame(30, 5, seed=1)
        reused = DataTypical(**KW)
        reused.fit_transform(first)
        got = reused.fit_transform(second)
        expected = DataTypical(**KW).fit_transform(second)
        for col in RANKS:
            np.testing.assert_allclose(got[col].to_numpy(),
                                       expected[col].to_numpy(), atol=1e-9)

    def test_dropped_columns_are_reset(self):
        dirty = _frame(30, 4, seed=2)
        dirty["flat"] = 1.0
        clean = _frame(30, 4, seed=3)
        dt = DataTypical(**KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(dirty)
        assert dt.dropped_columns_ == ["flat"]
        dt.fit(clean)
        assert dt.dropped_columns_ == [], \
            "the previous fit's dropped columns were still reported"

    def test_missingness_is_reset(self):
        holed = _frame(30, 4, seed=4)
        holed.iloc[0, 0] = np.nan
        dt = DataTypical(**KW)
        dt.fit(holed)
        assert any(v > 0 for v in dt.missingness_.values())
        dt.fit(_frame(30, 4, seed=5))
        assert all(v == 0 for v in dt.missingness_.values())

    def test_the_stereotype_source_is_cleared(self):
        df = _frame(30, 4, seed=6)
        dt = DataTypical(shapley_mode=True, shapley_n_permutations=4,
                         shapley_compute_formative=False,
                         stereotype_column="Age", **KW)
        dt.fit_transform(df)
        assert dt._stereotype_source_fit_ is not None
        dt.stereotype_column = None
        dt.fit_transform(df)
        assert dt._stereotype_source_fit_ is None, \
            "a stale stereotype source survived into a fit without one"

    def test_subsampling_state_is_cleared(self):
        df = _frame(60, 5, seed=7)
        dt = DataTypical(shapley_mode=True, shapley_n_permutations=4,
                         shapley_compute_formative=False, shapley_top_n=6, **KW)
        dt.fit_transform(df)
        assert hasattr(dt, "_union_core_samples")
        dt.shapley_top_n = None
        dt.fit_transform(_frame(25, 5, seed=8))
        assert not hasattr(dt, "_union_core_samples"), \
            "the previous fit's core-sample indices survived"

    def test_label_frame_is_cleared(self):
        labelled = _frame(30, 4, seed=9)
        labelled["outcome"] = np.arange(30) % 2
        dt = DataTypical(label_columns=["outcome"], **KW)
        dt.fit(labelled)
        assert dt.label_df_ is not None
        dt.label_columns = None
        dt.fit(_frame(30, 4, seed=10))
        assert dt.label_df_ is None


class TestCrossModalityRefit:
    """The dangerous case: tabular state surviving into a text fit."""

    def test_tabular_frame_does_not_survive_into_a_text_fit(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        dt.fit(_frame(30, 5, seed=11))
        dt.fit_text(CORPUS)
        assert dt._df_original_fit is None
        assert dt.feature_columns_ is None
        assert dt.keep_mask_ is None

    def test_the_visualisations_refuse_after_a_text_refit(self):
        """
        They guard on feature_columns_ being None. With stale tabular values
        they would have proceeded, plotting the wrong data.
        """
        from datatypical_viz import heatmap, profile_plot

        dt = DataTypical(shapley_mode=True, shapley_n_permutations=4,
                         shapley_compute_formative=False,
                         archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        dt.fit_transform(_frame(30, 5, seed=12))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_text(CORPUS)
        with pytest.raises(RuntimeError, match="requires a tabular"):
            heatmap(dt, samples=[0, 1])
        with pytest.raises(RuntimeError, match="requires a tabular"):
            profile_plot(dt, 0)

    def test_text_vectorizer_does_not_survive_into_a_tabular_fit(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        dt.fit_text(CORPUS)
        assert dt.vectorizer_ is not None
        dt.fit(_frame(30, 5, seed=13))
        assert dt.vectorizer_ is None

    def test_transform_text_refuses_after_a_tabular_refit(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        dt.fit_text(CORPUS)
        dt.fit(_frame(30, 5, seed=14))
        with pytest.raises(RuntimeError, match="Call fit_text first"):
            dt.transform_text(CORPUS)

    def test_graph_topology_survives_its_own_fit(self):
        """fit() clears state, so fit_transform_graph must publish after it."""
        rng = np.random.default_rng(15)
        features = pd.DataFrame(rng.normal(size=(8, 3)), columns=list("xyz"))
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                          [6, 7], [7, 0]])
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform_graph(features, edges=edges)
        assert dt.graph_topology_df_ is not None
        assert len(dt.graph_topology_df_) == 8

    def test_graph_topology_is_cleared_by_a_later_tabular_fit(self):
        rng = np.random.default_rng(16)
        features = pd.DataFrame(rng.normal(size=(8, 3)), columns=list("xyz"))
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform_graph(features, edges=edges)
        dt.fit(_frame(30, 4, seed=17))
        assert dt.graph_topology_df_ is None


class TestPresetsAreReconsidered:
    def test_switching_fast_mode_reapplies_the_presets(self):
        df = _frame(30, 5, seed=18)
        dt = DataTypical(nmf_rank=3, n_prototypes=5, random_state=0,
                         fast_mode=True)
        dt.fit(df)
        assert dt.archetypal_method == "nmf"
        dt.fast_mode = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df)
        assert dt.archetypal_method == "aa", \
            "the fast-mode preset survived a switch to publication mode"

    def test_an_explicit_choice_between_fits_is_not_overwritten(self):
        """Only values the presets filled in themselves may be reconsidered."""
        df = _frame(30, 5, seed=19)
        dt = DataTypical(nmf_rank=3, n_prototypes=5, random_state=0,
                         fast_mode=True)
        dt.fit(df)
        assert dt.archetypal_method == "nmf"     # filled in by the preset
        dt.archetypal_method = "auto"            # now an explicit choice
        dt.fast_mode = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df)
        assert dt.archetypal_method == "auto", (
            "an explicitly set archetypal_method was overwritten by the presets"
        )

    def test_reassigning_the_presets_own_value_is_indistinguishable(self):
        """
        A known and accepted limit. The presets remember the value they filled
        in and release it only while it still holds exactly that, so a caller
        who assigns the same value the preset chose cannot be told apart from
        the preset having chosen it. Assign a different value, or pass it to
        the constructor, to make a choice stick.
        """
        df = _frame(30, 5, seed=19)
        dt = DataTypical(nmf_rank=3, n_prototypes=5, random_state=0,
                         fast_mode=True)
        dt.fit(df)
        dt.archetypal_method = "nmf"     # the same value the preset chose
        dt.fast_mode = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df)
        assert dt.archetypal_method == "aa"   # reconsidered, not preserved

    def test_an_explicit_constructor_choice_is_never_touched(self):
        df = _frame(30, 5, seed=20)
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=5,
                         random_state=0, fast_mode=False)
        dt.fit(df)
        assert dt.archetypal_method == "nmf"
        dt.fit(df)
        assert dt.archetypal_method == "nmf"

    def test_permutations_preset_is_reconsidered(self):
        df = _frame(25, 4, seed=21)
        dt = DataTypical(nmf_rank=3, n_prototypes=4, random_state=0,
                         fast_mode=True, shapley_mode=True)
        dt.fit(df)
        assert dt.shapley_n_permutations == 30
        dt.fast_mode = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df)
        assert dt.shapley_n_permutations == 100


class TestPickleRoundTrip:
    def test_a_fitted_estimator_transforms_identically_after_pickling(self):
        df = _frame(40, 5, seed=22)
        dt = DataTypical(shapley_mode=True, shapley_n_permutations=4,
                         shapley_compute_formative=True, **KW)
        before = dt.fit_transform(df)
        after = pickle.loads(pickle.dumps(dt)).transform(df)
        for col in RANKS:
            np.testing.assert_allclose(before[col].to_numpy(),
                                       after[col].to_numpy(), atol=1e-9)

    def test_the_backend_and_settings_survive_pickling(self):
        dt = DataTypical(**KW)
        dt.fit(_frame(30, 5, seed=23))
        restored = pickle.loads(pickle.dumps(dt))
        assert restored.archetypal_backend_ == dt.archetypal_backend_
        assert restored.settings_ == dt.settings_

    def test_explanations_survive_pickling(self):
        df = _frame(30, 5, seed=24)
        dt = DataTypical(shapley_mode=True, shapley_n_permutations=4,
                         shapley_compute_formative=True, **KW)
        dt.fit_transform(df)
        restored = pickle.loads(pickle.dumps(dt))
        original = dt.get_shapley_explanations(df.index[0])
        recovered = restored.get_shapley_explanations(df.index[0])
        for key in original:
            np.testing.assert_allclose(original[key], recovered[key])

    def test_registered_ideals_survive_pickling(self):
        df = _frame(30, 5, seed=25)
        dt = DataTypical(**KW)
        dt.fit(df)
        dim = dt.H_.shape[1]
        dt.register_ideal("target", np.ones(dim))
        restored = pickle.loads(pickle.dumps(dt))
        np.testing.assert_allclose(restored.ideals_["target"], np.ones(dim))
