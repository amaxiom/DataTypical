"""
Tests for datatypical_viz.

Every test runs under the Agg backend (set in conftest) and asserts on the
returned Axes rather than on pixels, so nothing here depends on a display or on
exact rendering.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from datatypical import DataTypical
from datatypical_viz import get_top_sample, heatmap, profile_plot, significance_plot


# ---------------------------------------------------------------------------
# Fixtures: one full Shapley fit shared across the module, and a fast_mode fit
# for the "formative data not available" paths.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def viz_fit():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(size=(30, 6)), columns=["f%d" % i for i in range(6)]
    )
    df["Age"] = rng.integers(30, 85, size=30).astype(float)
    df["group"] = np.take(["a", "b"], np.arange(30) % 2)
    df["score"] = rng.normal(size=30)

    dt = DataTypical(
        shapley_mode=True,
        shapley_n_permutations=6,
        shapley_compute_formative=True,
        archetypal_method="nmf",
        nmf_rank=3,
        n_prototypes=5,
        stereotype_column="Age",
        stereotype_target=55,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = dt.fit_transform(df)
    return dt, results


@pytest.fixture(scope="module")
def fast_fit():
    """fast_mode skips formative, so the *_shapley_rank columns are empty."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        rng.normal(size=(20, 5)), columns=["f%d" % i for i in range(5)]
    )
    dt = DataTypical(
        fast_mode=True,
        shapley_mode=True,
        shapley_n_permutations=4,
        nmf_rank=2,
        n_prototypes=4,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = dt.fit_transform(df)
    return dt, results


# ---------------------------------------------------------------------------
# get_top_sample
# ---------------------------------------------------------------------------
class TestGetTopSample:
    def test_returns_a_single_index(self, viz_fit):
        _, results = viz_fit
        top = get_top_sample(results, "archetypal_rank")
        assert top == results["archetypal_rank"].idxmax()

    def test_returns_a_list_when_n_above_one(self, viz_fit):
        _, results = viz_fit
        top = get_top_sample(results, "archetypal_rank", n=3)
        assert isinstance(top, list)
        assert len(top) == 3

    def test_min_mode(self, viz_fit):
        _, results = viz_fit
        assert get_top_sample(results, "archetypal_rank", mode="min") == results[
            "archetypal_rank"
        ].idxmin()

    def test_min_mode_with_several(self, viz_fit):
        _, results = viz_fit
        assert len(get_top_sample(results, "archetypal_rank", n=4, mode="min")) == 4

    def test_missing_column_returns_none(self, viz_fit, capsys):
        _, results = viz_fit
        assert get_top_sample(results, "not_a_column") is None
        assert "not found" in capsys.readouterr().out

    def test_all_nan_column_returns_none(self, fast_fit, capsys):
        _, results = fast_fit
        assert get_top_sample(results, "archetypal_shapley_rank") is None
        out = capsys.readouterr().out
        assert "no data" in out
        assert "shapley_mode=True" in out

    def test_stereotypical_hint(self, capsys):
        results = pd.DataFrame({"stereotypical_shapley_rank": [np.nan, np.nan]})
        assert get_top_sample(results, "stereotypical_shapley_rank") is None
        assert "stereotype_column" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# significance_plot
# ---------------------------------------------------------------------------
class TestSignificancePlot:
    @pytest.mark.parametrize(
        "significance", ["archetypal", "prototypical", "stereotypical"]
    )
    def test_each_significance(self, viz_fit, significance):
        _, results = viz_fit
        ax = significance_plot(results, significance=significance)
        assert ax is not None
        assert "Actual" in ax.get_xlabel()

    def test_rejects_an_unknown_significance(self, viz_fit):
        _, results = viz_fit
        with pytest.raises(ValueError, match="significance must be one of"):
            significance_plot(results, significance="nonsense")

    def test_missing_actual_column(self):
        with pytest.raises(ValueError, match="not in results"):
            significance_plot(pd.DataFrame({"x": [1.0]}), significance="archetypal")

    def test_missing_formative_column(self):
        results = pd.DataFrame({"archetypal_rank": [0.1, 0.9]})
        with pytest.raises(ValueError, match="archetypal_shapley_rank"):
            significance_plot(results)

    def test_no_formative_data_returns_a_placeholder(self, fast_fit, capsys):
        _, results = fast_fit
        ax = significance_plot(results, significance="archetypal")
        assert ax is not None
        assert "not available" in capsys.readouterr().out

    def test_no_formative_data_with_a_supplied_axis(self, fast_fit):
        _, results = fast_fit
        fig, ax = plt.subplots()
        assert significance_plot(results, ax=ax) is ax

    def test_no_formative_data_with_a_title(self, fast_fit):
        _, results = fast_fit
        ax = significance_plot(results, title="Custom")
        assert ax.get_title() == "Custom"

    def test_binary_colour_mapping(self, viz_fit):
        _, results = viz_fit
        ax = significance_plot(results, color_by="group")
        assert ax.get_legend() is not None

    def test_multiclass_discrete_colour_mapping(self, viz_fit):
        _, results = viz_fit
        df = results.copy()
        df["cls"] = np.arange(len(df)) % 4
        ax = significance_plot(df, color_by="cls")
        assert ax.get_legend() is not None

    def test_marker_range_of_categories(self, viz_fit):
        """6 to 12 categories get distinct markers as well as colours."""
        _, results = viz_fit
        df = results.copy()
        df["cls"] = np.arange(len(df)) % 8
        ax = significance_plot(df, color_by="cls")
        assert ax.get_legend() is not None

    def test_continuous_colour_mapping(self, viz_fit):
        _, results = viz_fit
        ax = significance_plot(results, color_by="score")
        assert ax.get_legend() is None

    def test_unknown_color_by_column(self, viz_fit):
        _, results = viz_fit
        with pytest.raises(ValueError, match="color_by column"):
            significance_plot(results, color_by="nope")

    def test_colour_column_with_no_valid_values(self, viz_fit):
        _, results = viz_fit
        df = results.copy()
        df["empty"] = pd.Series([None] * len(df), dtype=object)
        with pytest.raises(ValueError, match="no valid values"):
            significance_plot(df, color_by="empty")

    def test_size_by(self, viz_fit):
        _, results = viz_fit
        assert significance_plot(results, size_by="score") is not None

    def test_size_by_constant_column(self, viz_fit):
        _, results = viz_fit
        df = results.copy()
        df["flat"] = 1.0
        assert significance_plot(df, size_by="flat") is not None

    def test_size_by_with_discrete_colour(self, viz_fit):
        _, results = viz_fit
        assert significance_plot(results, color_by="group", size_by="score") is not None

    def test_unknown_size_by_column(self, viz_fit):
        _, results = viz_fit
        with pytest.raises(ValueError, match="size_by column"):
            significance_plot(results, size_by="nope")

    def test_quadrant_lines_can_be_turned_off(self, viz_fit):
        _, results = viz_fit
        ax = significance_plot(results, quadrant_lines=False)
        assert ax is not None

    def test_custom_quadrant_threshold(self, viz_fit):
        _, results = viz_fit
        ax = significance_plot(results, quadrant_threshold=(0.3, 0.7))
        assert ax is not None

    def test_label_top(self, viz_fit):
        _, results = viz_fit
        labels = {idx: "s%d" % i for i, idx in enumerate(results.index)}
        ax = significance_plot(results, labels=labels, label_top=3)
        assert len(ax.texts) >= 3

    def test_label_top_without_labels_is_ignored(self, viz_fit):
        _, results = viz_fit
        ax = significance_plot(results, label_top=3)
        assert ax is not None

    def test_custom_title(self, viz_fit):
        _, results = viz_fit
        assert significance_plot(results, title="My Title").get_title() == "My Title"

    def test_default_title(self, viz_fit):
        _, results = viz_fit
        assert "Dual-Perspective" in significance_plot(results).get_title()

    def test_supplied_axis_is_used(self, viz_fit):
        _, results = viz_fit
        fig, ax = plt.subplots()
        assert significance_plot(results, ax=ax) is ax

    def test_custom_figsize(self, viz_fit):
        _, results = viz_fit
        assert significance_plot(results, figsize=(8, 6)) is not None


# ---------------------------------------------------------------------------
# heatmap
# ---------------------------------------------------------------------------
class TestHeatmap:
    def test_basic(self, viz_fit):
        dt, results = viz_fit
        ax = heatmap(dt, results=results, top_n=5)
        assert ax is not None
        assert "Features" in ax.get_xlabel()

    @pytest.mark.parametrize(
        "significance", ["archetypal", "prototypical", "stereotypical"]
    )
    def test_each_significance(self, viz_fit, significance):
        dt, results = viz_fit
        assert heatmap(dt, results=results, significance=significance, top_n=4) is not None

    def test_requires_shapley_mode(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit(df_tiny)
        with pytest.raises(RuntimeError, match="Shapley mode not enabled"):
            heatmap(dt)

    def test_unknown_significance(self, viz_fit):
        dt, results = viz_fit
        with pytest.raises(ValueError, match="Unknown significance"):
            heatmap(dt, results=results, significance="nonsense")

    def test_missing_explanations_returns_a_placeholder(self, viz_fit, capsys):
        """selected_significance can leave a Phi matrix unset."""
        dt, results = viz_fit
        original = dt.Phi_stereotypical_explanations_
        dt.Phi_stereotypical_explanations_ = None
        try:
            ax = heatmap(dt, results=results, significance="stereotypical")
            assert ax is not None
            assert "not available" in capsys.readouterr().out
        finally:
            dt.Phi_stereotypical_explanations_ = original

    def test_missing_explanations_for_a_non_stereotypical_type(self, viz_fit, capsys):
        dt, results = viz_fit
        original = dt.Phi_archetypal_explanations_
        dt.Phi_archetypal_explanations_ = None
        try:
            heatmap(dt, results=results, significance="archetypal")
            assert "shapley_mode=True" in capsys.readouterr().out
        finally:
            dt.Phi_archetypal_explanations_ = original

    def test_missing_explanations_with_a_supplied_axis(self, viz_fit):
        dt, results = viz_fit
        original = dt.Phi_archetypal_explanations_
        dt.Phi_archetypal_explanations_ = None
        try:
            fig, ax = plt.subplots()
            assert heatmap(dt, results=results, ax=ax) is ax
        finally:
            dt.Phi_archetypal_explanations_ = original

    def test_needs_samples_or_results(self, viz_fit):
        dt, _ = viz_fit
        with pytest.raises(ValueError, match="Must provide either"):
            heatmap(dt)

    def test_explicit_sample_list(self, viz_fit):
        dt, _ = viz_fit
        assert heatmap(dt, samples=[0, 1, 2]) is not None

    def test_order_formative(self, viz_fit):
        dt, results = viz_fit
        assert heatmap(dt, results=results, order="formative", top_n=4) is not None

    def test_order_formative_falls_back_when_empty(self, fast_fit, capsys):
        dt, results = fast_fit
        heatmap(dt, results=results, order="formative", top_n=3)
        assert "Falling back" in capsys.readouterr().out

    def test_invalid_order(self, viz_fit):
        dt, results = viz_fit
        with pytest.raises(ValueError, match="order must be"):
            heatmap(dt, results=results, order="sideways")

    def test_top_features(self, viz_fit):
        dt, results = viz_fit
        ax = heatmap(dt, results=results, top_n=4, top_features=3)
        assert len(ax.get_xticklabels()) == 3

    def test_top_n_defaults_from_the_fit(self, viz_fit):
        dt, results = viz_fit
        assert heatmap(dt, results=results) is not None

    def test_top_n_from_a_fractional_shapley_top_n(self, viz_fit):
        dt, results = viz_fit
        original = dt.shapley_top_n
        dt.shapley_top_n = 0.5
        try:
            assert heatmap(dt, results=results) is not None
        finally:
            dt.shapley_top_n = original

    def test_top_n_from_an_integer_shapley_top_n(self, viz_fit):
        dt, results = viz_fit
        original = dt.shapley_top_n
        dt.shapley_top_n = 5
        try:
            assert heatmap(dt, results=results) is not None
        finally:
            dt.shapley_top_n = original

    def test_custom_title(self, viz_fit):
        dt, results = viz_fit
        assert heatmap(dt, results=results, top_n=3, title="H").get_title() == "H"

    def test_default_title(self, viz_fit):
        dt, results = viz_fit
        assert "Explanations" in heatmap(dt, results=results, top_n=3).get_title()

    def test_supplied_axis(self, viz_fit):
        dt, results = viz_fit
        fig, ax = plt.subplots()
        assert heatmap(dt, results=results, top_n=3, ax=ax) is ax

    def test_requires_a_tabular_fit(self, corpus):
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, nmf_rank=2,
            n_prototypes=3, archetypal_method="nmf", random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform_text(corpus)
        with pytest.raises(RuntimeError, match="requires a tabular"):
            heatmap(dt, samples=[0, 1])


# ---------------------------------------------------------------------------
# profile_plot
# ---------------------------------------------------------------------------
class TestProfilePlot:
    def test_basic(self, viz_fit):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        ax = profile_plot(dt, top)
        assert ax is not None

    @pytest.mark.parametrize(
        "significance", ["archetypal", "prototypical", "stereotypical"]
    )
    def test_each_significance(self, viz_fit, significance):
        dt, results = viz_fit
        top = results["%s_rank" % significance].idxmax()
        assert profile_plot(dt, top, significance=significance) is not None

    @pytest.mark.parametrize("order", ["local", "global"])
    def test_both_orderings(self, viz_fit, order):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        assert profile_plot(dt, top, order=order) is not None

    def test_requires_shapley_mode(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit(df_tiny)
        with pytest.raises(RuntimeError, match="Shapley mode not enabled"):
            profile_plot(dt, 0)

    def test_invalid_significance(self, viz_fit):
        dt, results = viz_fit
        with pytest.raises(ValueError, match="significance must be one of"):
            profile_plot(dt, results.index[0], significance="nonsense")

    def test_invalid_order(self, viz_fit):
        dt, results = viz_fit
        with pytest.raises(ValueError, match="order must be one of"):
            profile_plot(dt, results.index[0], order="sideways")

    def test_top_features(self, viz_fit):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        assert profile_plot(dt, top, top_features=3) is not None

    def test_top_features_must_be_at_least_one(self, viz_fit):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        with pytest.raises(ValueError, match="top_features must be"):
            profile_plot(dt, top, top_features=0)

    def test_top_features_larger_than_the_feature_count(self, viz_fit):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        assert profile_plot(dt, top, top_features=999) is not None

    def test_missing_explanations_are_reported(self, viz_fit):
        """Clearing a Phi matrix drops the key from get_shapley_explanations."""
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        original = dt.Phi_archetypal_explanations_
        dt.Phi_archetypal_explanations_ = None
        try:
            with pytest.raises(RuntimeError, match="No explanations available"):
                profile_plot(dt, top, order="global")
        finally:
            dt.Phi_archetypal_explanations_ = original

    def test_global_order_when_only_the_global_matrix_is_missing(
        self, viz_fit, monkeypatch
    ):
        """The guard fires when the explanations dict and the Phi matrix disagree."""
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        monkeypatch.setattr(
            dt,
            "get_shapley_explanations",
            lambda idx: {"archetypal": np.zeros(len(dt.feature_columns_))},
        )
        monkeypatch.setattr(dt, "Phi_archetypal_explanations_", None)
        with pytest.raises(RuntimeError, match="Global ordering requires"):
            profile_plot(dt, top, order="global")

    def test_custom_title(self, viz_fit):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        assert profile_plot(dt, top, title="P").get_title() == "P"

    def test_supplied_axis(self, viz_fit):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        fig, ax = plt.subplots(figsize=(10, 4))
        assert profile_plot(dt, top, ax=ax) is ax

    def test_custom_figsize(self, viz_fit):
        dt, results = viz_fit
        top = results["archetypal_rank"].idxmax()
        assert profile_plot(dt, top, figsize=(10, 4)) is not None

    def test_requires_a_tabular_fit(self, corpus):
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, nmf_rank=2,
            n_prototypes=3, archetypal_method="nmf", random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform_text(corpus)
        with pytest.raises(RuntimeError, match="requires a tabular"):
            profile_plot(dt, 0)

    def test_constant_feature_does_not_divide_by_zero(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
        df["flat"] = 2.0
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, nmf_rank=2,
            n_prototypes=4, archetypal_method="nmf", random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = dt.fit_transform(df)
        top = results["archetypal_rank"].idxmax()
        assert profile_plot(dt, top) is not None
