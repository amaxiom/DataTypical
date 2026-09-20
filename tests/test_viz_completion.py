"""
The remaining datatypical_viz branches: the diagnostics that only fire when
explanations are missing for the requested samples, and the title and label
fallbacks.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from datatypical import DataTypical
from datatypical_viz import heatmap, profile_plot


@pytest.fixture(scope="module")
def shapley_fit():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(size=(24, 5)), columns=list("abcde"))
    df["Age"] = rng.integers(30, 85, size=24).astype(float)
    dt = DataTypical(
        shapley_mode=True,
        shapley_n_permutations=4,
        shapley_compute_formative=True,
        archetypal_method="nmf",
        nmf_rank=2,
        n_prototypes=4,
        stereotype_column="Age",
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = dt.fit_transform(df)
    return dt, results


class TestHeatmapDiagnostics:
    def test_missing_rank_column_raises(self, shapley_fit):
        dt, results = shapley_fit
        trimmed = results.drop(columns=["archetypal_rank"])
        with pytest.raises(RuntimeError, match="Cannot find archetypal_rank"):
            heatmap(dt, results=trimmed, top_n=3)

    def test_samples_without_explanations_are_skipped(self, shapley_fit, monkeypatch):
        """A sample outside the shapley_top_n subsample raises and is dropped."""
        dt, results = shapley_fit
        real = dt.get_shapley_explanations
        calls = {"n": 0}

        def flaky(idx):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise KeyError("no explanations for %r" % idx)
            return real(idx)

        monkeypatch.setattr(dt, "get_shapley_explanations", flaky)
        ax = heatmap(dt, results=results, top_n=4)
        assert ax is not None

    def test_no_explanations_at_all_returns_a_placeholder(
        self, shapley_fit, monkeypatch, capsys
    ):
        dt, results = shapley_fit

        def always_missing(idx):
            raise KeyError(idx)

        monkeypatch.setattr(dt, "get_shapley_explanations", always_missing)
        ax = heatmap(dt, results=results, top_n=3)
        assert ax is not None
        out = capsys.readouterr().out
        assert "None of the top" in out
        assert "shapley_top_n is too small" in out

    def test_no_explanations_with_a_supplied_axis(self, shapley_fit, monkeypatch):
        dt, results = shapley_fit
        monkeypatch.setattr(
            dt, "get_shapley_explanations", lambda idx: (_ for _ in ()).throw(KeyError(idx))
        )
        fig, ax = plt.subplots()
        assert heatmap(dt, results=results, top_n=3, ax=ax) is ax

    def test_zero_shapley_rows_are_reported_for_formative_ordering(
        self, shapley_fit, monkeypatch, capsys
    ):
        """Formative instances need not themselves be significant."""
        dt, results = shapley_fit
        n_features = dt.Phi_archetypal_explanations_.shape[1]

        def zeroed(idx):
            return {
                "archetypal": np.zeros(n_features),
                "prototypical": np.zeros(n_features),
                "stereotypical": np.zeros(n_features),
            }

        monkeypatch.setattr(dt, "get_shapley_explanations", zeroed)
        monkeypatch.setattr(dt, "shapley_top_n", None)
        heatmap(dt, results=results, order="formative", top_n=3)
        out = capsys.readouterr().out
        assert "zero Shapley values" in out
        assert "CREATE structure" in out

    def test_sample_labels_without_a_training_index(self, shapley_fit, monkeypatch):
        dt, _ = shapley_fit
        monkeypatch.setattr(dt, "train_index_", None)
        ax = heatmap(dt, samples=[0, 1, 2])
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert any(label.startswith("Sample ") for label in labels)


class TestProfilePlotRemainder:
    @pytest.mark.parametrize("significance", ["prototypical", "stereotypical"])
    def test_global_ordering_for_each_significance(self, shapley_fit, significance):
        dt, results = shapley_fit
        top = results["%s_rank" % significance].idxmax()
        assert profile_plot(dt, top, significance=significance, order="global") is not None

    def test_requires_the_original_frame(self, shapley_fit, monkeypatch):
        dt, results = shapley_fit
        top = results["archetypal_rank"].idxmax()
        monkeypatch.setattr(dt, "_df_original_fit", None)
        with pytest.raises(RuntimeError, match="requires original tabular data"):
            profile_plot(dt, top)

    def test_label_columns_are_consulted_for_the_title(self, shapley_fit):
        """label_df_ is read when building the default title."""
        rng = np.random.default_rng(6)
        df = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
        df["name"] = ["row%d" % i for i in range(20)]
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=False, archetypal_method="nmf",
            nmf_rank=2, n_prototypes=4, label_columns=["name"], random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = dt.fit_transform(df)
        assert dt.label_df_ is not None
        top = results["archetypal_rank"].idxmax()
        ax = profile_plot(dt, top)
        assert "Explanations" in ax.get_title()
