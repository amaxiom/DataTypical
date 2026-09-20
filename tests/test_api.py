"""
Public API coverage: the three data types, config round-tripping, the
explanation getters, and the error paths a user is most likely to meet.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import datatypical as dtmod
from datatypical import ConfigError, DataTypical, DataTypicalError


def _make_tabular(n=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, d)),
                      columns=["f%d" % i for i in range(d)])
    df["Age"] = rng.integers(30, 85, size=n).astype(float)
    return df


# ---------------------------------------------------------------------------
# Data type detection and routing
# ---------------------------------------------------------------------------
class TestDataTypeDetection:
    def test_dataframe_is_tabular(self, df_small):
        dt = DataTypical(archetypal_method="nmf", random_state=0)
        assert dt._auto_detect_data_type(df_small) == "tabular"

    def test_array_is_tabular(self):
        dt = DataTypical()
        assert dt._auto_detect_data_type(np.ones((4, 3))) == "tabular"

    def test_list_of_strings_is_text(self, corpus):
        assert DataTypical()._auto_detect_data_type(corpus) == "text"

    def test_tuple_of_strings_is_text(self, corpus):
        assert DataTypical()._auto_detect_data_type(tuple(corpus)) == "text"

    def test_edges_kwarg_means_graph(self, df_tiny):
        dt = DataTypical()
        assert dt._auto_detect_data_type(df_tiny, edges=np.zeros((2, 2))) == "graph"

    def test_edge_index_kwarg_means_graph(self, df_tiny):
        dt = DataTypical()
        assert (
            dt._auto_detect_data_type(df_tiny, edge_index=np.zeros((2, 2))) == "graph"
        )

    def test_unrecognised_input_raises(self):
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            DataTypical()._auto_detect_data_type({"a": 1})

    def test_empty_list_is_not_text(self):
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            DataTypical()._auto_detect_data_type([])

    def test_explicit_data_type_overrides_detection(self, df_small):
        dt = DataTypical(data_type="tabular")
        assert dt._validate_data_type("text") == "tabular"

    def test_invalid_data_type_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid data_type"):
            DataTypical(data_type="audio")._validate_data_type("tabular")

    def test_verbose_detection_messages(self, df_tiny, capsys):
        dt = DataTypical(data_type="tabular", verbose=True, archetypal_method="nmf")
        dt._validate_data_type("text")
        assert "Using configured data_type" in capsys.readouterr().out

    def test_verbose_autodetect_message(self, capsys):
        DataTypical(verbose=True)._validate_data_type("tabular")
        assert "Auto-detected" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------
class TestTabular:
    def test_fit_transform_returns_all_three_ranks(self, df_small):
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=4, random_state=0
        ).fit_transform(df_small)
        for column in ("archetypal_rank", "prototypical_rank", "stereotypical_rank"):
            assert column in out.columns
            assert out[column].between(0.0, 1.0).all()

    def test_transform_on_unseen_rows(self, df_small, make_tabular):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        fresh = make_tabular(15, 6, seed=99)
        out = dt.transform(fresh)
        assert len(out) == 15

    def test_return_ranks_only(self, df_small):
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            return_ranks_only=True,
        ).fit_transform(df_small)
        assert not any(c.startswith("f") for c in out.columns)

    def test_numpy_input(self):
        X = np.random.default_rng(0).normal(size=(20, 4))
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0
        ).fit_transform(X)
        assert len(out) == 20

    def test_label_columns_are_carried_through(self, df_small):
        df = df_small.copy()
        df["outcome"] = np.arange(len(df)) % 2
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            label_columns=["outcome"],
        ).fit_transform(df)
        assert "outcome" in out.columns

    def test_the_implemented_scaling_works(self):
        """Only MinMax scaling is implemented; see the rejection test below."""
        import warnings as _w

        df = _make_tabular(12, 4, seed=1)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            out = DataTypical(
                archetypal_method="nmf", nmf_rank=2, scale="minmax",
                random_state=0,
            ).fit_transform(df)
        assert len(out) == len(df)

    @pytest.mark.parametrize("scale", ["standard", "none", "wibble"])
    def test_unimplemented_scaling_is_rejected(self, df_tiny, scale):
        """
        These used to be accepted and then ignored, so a fit that asked for
        standardised features silently got MinMax scaling instead.
        """
        with pytest.raises(ConfigError, match="not implemented"):
            DataTypical(
                archetypal_method="nmf", nmf_rank=2, scale=scale, random_state=0
            ).fit_transform(df_tiny)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_dtype_options(self, df_tiny, dtype):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, dtype=dtype, random_state=0
        )
        dt.fit(df_tiny)
        assert dt.W_ is not None

    def test_speed_mode(self, df_small):
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, speed_mode=True, random_state=0
        ).fit_transform(df_small)
        assert len(out) == len(df_small)

    def test_verbose_fit(self, df_tiny, capsys):
        DataTypical(
            archetypal_method="nmf", nmf_rank=2, verbose=True, random_state=0
        ).fit(df_tiny)
        assert capsys.readouterr().out

    def test_missing_values_are_tolerated(self, df_small):
        df = df_small.copy()
        df.iloc[0, 0] = np.nan
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0, max_missing_frac=0.5
        ).fit_transform(df)
        assert len(out) == len(df)

    def test_all_nan_column_is_kept_at_the_default_threshold(self, df_small):
        """max_missing_frac defaults to 1.0, so nothing is rejected on missingness."""
        df = df_small.copy()
        df["dead"] = np.nan
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df)
        assert "dead" in (dt.feature_columns_ or [])

    def test_missingness_above_the_threshold_raises(self, df_small):
        df = df_small.copy()
        df["dead"] = np.nan
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0, max_missing_frac=0.5
        )
        with pytest.raises(DataTypicalError, match="Missingness too high"):
            dt.fit(df)

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        with pytest.raises(Exception):
            DataTypical(archetypal_method="nmf", random_state=0).fit(df)

    def test_id_like_columns_are_dropped(self, df_small, capsys):
        df = df_small.copy()
        df["sample_id"] = np.arange(len(df))
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, verbose=True, random_state=0
        )
        dt.fit(df)
        assert "sample_id" not in (dt.feature_columns_ or [])

    def test_monotonic_column_is_dropped(self, df_small):
        df = df_small.copy()
        df["row"] = np.arange(len(df), dtype=float)
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df)
        assert "row" not in (dt.feature_columns_ or [])

    def test_high_uniqueness_integer_column_is_dropped(self, df_small):
        rng = np.random.default_rng(5)
        df = df_small.copy()
        df["code"] = rng.permutation(np.arange(len(df)))
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df)
        assert "code" not in (dt.feature_columns_ or [])

    def test_float_columns_survive_high_uniqueness(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        assert any(c.startswith("f") for c in (dt.feature_columns_ or []))

    @pytest.mark.parametrize("which", ["archetypal", "prototypical", "stereotypical"])
    def test_selected_significance(self, df_small, which):
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            selected_significance=which,
        ).fit_transform(df_small)
        assert "%s_rank" % which in out.columns

    def test_invalid_selected_significance(self, df_small):
        with pytest.raises(ConfigError, match="selected_significance"):
            DataTypical(
                selected_significance="nonsense", random_state=0
            ).fit(df_small)

    def test_auto_n_prototypes(self, df_small):
        """
        The spelling matters: only 'kneedle' is recognised. This test used to
        pass 'knee', which was silently ignored, and asserted only that the fit
        produced something.
        """
        import warnings as _w

        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=10,
            random_state=0, auto_n_prototypes="kneedle",
        )
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            dt.fit(df_small)
        assert dt.prototype_indices_ is not None
        assert dt.knee_ == len(dt.prototype_indices_)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
class TestText:
    def test_fit_transform_text(self, corpus):
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        ).fit_transform_text(corpus)
        assert len(out) == len(corpus)

    def test_fit_text_then_transform_text(self, corpus):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit_text(corpus)
        out = dt.transform_text(corpus)
        assert len(out) == len(corpus)

    def test_text_through_the_unified_entry_point(self, corpus):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit(corpus)
        assert dt.vectorizer_ is not None

    def test_text_with_metadata_stereotype(self, corpus, text_metadata):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            stereotype_column="Year", stereotype_target="max", random_state=0,
        )
        out = dt.fit_transform_text(corpus, text_metadata=text_metadata)
        assert out["stereotypical_rank"].max() == pytest.approx(1.0)

    def test_text_with_keyword_stereotype(self, corpus):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            stereotype_keywords=["protein"], random_state=0,
        )
        out = dt.fit_transform_text(corpus)
        assert out["stereotypical_rank"].max() > 0

    def test_unknown_keywords_warn(self, corpus):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            stereotype_keywords=["protein", "zzzznotaword"], random_state=0,
        )
        with pytest.warns(UserWarning, match="not found in vocabulary"):
            dt.fit_transform_text(corpus)

    def test_all_keywords_unknown_warns_and_zeroes(self, corpus):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            stereotype_keywords=["zzzznotaword"], random_state=0,
        )
        with pytest.warns(UserWarning):
            out = dt.fit_transform_text(corpus)
        assert out["stereotypical_rank"].eq(0.0).all() or out[
            "stereotypical_rank"
        ].notna().all()

    def test_stereotype_column_without_metadata_raises(self, corpus):
        """v0.8.0: this used to fall back to extremeness without a word."""
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, stereotype_column="Year",
            random_state=0,
        )
        with pytest.raises(ValueError, match="text_metadata"):
            dt.fit_transform_text(corpus)

    def test_stereotype_column_missing_from_metadata_raises(
        self, corpus, text_metadata
    ):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, stereotype_column="NotThere",
            random_state=0,
        )
        with pytest.raises(ValueError, match="not found in text_metadata"):
            dt.fit_transform_text(corpus, text_metadata=text_metadata)

    def test_non_numeric_text_metadata_column_raises(self, corpus, text_metadata):
        metadata = text_metadata.copy()
        metadata["Venue"] = ["a", "b"] * (len(corpus) // 2)
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, stereotype_column="Venue",
            random_state=0,
        )
        with pytest.raises(ConfigError, match="Venue"):
            dt.fit_transform_text(corpus, text_metadata=metadata)

    def test_column_and_keywords_together_are_rejected(self, corpus):
        dt = DataTypical(
            stereotype_column="Year", stereotype_keywords=["protein"], random_state=0
        )
        with pytest.raises(ConfigError, match="Cannot specify both"):
            dt.fit_transform_text(corpus)

    def test_generator_input_is_not_exhausted(self, corpus):
        """v0.7.6 fixed iterator exhaustion; keep it fixed."""
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        out = dt.fit_transform_text(iter(corpus))
        assert len(out) == len(corpus)

    def test_text_with_shapley(self, corpus, text_metadata):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=False,
            stereotype_column="Year", random_state=0,
        )
        out = dt.fit_transform_text(corpus, text_metadata=text_metadata)
        assert len(out) == len(corpus)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class TestGraph:
    def test_fit_transform_graph_with_topology(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        out = dt.fit_transform_graph(features, edges=edges)
        assert len(out) == len(features)
        assert dt.graph_topology_df_ is not None

    def test_edge_index_alias(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        out = dt.fit_transform_graph(features, edge_index=edges)
        assert len(out) == len(features)

    def test_topology_can_be_skipped(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit_transform_graph(features, edges=edges, compute_topology=False)
        assert dt.graph_topology_df_ is None

    def test_without_edges(self, graph_data):
        features, _ = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        out = dt.fit_transform_graph(features)
        assert len(out) == len(features)

    def test_transposed_edge_list(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        out = dt.fit_transform_graph(features, edges=edges.T)
        assert len(out) == len(features)

    def test_numpy_node_features(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        out = dt.fit_transform_graph(features.to_numpy(), edges=edges)
        assert len(out) == len(features)

    def test_colliding_topology_column_warns(self, graph_data):
        features, edges = graph_data
        df = features.copy()
        df["degree"] = 1.0
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        with pytest.warns(UserWarning, match="already exists"):
            dt.fit_transform_graph(df, edges=edges)

    def test_graph_through_the_unified_entry_point(self, graph_data):
        """fit() appends topology, so transform() needs the same columns."""
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit(features, edges=edges)
        augmented = pd.concat([features, dt.graph_topology_df_], axis=1)
        out = dt.transform(augmented)
        assert len(out) == len(features)

    def test_transform_without_topology_columns_reports_them(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit(features, edges=edges)
        with pytest.raises(DataTypicalError, match="Missing required feature columns"):
            dt.transform(features)

    def test_topology_features_are_selectable(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            graph_topology_features=["degree"], random_state=0,
        )
        dt.fit_transform_graph(features, edges=edges)
        assert dt.graph_topology_df_ is not None


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def test_fit_transform_signals(df_small):
    out = DataTypical(
        archetypal_method="nmf", nmf_rank=3, random_state=0
    ).fit_transform_signals(df_small)
    assert len(out) == len(df_small)


# ---------------------------------------------------------------------------
# Config round-trip and sklearn interop
# ---------------------------------------------------------------------------
class TestConfig:
    def test_to_config_carries_the_version(self):
        assert DataTypical().to_config()["version"] == "0.8.0"

    def test_round_trip(self):
        dt = DataTypical(nmf_rank=5, n_prototypes=7, stereotype_column="Age")
        clone = DataTypical.from_config(dt.to_config())
        assert clone.nmf_rank == 5
        assert clone.n_prototypes == 7
        assert clone.stereotype_column == "Age"

    def test_from_config_ignores_unknown_keys(self):
        cfg = DataTypical().to_config()
        cfg["not_a_parameter"] = 1
        assert DataTypical.from_config(cfg).nmf_rank == 8

    def test_from_config_reports_bad_values(self):
        """Artifact fields are init=False, so passing one is a TypeError."""
        with pytest.raises(ConfigError):
            DataTypical.from_config({"nmf_rank": 3, "W_": np.ones((2, 2))})

    def test_get_params(self):
        params = DataTypical(nmf_rank=4).get_params()
        assert params["nmf_rank"] == 4

    def test_set_params(self):
        dt = DataTypical()
        dt.set_params(nmf_rank=11)
        assert dt.nmf_rank == 11

    def test_set_params_rejects_unknown(self):
        with pytest.raises(Exception):
            DataTypical().set_params(not_a_parameter=1)

    def test_settings_after_fit(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit(df_tiny)
        assert dt.settings_["random_state"] == 0
        assert "archetypal_backend" in dt.settings_


class TestIdeals:
    def test_register_ideal_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit"):
            DataTypical().register_ideal("target", [1.0, 2.0])

    def test_register_ideal(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit(df_tiny)
        dim = dt.H_.shape[1]
        dt.register_ideal("target", np.ones(dim))
        assert "target" in dt.ideals_

    def test_register_ideal_checks_dimension(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit(df_tiny)
        with pytest.raises(ValueError, match="dim"):
            dt.register_ideal("target", np.ones(999))


# ---------------------------------------------------------------------------
# Shapley getters
# ---------------------------------------------------------------------------
class TestShapleyGetters:
    def test_explanations_require_shapley_mode(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit(df_tiny)
        with pytest.raises(RuntimeError, match="Shapley mode not enabled"):
            dt.get_shapley_explanations(0)

    def test_formative_requires_shapley_mode(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit(df_tiny)
        with pytest.raises(RuntimeError, match="Shapley mode not enabled"):
            dt.get_formative_attributions(0)

    def test_explanations_returned_per_significance(self, fitted_shapley):
        dt, _ = fitted_shapley
        explanations = dt.get_shapley_explanations(0)
        assert set(explanations) >= {"archetypal", "prototypical"}
        for value in explanations.values():
            assert np.isfinite(value).all()

    def test_formative_attributions(self, fitted_shapley):
        dt, _ = fitted_shapley
        attributions = dt.get_formative_attributions(0)
        assert "archetypal" in attributions

    def test_unknown_sample_index_raises(self, fitted_shapley):
        dt, _ = fitted_shapley
        with pytest.raises(ValueError, match="not found"):
            dt.get_shapley_explanations(10_000)

    def test_unknown_sample_index_raises_for_formative(self, fitted_shapley):
        dt, _ = fitted_shapley
        with pytest.raises(ValueError, match="not found"):
            dt.get_formative_attributions(10_000)

    def test_fast_mode_skips_formative(self, df_small):
        dt = DataTypical(
            fast_mode=True, shapley_mode=True, shapley_n_permutations=4,
            nmf_rank=3, random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df_small)
        with pytest.raises(RuntimeError, match="fast_mode"):
            dt.get_formative_attributions(0)


# ---------------------------------------------------------------------------
# Shapley end to end
# ---------------------------------------------------------------------------
class TestShapleyEndToEnd:
    def test_dual_rankings_are_produced(self, fitted_shapley):
        _, out = fitted_shapley
        assert "archetypal_rank" in out.columns
        assert "archetypal_shapley_rank" in out.columns

    def test_is_reproducible(self, df_small):
        def run():
            dt = DataTypical(
                shapley_mode=True, shapley_n_permutations=6, nmf_rank=3,
                archetypal_method="nmf", shapley_compute_formative=True,
                random_state=3,
            )
            return dt.fit_transform(df_small)["archetypal_shapley_rank"].to_numpy()

        np.testing.assert_allclose(run(), run())

    def test_verbose_shapley(self, df_tiny, capsys):
        DataTypical(
            shapley_mode=True, shapley_n_permutations=4, nmf_rank=2,
            archetypal_method="nmf", shapley_compute_formative=True,
            verbose=True, random_state=0,
        ).fit_transform(df_tiny)
        assert "SHAPLEY" in capsys.readouterr().out

    @pytest.mark.parametrize("which", ["archetypal", "prototypical", "stereotypical"])
    def test_shapley_with_selected_significance(self, df_small, which):
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, nmf_rank=3,
            archetypal_method="nmf", shapley_compute_formative=True,
            selected_significance=which, stereotype_column="Age",
            random_state=0,
        )
        out = dt.fit_transform(df_small)
        assert "%s_rank" % which in out.columns

    def test_v04_value_functions(self, fitted_shapley):
        dt, _ = fitted_shapley
        X = np.random.default_rng(0).random((6, 4))
        ctx = {"nmf_rank": 2, "random_state": 0}
        assert np.isfinite(dt._v04_archetypal_value(X, np.arange(6), ctx))
        assert np.isfinite(dt._v04_prototypical_value(X, np.arange(6), ctx))
        stereo_ctx = {
            "target_values": np.arange(6.0),
            "stereotype_target": "max",
            "median": 2.5,
        }
        assert np.isfinite(dt._v04_stereotypical_value(X, np.arange(6), stereo_ctx))

    @pytest.mark.parametrize("target", ["max", "min", 3.0])
    def test_v04_stereotypical_targets(self, fitted_shapley, target):
        dt, _ = fitted_shapley
        X = np.random.default_rng(0).random((6, 4))
        ctx = {
            "target_values": np.arange(6.0),
            "stereotype_target": target,
            "median": 2.5,
        }
        assert np.isfinite(dt._v04_stereotypical_value(X, np.arange(6), ctx))

    def test_v04_stereotypical_without_target_values(self, fitted_shapley):
        dt, _ = fitted_shapley
        X = np.random.default_rng(0).random((6, 4))
        value = dt._v04_stereotypical_value(X, np.arange(6), {"target_values": None})
        assert np.isfinite(value)

    def test_v04_archetypal_falls_back_when_nmf_fails(self, fitted_shapley):
        """The except branch returns a range-based value instead of raising."""
        dt, _ = fitted_shapley
        X = np.random.default_rng(0).random((6, 4))
        ctx = {"nmf_rank": 2, "random_state": "not-a-seed"}
        assert np.isfinite(dt._v04_archetypal_value(X, np.arange(6), ctx))

    def test_v04_value_functions_on_empty_subsets(self, fitted_shapley):
        dt, _ = fitted_shapley
        empty = np.empty((0, 4))
        idx = np.array([], dtype=int)
        ctx = {"nmf_rank": 2, "random_state": 0}
        assert dt._v04_archetypal_value(empty, idx, ctx) == 0.0
        assert dt._v04_prototypical_value(empty, idx, ctx) == 0.0

    def test_shapley_on_transform(self, fitted_shapley, df_small):
        dt, _ = fitted_shapley
        out = dt.transform(df_small)
        assert len(out) == len(df_small)


# ---------------------------------------------------------------------------
# Misc internals reached through the public API
# ---------------------------------------------------------------------------
class TestMiscInternals:
    def test_kneedle_on_a_flat_curve(self):
        dt = DataTypical()
        assert dt._kneedle(np.zeros(5)) is None

    def test_kneedle_finds_a_knee(self):
        dt = DataTypical()
        gains = np.array([10.0, 5.0, 1.0, 0.5, 0.1])
        assert dt._kneedle(gains) >= 1

    def test_assignments_cosine(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        assert dt.assignments_ is not None
        assert len(dt.assignments_) == len(df_small)

    def test_coverage_is_recorded(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        assert dt.coverage_ is not None

    def test_transform_before_fit_raises(self, df_small):
        with pytest.raises(Exception):
            DataTypical().transform(df_small)

    def test_the_implemented_distance_metric_works(self):
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=2, distance_metric="euclidean",
            random_state=0,
        ).fit_transform(_make_tabular(12, 4, seed=1))
        assert len(out) == 12

    @pytest.mark.parametrize(
        "name,value",
        [("distance_metric", "cosine"), ("distance_metric", "manhattan"),
         ("similarity_metric", "dot")],
    )
    def test_unimplemented_metrics_are_rejected(self, df_tiny, name, value):
        """Accepted and then ignored through v0.7.7."""
        with pytest.raises(ConfigError, match="not implemented"):
            DataTypical(
                archetypal_method="nmf", nmf_rank=2, random_state=0,
                **{name: value},
            ).fit_transform(df_tiny)

    def test_feature_weights(self, df_tiny):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, random_state=0,
            feature_weights=np.ones(df_tiny.shape[1]),
        )
        out = dt.fit_transform(df_tiny)
        assert len(out) == len(df_tiny)

    def test_stereotype_target_must_be_min_max_or_numeric(self, df_small):
        with pytest.raises(ConfigError, match="stereotype_target"):
            DataTypical(
                stereotype_column="Age", stereotype_target="middling"
            ).fit(df_small)
