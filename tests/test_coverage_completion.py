"""
Targeted tests for the paths the main suites do not reach: import fallbacks,
verbose branches, chunked kernels, graph topology features, and the error
messages that only fire on a misconfigured fit.
"""
import builtins
import importlib.util
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import datatypical as dtmod
from datatypical import ConfigError, DataTypical, DataTypicalError, FacilityLocationSelector


# ---------------------------------------------------------------------------
# Module-level import fallbacks
#
# Each optional dependency has an except branch that sets a sentinel. Loading a
# private copy of the module with the dependency blocked exercises them without
# disturbing the datatypical already imported by the rest of the suite.
# ---------------------------------------------------------------------------
def _load_private_copy(name, blocked):
    """Import datatypical.py again, under `name`, with `blocked` unimportable."""
    real_import = builtins.__import__

    def fake_import(mod, *args, **kwargs):
        root = mod.split(".")[0]
        if root in blocked or mod in blocked:
            raise ImportError("blocked for testing: %s" % mod)
        return real_import(mod, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(name, dtmod.__file__)
    module = importlib.util.module_from_spec(spec)
    # Patching __import__ is enough: an `import x` statement always goes through
    # it, even when x is already in sys.modules. Removing entries from
    # sys.modules instead breaks the packages that are already loaded.
    #
    # The copy must be registered in sys.modules before it executes, because
    # @dataclass resolves annotations through sys.modules[cls.__module__].
    sys.modules[name] = module
    builtins.__import__ = fake_import
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = real_import
        sys.modules.pop(name, None)
    return module


class TestImportFallbacks:
    def test_without_numba(self):
        module = _load_private_copy("datatypical_no_numba", {"numba"})
        assert module.NUMBA_AVAILABLE is False
        # the stand-in decorator must leave the function callable
        decorated = module.jit(nopython=True)(lambda x: x + 1)
        assert decorated(1) == 2
        assert list(module.prange(3)) == [0, 1, 2]

    def test_without_scipy_sparse(self):
        module = _load_private_copy("datatypical_no_scipy", {"scipy"})
        assert module.sp is None
        assert module.ConvexHull is None
        assert module.cdist is None

    def test_without_py_pcha(self):
        module = _load_private_copy("datatypical_no_pcha", {"py_pcha"})
        assert module.PCHA is None


# ---------------------------------------------------------------------------
# Chunked distance path
# ---------------------------------------------------------------------------
class TestChunkedDistances:
    def test_large_problem_takes_the_chunked_branch(self):
        """n * m >= 100000 switches from the direct kernel to chunking."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 3))
        Y = rng.normal(size=(300, 3))
        chunked = dtmod._euclidean_min_to_set_dense(X, Y)
        direct = dtmod._euclidean_min_jit(X, Y)
        np.testing.assert_allclose(chunked, direct, rtol=1e-6)


# ---------------------------------------------------------------------------
# Facility location: everything forbidden
# ---------------------------------------------------------------------------
def test_facility_location_with_every_candidate_forbidden():
    X = dtmod._l2_normalize_rows_dense(np.random.default_rng(0).normal(size=(6, 3)))
    idx, gains = FacilityLocationSelector(n_prototypes=3).select(
        X, forbidden=set(range(6))
    )
    assert len(idx) == 0
    assert len(gains) == 0


# ---------------------------------------------------------------------------
# Shapley engine: verbose early stopping and the context-carrying generic path
# ---------------------------------------------------------------------------
class TestEngineBranches:
    def test_verbose_early_stop_sample_level(self, capsys):
        X = np.random.default_rng(0).normal(size=(6, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=400, random_state=0, verbose=True,
            early_stopping_patience=1, early_stopping_tolerance=1e9,
        )
        engine.compute_shapley_values(X, lambda a, b, c=None: 1.0, "const")
        assert "Early stop" in capsys.readouterr().out

    def test_verbose_early_stop_feature_level(self, capsys):
        X = np.random.default_rng(0).normal(size=(5, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=400, random_state=0, verbose=True,
            early_stopping_patience=1, early_stopping_tolerance=1e9,
        )
        engine.compute_feature_shapley_values(X, lambda a, b, c=None: 1.0, "const")
        assert "Early stop" in capsys.readouterr().out

    def test_generic_path_receives_the_context(self):
        seen = {}

        def value_fn(X_subset, indices, ctx):
            seen["ctx"] = ctx
            return float(len(X_subset))

        X = np.random.default_rng(0).normal(size=(5, 2))
        engine = dtmod.ShapleySignificanceEngine(n_permutations=2, random_state=0)
        engine.compute_shapley_values(X, value_fn, "ctx", {"marker": 7})
        assert seen["ctx"]["marker"] == 7


# ---------------------------------------------------------------------------
# Value function edge cases
# ---------------------------------------------------------------------------
class TestValueFunctionEdges:
    def test_convex_hull_high_dimensional_range_fallback(self):
        """More than 8 dimensions uses the product of ranges."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 12))
        value = dtmod.formative_archetypal_convex_hull(X, np.arange(12))
        assert np.isfinite(value)
        assert value != 0.0

    def test_prototypical_coverage_with_a_single_row(self):
        assert dtmod.formative_prototypical_coverage(
            np.ones((1, 3)), np.arange(1)
        ) == 0.0


# ---------------------------------------------------------------------------
# Graph topology features
# ---------------------------------------------------------------------------
class TestGraphTopologyFeatures:
    @pytest.mark.parametrize(
        "feature",
        ["degree", "clustering", "pagerank", "triangles", "betweenness",
         "closeness", "eigenvector"],
    )
    def test_each_topology_feature(self, graph_data, feature):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            graph_topology_features=[feature], random_state=0,
        )
        topology = dt._compute_graph_topology_features(edges, len(features))
        assert feature in topology.columns
        assert len(topology) == len(features)

    def test_unknown_topology_feature_warns(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(graph_topology_features=["not_a_feature"], random_state=0)
        with pytest.warns(UserWarning, match="Unknown topology feature"):
            dt._compute_graph_topology_features(edges, len(features))

    def test_eigenvector_failure_falls_back_to_zeros(self, monkeypatch):
        """Eigenvector centrality does not converge on every graph."""
        import networkx as nx

        dt = DataTypical(graph_topology_features=["eigenvector"], random_state=0)

        def boom(*args, **kwargs):
            raise nx.PowerIterationFailedConvergence(100)

        monkeypatch.setattr(nx, "eigenvector_centrality", boom)
        edges = np.array([[0, 1], [1, 2], [2, 0]])
        with pytest.warns(UserWarning, match="Eigenvector centrality failed"):
            topology = dt._compute_graph_topology_features(edges, 3)
        assert topology["eigenvector"].eq(0.0).all()

    def test_networkx_missing_raises_a_useful_error(self, graph_data, monkeypatch):
        features, edges = graph_data
        real_import = builtins.__import__

        def fake_import(mod, *args, **kwargs):
            if mod.split(".")[0] == "networkx":
                raise ImportError("no networkx")
            return real_import(mod, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "networkx", None)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        dt = DataTypical(random_state=0)
        with pytest.raises(ImportError):
            dt._compute_graph_topology_features(edges, len(features))


# ---------------------------------------------------------------------------
# Text paths reached through fit()/transform() rather than fit_text()
# ---------------------------------------------------------------------------
class TestTextThroughUnifiedEntryPoints:
    def test_fit_and_transform_with_metadata(self, corpus, text_metadata):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            stereotype_column="Year", random_state=0,
        )
        dt.fit(corpus, text_metadata=text_metadata)
        out = dt.transform(corpus)
        assert len(out) == len(corpus)

    def test_fit_with_shapley_and_metadata(self, corpus, text_metadata):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=True,
            stereotype_column="Year", verbose=True, random_state=0,
        )
        dt.fit(corpus, text_metadata=text_metadata)
        out = dt.transform(corpus)
        assert "archetypal_shapley_rank" in out.columns

    def test_fit_with_shapley_and_keywords(self, corpus):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=True,
            stereotype_keywords=["protein"], random_state=0,
        )
        dt.fit(corpus)
        assert dt._stereotype_source_fit_ is not None

    def test_fit_text_with_shapley_and_keywords(self, corpus, capsys):
        """The public fit_text() duplicates the internal path."""
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=True,
            stereotype_keywords=["protein"], verbose=True, random_state=0,
        )
        dt.fit_text(corpus)
        assert "SHAPLEY" in capsys.readouterr().out

    def test_transform_text_with_shapley(self, corpus, text_metadata):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=True,
            stereotype_column="Year", random_state=0,
        )
        dt.fit_text(corpus, text_metadata=text_metadata)
        out = dt.transform_text(corpus)
        assert "archetypal_shapley_rank" in out.columns

    def test_metadata_length_must_match_the_corpus(self, corpus):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        with pytest.raises(ValueError, match="must match"):
            dt.fit_text(corpus, text_metadata=pd.DataFrame({"Year": [1.0, 2.0]}))

    def test_non_tfidf_vectorizer_is_not_implemented(self, corpus):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        with pytest.raises(NotImplementedError, match="TF-IDF"):
            dt.fit_text(corpus, vectorizer="word2vec")

    def test_transform_text_before_fit(self, corpus):
        with pytest.raises(RuntimeError, match="Call fit_text first"):
            DataTypical().transform_text(corpus)

    def test_stereotype_source_text_returns_none_without_configuration(self, corpus):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, random_state=0)
        dt.fit_text(corpus)
        assert dt._get_stereotype_source_text() is None


# ---------------------------------------------------------------------------
# Graph paths through _fit_graph / _transform_graph
# ---------------------------------------------------------------------------
class TestGraphInternals:
    def test_numpy_node_features_through_fit(self, graph_data):
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit(features.to_numpy(), edges=edges)
        assert dt.graph_topology_df_ is not None

    def test_topology_collision_through_fit_warns(self, graph_data):
        features, edges = graph_data
        df = features.copy()
        df["degree"] = 1.0
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        with pytest.warns(UserWarning, match="already exists"):
            dt.fit(df, edges=edges)

    def test_transform_graph_with_numpy_features(self, graph_data):
        """Arrays on both sides, with topology off so the columns line up."""
        features, edges = graph_data
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0
        )
        dt.fit(features.to_numpy(), edges=edges, compute_topology=False)
        out = dt.transform(features.to_numpy())
        assert len(out) == len(features)


# ---------------------------------------------------------------------------
# Verbose Shapley reporting, subsampling and the two-tier strategy
# ---------------------------------------------------------------------------
class TestVerboseShapley:
    def test_subsampling_with_formative(self, make_tabular, capsys):
        df = make_tabular(40, 6, seed=11)
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, shapley_top_n=5,
            shapley_compute_formative=True, archetypal_method="nmf", nmf_rank=3,
            n_prototypes=4, stereotype_column="Age", verbose=True, random_state=0,
        )
        dt.fit_transform(df)
        out = capsys.readouterr().out
        assert "Subsampling" in out
        assert "Union" in out
        assert "Core samples" in out

    def test_subsampling_without_formative(self, make_tabular, capsys):
        df = make_tabular(40, 6, seed=12)
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, shapley_top_n=5,
            shapley_compute_formative=False, archetypal_method="nmf", nmf_rank=3,
            n_prototypes=4, stereotype_column="Age", verbose=True, random_state=0,
        )
        dt.fit_transform(df)
        out = capsys.readouterr().out
        assert "SKIPPED" in out
        assert "formative skipped" in out

    def test_two_tier_strategy_is_reported(self, make_tabular, capsys):
        df = make_tabular(60, 6, seed=13)
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, shapley_top_n=6,
            shapley_compute_formative=False, archetypal_method="nmf", nmf_rank=3,
            n_prototypes=4, stereotype_column="Age", verbose=True, random_state=0,
        )
        dt.fit_transform(df)
        out = capsys.readouterr().out
        assert "Two-tier permutation strategy" in out
        assert "Secondary samples" in out


# ---------------------------------------------------------------------------
# Explanation value function branches
# ---------------------------------------------------------------------------
class TestExplanationValueFunctions:
    def _explain(self, dt):
        """Pull the hoisted stereotypical closure out of a fitted estimator."""
        # It is a local closure, so exercise it through a fit with target='min'.
        return dt

    def test_stereotype_target_min(self, make_tabular):
        df = make_tabular(40, 6, seed=14)
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4, shapley_top_n=4,
            shapley_compute_formative=False, archetypal_method="nmf", nmf_rank=3,
            n_prototypes=4, stereotype_column="Age", stereotype_target="min",
            random_state=0,
        )
        dt.fit_transform(df)
        assert np.isfinite(dt.Phi_stereotypical_explanations_).all()

    def test_empty_feature_subsets_return_zero(self, make_tabular):
        """A zero-width subset must short-circuit rather than divide by zero."""
        df = make_tabular(20, 2, seed=15)
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=False, archetypal_method="nmf", nmf_rank=2,
            n_prototypes=3, stereotype_column="Age", random_state=0,
        )
        dt.fit_transform(df)
        assert np.isfinite(dt.Phi_archetypal_explanations_).all()

    def test_formative_ranks_are_constant_when_phi_is_flat(self, df_small, monkeypatch):
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=True, archetypal_method="nmf", nmf_rank=3,
            n_prototypes=4, random_state=0,
        )
        dt.fit_transform(df_small)
        dt.Phi_archetypal_formative_ = np.ones((len(df_small), 3))
        ranks = dt._compute_shapley_formative_ranks()
        assert ranks["archetypal_shapley_rank"].eq(0.5).all()

    def test_positional_index_when_train_index_is_absent(self, df_small):
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=True, archetypal_method="nmf", nmf_rank=3,
            n_prototypes=4, stereotype_column="Age", random_state=0,
        )
        dt.fit_transform(df_small)
        dt.train_index_ = None
        assert "archetypal" in dt.get_shapley_explanations(0)
        assert "archetypal" in dt.get_formative_attributions(0)

    def test_stereotypical_formative_attribution_is_returned(self, df_small):
        dt = DataTypical(
            shapley_mode=True, shapley_n_permutations=4,
            shapley_compute_formative=True, archetypal_method="nmf", nmf_rank=3,
            n_prototypes=4, stereotype_column="Age", random_state=0,
        )
        dt.fit_transform(df_small)
        assert "stereotypical" in dt.get_formative_attributions(df_small.index[0])


# ---------------------------------------------------------------------------
# Preprocessing branches
# ---------------------------------------------------------------------------
class TestPreprocessingBranches:
    def test_verbose_column_drop_messages(self, df_small, capsys):
        df = df_small.copy()
        df["row"] = np.arange(len(df), dtype=float)
        df["code"] = np.random.default_rng(0).permutation(np.arange(len(df)))
        DataTypical(
            archetypal_method="nmf", nmf_rank=3, verbose=True, random_state=0
        ).fit(df)
        out = capsys.readouterr().out
        assert "strictly monotonic" in out
        assert "high-uniqueness" in out

    def test_missing_label_columns_warn(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            label_columns=["not_here"],
        )
        with pytest.warns(UserWarning, match="Label columns not found"):
            dt.fit(df_small)

    def test_label_columns_covering_everything(self, df_small):
        """If every column is a label there are no features left."""
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            label_columns=list(df_small.columns),
        )
        with pytest.raises(DataTypicalError):
            dt.fit(df_small)


# ---------------------------------------------------------------------------
# Archetypal backend: verbose output and the remaining PCHA branches
# ---------------------------------------------------------------------------
class TestArchetypalBranches:
    def test_verbose_pcha(self, df_wide, capsys):
        pytest.importorskip("py_pcha")
        DataTypical(
            archetypal_method="aa", nmf_rank=3, verbose=True, random_state=0
        ).fit(df_wide)
        assert "PCHA" in capsys.readouterr().out

    def test_verbose_convexhull(self, df_tiny, monkeypatch, capsys):
        monkeypatch.setattr(dtmod, "PCHA", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DataTypical(
                archetypal_method="auto", nmf_rank=3, verbose=True, random_state=0
            ).fit(df_tiny)
        assert "convex hull" in capsys.readouterr().out

    def test_verbose_convexhull_failure_and_nmf_fallback(
        self, df_tiny, monkeypatch, capsys
    ):
        monkeypatch.setattr(dtmod, "PCHA", None)

        def boom(*args, **kwargs):
            raise RuntimeError("hull refused")

        monkeypatch.setattr(dtmod, "ConvexHull", boom)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DataTypical(
                archetypal_method="auto", nmf_rank=3, verbose=True, random_state=0
            ).fit(df_tiny)
        out = capsys.readouterr().out
        assert "ConvexHull failed" in out
        assert "NMF fallback" in out

    def test_verbose_pcha_failure(self, df_wide, monkeypatch, capsys):
        def boom(*args, **kwargs):
            raise RuntimeError("pcha refused")

        monkeypatch.setattr(dtmod, "PCHA", boom)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DataTypical(
                archetypal_method="auto", nmf_rank=3, verbose=True, random_state=0
            ).fit(df_wide)
        assert "PCHA failed" in capsys.readouterr().out

    def test_pcha_shape_mismatch_is_caught(self, df_wide, monkeypatch):
        def wrong_shape(X, noc, delta=0.0):
            bad = np.ones((2, 2))
            return bad, bad, bad, 1.0, 0.5

        monkeypatch.setattr(dtmod, "PCHA", wrong_shape)
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        with pytest.raises(ConfigError, match="shape error"):
            dt.fit(df_wide)

    def test_negative_values_are_shifted_before_pcha(self):
        """
        PCHA cannot take a negative matrix, so the input is translated first.
        Driven directly, because the preprocessing pipeline always scales into
        [0, 1] before the archetypal step sees anything.
        """
        pytest.importorskip("py_pcha")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 6)) - 5.0
        assert X.min() < 0
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        W, _ = dt._fit_archetypal_aa(X)
        assert dt.archetypal_backend_ == "pcha"
        assert np.isfinite(W).all()

    def test_nmf_with_no_usable_components(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=0, random_state=0)
        with pytest.raises(DataTypicalError, match="at least 1 component"):
            dt._fit_archetypal_nmf(np.ones((4, 3)))

    def test_float32_dtype_branch_in_nmf(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, dtype="float32")
        W, H = dt._fit_archetypal_nmf(np.abs(np.ones((6, 4), dtype=np.float32)))
        assert W.dtype == np.float32

    def test_float32_dtype_branch_in_aa(self, monkeypatch):
        pytest.importorskip("py_pcha")
        rng = np.random.default_rng(0)
        X = np.abs(rng.normal(size=(20, 5))).astype(np.float32)
        dt = DataTypical(archetypal_method="auto", nmf_rank=3, dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W, H = dt._fit_archetypal_aa(X)
        assert W.dtype == np.float32


# ---------------------------------------------------------------------------
# Component fitting: selected_significance skips, overlap, auto-k
# ---------------------------------------------------------------------------
class TestComponentFitting:
    def test_verbose_skip_messages(self, df_small, capsys):
        DataTypical(
            archetypal_method="nmf", nmf_rank=3, verbose=True, random_state=0,
            selected_significance="stereotypical", stereotype_column="Age",
        ).fit(df_small)
        out = capsys.readouterr().out
        assert "Skipping archetypal" in out
        assert "Skipping prototypical" in out

    def test_disallow_overlap_builds_a_forbidden_set(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=4, random_state=0
        )
        dt.disallow_overlap = True
        dt.overlap_alpha = 0.2
        dt.fit(df_small)
        assert dt.prototype_indices_ is not None

    def test_corner_scores_without_edge_hitting_columns(self):
        """One feature means fewer than two columns span the unit interval."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"only": rng.normal(size=25)})
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=1, n_prototypes=4, random_state=0
        )
        dt.disallow_overlap = True
        dt.overlap_alpha = 0.2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df)
        assert dt.prototype_indices_ is not None

    def test_auto_n_prototypes_kneedle(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=8,
            auto_n_prototypes="kneedle", random_state=0,
        )
        dt.fit(df_small)
        assert dt.prototype_indices_ is not None

    def test_knee_with_very_few_prototypes(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=3, random_state=0
        )
        dt.fit(df_small)
        assert dt.knee_ is not None

    def test_verbose_stereotype_reporting(self, df_small, capsys):
        DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=4, verbose=True,
            stereotype_column="Age", stereotype_target="max", random_state=0,
        ).fit(df_small)
        assert "Targeting samples with maximum" in capsys.readouterr().out

    def test_verbose_numeric_stereotype_target(self, df_small, capsys):
        DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=4, verbose=True,
            stereotype_column="Age", stereotype_target=55.0, random_state=0,
        ).fit(df_small)
        assert "Mean distance to target" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Scoring errors
# ---------------------------------------------------------------------------
class TestScoringErrors:
    def test_archetypal_requested_but_not_fitted(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            selected_significance="prototypical",
        )
        dt.fit(df_small)
        dt.selected_significance = None
        with pytest.raises(RuntimeError, match="archetypes were not fitted"):
            dt.transform(df_small)

    def test_prototypical_requested_but_not_fitted(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0,
            selected_significance="archetypal",
        )
        dt.fit(df_small)
        dt.selected_significance = None
        with pytest.raises(RuntimeError, match="prototypes were not fitted"):
            dt.transform(df_small)

    def test_feature_dimension_mismatch(self, df_small, make_tabular):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        dt.feature_columns_ = None
        with pytest.raises(Exception):
            dt.transform(make_tabular(10, 3, seed=1))

    def test_assignments_without_stored_prototype_rows(self, df_small):
        dt = DataTypical(
            archetypal_method="nmf", nmf_rank=3, n_prototypes=4, random_state=0
        )
        dt.fit(df_small)
        dt.prototype_features_l2_ = None
        best_cos, labels = dt._assignments_cosine(
            np.asarray(dt.prototype_features_l2_ if False else np.eye(len(df_small))),
            dt.prototype_indices_,
        )
        assert len(labels) == len(df_small)


# ---------------------------------------------------------------------------
# Defensive branches that only fire on a corrupted estimator
# ---------------------------------------------------------------------------
class TestDefensiveBranches:
    def test_unknown_data_type_in_fit(self, df_small, monkeypatch):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        monkeypatch.setattr(dt, "_validate_data_type", lambda detected: "audio")
        with pytest.raises(RuntimeError, match="Unknown data type"):
            dt.fit(df_small)

    def test_unknown_detected_type_in_transform(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        dt.fit(df_small)
        dt._detected_data_type = "audio"
        with pytest.raises(RuntimeError, match="Unknown detected type"):
            dt.transform(df_small)

    def test_fit_transform_passes_return_ranks_only(self, df_small):
        out = DataTypical(
            archetypal_method="nmf", nmf_rank=3, random_state=0
        ).fit_transform(df_small, return_ranks_only=True)
        assert not any(str(c).startswith("f") for c in out.columns)

    def test_stereotypical_rank_rejects_a_bad_target_late(self, df_small):
        """The guard inside _compute_stereotypical_rank, past config validation."""
        dt = DataTypical(stereotype_column="Age", random_state=0)
        dt.stereotype_target = "middling"
        with pytest.raises(ValueError, match="stereotype_target must be"):
            dt._compute_stereotypical_rank(
                np.zeros((len(df_small), 2)), df_small.index, df_small["Age"]
            )
