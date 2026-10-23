"""
Tests for v0.8.2.

Each class pins one change so that it cannot quietly come undone: the import
that hung on a read-only install, the Numba fallback that did not fall back,
the exact feature-level explanations, the stereotypical game that disagreed
with the stereotypical rank, the error message that named the wrong default,
early stopping that could never fire, and the visualisations' handling of text
fits, plot size, wide data and long labels.
"""
import builtins
import importlib
import importlib.util
import itertools
import math
import os
import sys
import types
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import datatypical as dtmod
import datatypical_viz as viz
from datatypical import (ConfigError, DataTypical,
                         exact_explanations_archetypal,
                         exact_explanations_prototypical,
                         exact_explanations_stereotypical,
                         exact_formative_stereotypical)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _load_copy(name, modules=None, blocked=()):
    """Execute datatypical.py again under `name`, with modules swapped or blocked."""
    real_import = builtins.__import__

    def fake_import(mod, *args, **kwargs):
        if mod.split(".")[0] in blocked:
            raise ImportError("blocked for testing: %s" % mod)
        return real_import(mod, *args, **kwargs)

    saved = {k: sys.modules.get(k) for k in (modules or {})}
    spec = importlib.util.spec_from_file_location(name, dtmod.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.modules.update(modules or {})
    builtins.__import__ = fake_import
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = real_import
        sys.modules.pop(name, None)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def _fake_numba(disable_jit=False, compiles=True):
    """A stand-in numba module whose compiler works, is disabled, or fails."""
    fake = types.ModuleType("numba")
    fake.config = types.SimpleNamespace(DISABLE_JIT=disable_jit)

    def jit(*args, **kwargs):
        def decorate(f):
            if compiles:
                return f

            def broken(*a, **k):
                raise TypeError("cannot compile for this NumPy")
            return broken
        return decorate

    fake.jit = jit
    fake.prange = range
    return fake


# ---------------------------------------------------------------------------
# Numba: the import guard and the NumPy fallback
# ---------------------------------------------------------------------------
class TestNumbaGuard:
    def test_a_missing_numba_is_reported_and_falls_back(self):
        with pytest.warns(RuntimeWarning, match="without compiled kernels"):
            module = _load_copy("dt_no_numba_082", blocked={"numba"})
        assert module.NUMBA_AVAILABLE is False
        assert module.JIT_CACHE_ENABLED is False
        assert "could not be imported" in module._NUMBA_FAILURE

    def test_a_numba_that_cannot_compile_is_reported(self):
        with pytest.warns(RuntimeWarning, match="cannot compile"):
            module = _load_copy("dt_broken_numba_082",
                                modules={"numba": _fake_numba(compiles=False)})
        assert module.NUMBA_AVAILABLE is False
        # the kernels are plain functions again, so nothing broken is called
        X = np.random.default_rng(0).random((4, 2))
        assert module._euclidean_min_to_set_dense(X, X).shape == (4,)

    def test_disabling_the_jit_selects_numpy_without_a_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            module = _load_copy("dt_jit_off_082",
                                modules={"numba": _fake_numba(disable_jit=True)})
        assert module.NUMBA_AVAILABLE is False
        assert module._NUMBA_FAILURE is None

    def test_the_flag_is_read(self):
        """v0.8.1 assigned NUMBA_AVAILABLE and never read it."""
        source = open(dtmod.__file__, encoding="utf-8").read()
        reads = source.count("if NUMBA_AVAILABLE") + source.count("NUMBA_AVAILABLE and")
        assert reads >= 6


class TestKernelsAgreeWithNumpy:
    """Every compiled kernel against its NumPy twin, on the same inputs."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(3)
        return rng.normal(size=(23, 4)), rng.normal(size=(17, 4))

    def test_euclidean_min(self, data):
        X, Y = data
        np.testing.assert_allclose(dtmod._euclidean_min_np(X, Y),
                                   dtmod._euclidean_min_jit(X, Y), rtol=1e-10)

    def test_euclidean_min_with_an_empty_reference(self, data):
        X, _ = data
        assert np.all(np.isinf(dtmod._euclidean_min_np(X, np.empty((0, 4)))))
        assert np.all(np.isinf(dtmod._euclidean_chunk_np(X, np.empty((0, 4)), (X * X).sum(1))))

    def test_euclidean_min_without_scipy(self, data, monkeypatch):
        X, Y = data
        monkeypatch.setattr(dtmod, "cdist", None)
        np.testing.assert_allclose(dtmod._euclidean_min_np(X, Y),
                                   dtmod._euclidean_min_jit(X, Y), rtol=1e-10)

    def test_euclidean_chunk(self, data):
        X, Y = data
        x2 = (X * X).sum(axis=1)
        np.testing.assert_allclose(dtmod._euclidean_chunk_np(X, Y, x2),
                                   dtmod._euclidean_chunk_jit(X, Y, x2), rtol=1e-9)

    def test_cosine(self, data):
        X, Y = data
        Xl = dtmod._l2_normalize_rows_dense(X)
        Yl = dtmod._l2_normalize_rows_dense(Y)
        np.testing.assert_allclose(dtmod._cosine_similarity_np(Xl, Yl),
                                   dtmod._cosine_similarity_jit(Xl, Yl), atol=1e-12)

    def test_marginals(self):
        rng = np.random.default_rng(0)
        perm = rng.permutation(9)
        values = np.concatenate([[0.0], rng.normal(size=9)])
        np.testing.assert_allclose(dtmod._compute_marginals_np(perm, values, 9, 3),
                                   dtmod._compute_marginals_jit(perm, values, 9, 3))
        np.testing.assert_allclose(dtmod._compute_feature_marginals_np(perm, values, 9),
                                   dtmod._compute_feature_marginals_jit(perm, values, 9))

    def test_prefix_kernels(self, data):
        X, Y = data
        perm = np.random.default_rng(1).permutation(X.shape[0]).astype(np.int64)
        arche = Y[:3].copy()
        np.testing.assert_allclose(dtmod._prefix_archetypal_np(perm, X, arche),
                                   dtmod._prefix_archetypal_kernel(perm, X, arche), rtol=1e-12)
        terms = np.abs(X[:, 0])
        np.testing.assert_allclose(dtmod._prefix_cummean_np(perm, terms),
                                   dtmod._prefix_cummean_kernel(perm, terms), rtol=1e-12)
        Xl = dtmod._l2_normalize_rows_dense(X)
        np.testing.assert_allclose(dtmod._prefix_prototypical_np(perm, Xl),
                                   dtmod._prefix_prototypical_kernel(perm, Xl), rtol=1e-10)

    def test_prefix_archetypal_on_no_samples(self):
        out = dtmod._prefix_archetypal_np(np.zeros(0, dtype=np.int64),
                                          np.zeros((0, 2)), np.ones((2, 2)))
        np.testing.assert_array_equal(out, [0.0])

    def test_large_problems_take_the_chunked_numpy_path(self, monkeypatch):
        rng = np.random.default_rng(5)
        X, Y = rng.normal(size=(400, 3)), rng.normal(size=(300, 3))
        compiled = dtmod._euclidean_min_to_set_dense(X, Y)
        monkeypatch.setattr(dtmod, "NUMBA_AVAILABLE", False)
        np.testing.assert_allclose(dtmod._euclidean_min_to_set_dense(X, Y), compiled, rtol=1e-9)


class TestTheFallbackGivesTheSameFit:
    @pytest.mark.parametrize("method", ["exact", "monte_carlo"])
    def test_same_ranks_and_explanations(self, df_small, monkeypatch, method):
        """
        The kernel leg is forced on, so under NUMBA_DISABLE_JIT (the coverage
        convention) it runs the kernels' own loops interpreted rather than
        silently comparing the fallback with itself.
        """
        kw = dict(shapley_mode=True, shapley_n_permutations=6, archetypal_method="nmf",
                  nmf_rank=3, n_prototypes=4, stereotype_column="Age",
                  formative_method=method, explanation_method=method, random_state=0)
        monkeypatch.setattr(dtmod, "NUMBA_AVAILABLE", True)
        compiled = DataTypical(**kw)
        a = _quiet(compiled.fit_transform, df_small)
        monkeypatch.setattr(dtmod, "NUMBA_AVAILABLE", False)
        fallback = DataTypical(**kw)
        b = _quiet(fallback.fit_transform, df_small)
        cols = [c for c in a.columns if c.endswith("_rank")]
        np.testing.assert_allclose(b[cols].to_numpy(float), a[cols].to_numpy(float), atol=1e-9)
        for kind in ("archetypal", "prototypical", "stereotypical"):
            np.testing.assert_allclose(getattr(fallback, "Phi_%s_explanations_" % kind),
                                       getattr(compiled, "Phi_%s_explanations_" % kind),
                                       atol=1e-9)


# ---------------------------------------------------------------------------
# The import that hung
# ---------------------------------------------------------------------------
class TestTheCacheProbeCannotHang:
    def test_a_writable_folder_is_accepted_and_left_clean(self, tmp_path):
        assert dtmod._directory_accepts_writes(str(tmp_path)) is True
        assert list(tmp_path.iterdir()) == []

    def test_a_refused_folder_is_rejected_after_one_attempt(self, tmp_path, monkeypatch):
        """
        The 0.8.1 import spun in tempfile, which retries a refused creation up
        to TMP_MAX times. The probe must try once and stop.
        """
        calls = []

        def refuse(path, flags, mode=0o777):
            calls.append(path)
            raise PermissionError(13, "Access is denied", path)

        monkeypatch.setattr(dtmod.os, "open", refuse)
        assert dtmod._directory_accepts_writes(str(tmp_path)) is False
        assert len(calls) == 1

    def test_a_probe_that_cannot_be_removed_still_counts_as_a_write(self, tmp_path, monkeypatch):
        def refuse(path):
            raise PermissionError(13, "Access is denied", path)

        monkeypatch.setattr(dtmod.os, "remove", refuse)
        assert dtmod._directory_accepts_writes(str(tmp_path)) is True

    def test_an_unusable_path_is_rejected(self, tmp_path):
        a_file = tmp_path / "not-a-folder"
        a_file.write_text("x")
        assert dtmod._directory_accepts_writes(str(a_file / "cache")) is False

    def test_numba_cache_dir_is_the_folder_checked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
        assert dtmod._jit_cache_is_safe() is True
        a_file = tmp_path / "file"
        a_file.write_text("x")
        monkeypatch.setenv("NUMBA_CACHE_DIR", str(a_file / "cache"))
        assert dtmod._jit_cache_is_safe() is False

    def test_otherwise_the_folder_beside_the_source(self, monkeypatch):
        monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
        seen = []
        monkeypatch.setattr(dtmod, "_directory_accepts_writes",
                            lambda path: seen.append(path) or True)
        assert dtmod._jit_cache_is_safe() is True
        assert seen == [os.path.join(os.path.dirname(os.path.abspath(dtmod.__file__)),
                                     "__pycache__")]

    def test_a_read_only_install_imports_and_compiles_without_caching(self, monkeypatch):
        """The whole import, with every write refused, as on the 0.8.1 install."""
        monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
        real_open = os.open

        def refuse_probes(path, flags, *args):
            if "datatypical-write-probe" in str(path):
                raise PermissionError(13, "Access is denied", path)
            return real_open(path, flags, *args)

        monkeypatch.setattr(os, "open", refuse_probes)
        module = _load_copy("dt_read_only_082")
        assert module.JIT_CACHE_ENABLED is False
        X = np.random.default_rng(0).random((5, 3))
        np.testing.assert_allclose(module._euclidean_min_to_set_dense(X, X), np.zeros(5),
                                   atol=1e-7)

    def test_the_decorator_passes_the_decision_on(self, monkeypatch):
        captured = {}

        def fake_jit(*args, **kwargs):
            captured.update(kwargs)
            return lambda f: f

        monkeypatch.setattr(dtmod, "_numba_jit", fake_jit)
        monkeypatch.setattr(dtmod, "JIT_CACHE_ENABLED", False)
        dtmod.jit(nopython=True, cache=True)(lambda: None)
        assert captured["cache"] is False
        dtmod.jit(nopython=True)(lambda: None)
        assert "cache" not in captured or captured["cache"] is False


# ---------------------------------------------------------------------------
# Exact feature-level explanations
# ---------------------------------------------------------------------------
def _brute(d, v):
    """Shapley values of v over d players, by enumerating every coalition."""
    phi = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for size in range(d):
            w = math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
            for S in itertools.combinations(others, size):
                phi[i] += w * (v(list(S) + [i]) - v(list(S)))
    return phi


def _rows(seed, d):
    rng = np.random.default_rng(seed)
    rows = [rng.random(d), np.round(rng.random(d)), np.full(d, 0.3)]
    dup = rng.random(d)
    dup[: max(1, d // 2)] = dup[0]
    rows.append(dup)
    return rows


class TestExactExplanationsMatchEnumeration:
    @pytest.mark.parametrize("d", [1, 2, 3, 4, 6, 7])
    def test_archetypal(self, d):
        for x in _rows(d, d):
            a = 1.0 - 2.0 * np.minimum(x, 1.0 - x)
            v = lambda F: 0.0 if not F else float(np.mean(a[F]))
            np.testing.assert_allclose(exact_explanations_archetypal(x[None, :])[0],
                                       _brute(d, v), atol=1e-12)

    @pytest.mark.parametrize("d", [1, 2, 3, 4, 6, 7])
    def test_prototypical(self, d):
        for x in _rows(d + 10, d):
            v = lambda F: 0.0 if not F else float(np.var(x[F]))
            np.testing.assert_allclose(exact_explanations_prototypical(x[None, :])[0],
                                       _brute(d, v), atol=1e-12)

    @pytest.mark.parametrize("target", ["max", "min", 0.4])
    @pytest.mark.parametrize("d", [1, 3, 5])
    def test_stereotypical(self, d, target):
        rng = np.random.default_rng(d)
        X = rng.random((6, d))
        y = rng.normal(size=6)
        phi = exact_explanations_stereotypical(X, y, target)
        w = dtmod._stereotypical_explanation_weights(y, target)
        for r in range(6):
            v = lambda F: 0.0 if not F else float(w[r] * np.mean(np.abs(X[r, F])))
            np.testing.assert_allclose(phi[r], _brute(d, v), atol=1e-12)

    def test_efficiency(self):
        X = np.random.default_rng(0).random((20, 9))
        np.testing.assert_allclose(exact_explanations_prototypical(X).sum(axis=1),
                                   X.var(axis=1), atol=1e-12)
        a = 1.0 - 2.0 * np.minimum(X, 1.0 - X)
        np.testing.assert_allclose(exact_explanations_archetypal(X).sum(axis=1),
                                   a.mean(axis=1), atol=1e-12)

    def test_the_stereotypical_lengths_must_agree(self):
        with pytest.raises(ValueError, match="entries for"):
            exact_explanations_stereotypical(np.ones((3, 2)), np.ones(4), "max")

    def test_an_invalid_target_is_rejected(self):
        with pytest.raises(ConfigError, match="stereotype_target must be"):
            exact_explanations_stereotypical(np.ones((3, 2)), np.ones(3), "middling")

    def test_no_players_and_no_features(self):
        assert dtmod._mean_game_shapley(np.zeros(0)).shape == (0,)
        assert exact_explanations_archetypal(np.zeros((4, 0))).shape == (4, 0)


class TestTheMonteCarloExplanationsInParallel:
    def test_threads_give_the_serial_values(self):
        """Ten or more features take the threaded path; the permutations are drawn first."""
        X = np.random.default_rng(2).random((5, 12))
        value = lambda Xs, idx, ctx=None: float(np.mean(1.0 - 2.0 * np.minimum(Xs, 1.0 - Xs)))
        phi = {}
        for n_jobs in (1, 2):
            engine = dtmod.ShapleySignificanceEngine(n_permutations=20, random_state=0,
                                                     n_jobs=n_jobs)
            phi[n_jobs], _ = _quiet(engine.compute_feature_shapley_values, X, value, "arch")
        np.testing.assert_allclose(phi[2], phi[1], atol=1e-12)


class TestExactExplanationsInTheFit:
    KW = dict(shapley_mode=True, archetypal_method="nmf", nmf_rank=3,
              n_prototypes=4, stereotype_column="Age", stereotype_target="max")

    def test_they_are_the_default(self, df_small):
        dt = DataTypical(**self.KW)
        _quiet(dt.fit, df_small)
        assert dt.explanation_method == "exact"
        for kind in ("archetypal", "prototypical", "stereotypical"):
            assert dt.shapley_info_["%s_explanations" % kind]["method"] == "exact"

    def test_they_do_not_depend_on_the_seed(self, df_small):
        a, b = DataTypical(random_state=1, **self.KW), DataTypical(random_state=99, **self.KW)
        _quiet(a.fit, df_small)
        _quiet(b.fit, df_small)
        for kind in ("archetypal", "prototypical", "stereotypical"):
            np.testing.assert_array_equal(getattr(a, "Phi_%s_explanations_" % kind),
                                          getattr(b, "Phi_%s_explanations_" % kind))

    def test_they_are_what_the_sampler_estimates(self, df_small):
        """The sampled closures play the same games, so sampling converges here."""
        exact = DataTypical(**self.KW)
        _quiet(exact.fit, df_small)
        errors = []
        for m in (10, 300):
            sampled = DataTypical(explanation_method="monte_carlo",
                                  shapley_n_permutations=m, random_state=0, **self.KW)
            _quiet(sampled.fit, df_small)
            assert sampled.shapley_info_["archetypal_explanations"]["method"] == "monte_carlo"
            # worst error relative to the scale of each kind's values
            errors.append(max(
                np.abs(getattr(sampled, "Phi_%s_explanations_" % k)
                       - getattr(exact, "Phi_%s_explanations_" % k)).max()
                / np.abs(getattr(exact, "Phi_%s_explanations_" % k)).max()
                for k in ("archetypal", "prototypical", "stereotypical")))
        assert errors[1] < errors[0]
        assert errors[1] < 0.1

    def test_subsampling_computes_only_the_chosen_rows(self, df_small):
        dt = DataTypical(shapley_top_n=5, **self.KW)
        _quiet(dt.fit, df_small)
        full = DataTypical(**self.KW)
        _quiet(full.fit, df_small)
        computed = np.flatnonzero(np.any(dt.Phi_archetypal_explanations_ != 0, axis=1))
        assert 5 <= computed.size < len(df_small)
        np.testing.assert_allclose(dt.Phi_archetypal_explanations_[computed],
                                   full.Phi_archetypal_explanations_[computed])

    @pytest.mark.parametrize("method", ["exact", "monte_carlo"])
    def test_no_stereotype_column_means_no_stereotypical_explanations(self, df_small, method):
        """Under subsampling this used to be an all-zero array, drawn as real values."""
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=3,
                         n_prototypes=4, shapley_top_n=5, shapley_n_permutations=4,
                         explanation_method=method)
        _quiet(dt.fit, df_small)
        assert dt.Phi_stereotypical_explanations_ is None
        assert dt.Phi_archetypal_explanations_ is not None

    @pytest.mark.parametrize("name", ["formative_method", "explanation_method"])
    def test_an_unknown_method_names_the_real_default(self, df_small, name):
        """
        The formative_method message used to call 'monte_carlo' the default,
        which told a reader their exact fits had been sampled.
        """
        dt = DataTypical(**{name: "bogus"})
        with pytest.raises(ConfigError) as info:
            dt.fit(df_small)
        default = DataTypical.__dataclass_fields__[name].default
        assert "Use %r, the default" % default in str(info.value)
        assert DataTypical().__getattribute__(name) == default

    def test_the_configuration_round_trips(self):
        cfg = DataTypical(explanation_method="monte_carlo").to_config()
        assert cfg["explanation_method"] == "monte_carlo"
        assert DataTypical.from_config(cfg).explanation_method == "monte_carlo"


# ---------------------------------------------------------------------------
# The stereotypical rank and its formative game agree
# ---------------------------------------------------------------------------
class TestStereotypicalHalvesAgree:
    @pytest.mark.parametrize("target", ["max", "min", 0.0])
    @pytest.mark.parametrize("method", ["exact", "monte_carlo"])
    def test_the_rankings_coincide(self, make_tabular, target, method):
        """
        For 'max' and 'min' the game used the excess over the median, which
        tied half the column; the actual rank uses the distance to the extreme.
        """
        df = make_tabular(40, 4, seed=8)
        df["y"] = np.random.default_rng(8).normal(size=40)
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                         n_prototypes=4, stereotype_column="y", stereotype_target=target,
                         formative_method=method, shapley_n_permutations=400,
                         random_state=0)
        out = _quiet(dt.fit_transform, df)
        rho = out["stereotypical_rank"].corr(out["stereotypical_shapley_rank"],
                                              method="spearman")
        if method == "exact":
            assert rho == pytest.approx(1.0, abs=1e-9)
        else:
            assert rho > 0.9     # the same game, estimated

    @pytest.mark.parametrize("target", ["max", "min", 1.5])
    def test_the_streaming_path_plays_the_same_game(self, target):
        rng = np.random.default_rng(2)
        y = rng.normal(size=12)
        ctx = {"target_values": y, "target": target, "median": float(np.median(y))}
        perm = rng.permutation(12)
        streamed = dtmod._prefix_values_stereotypical(perm, np.zeros((12, 2)), ctx)
        direct = [0.0] + [dtmod.formative_stereotypical_extremeness(
            np.zeros((k, 2)), perm[:k], ctx) for k in range(1, 13)]
        np.testing.assert_allclose(streamed, direct, atol=1e-12)

    def test_tau_ignores_missing_values(self):
        y = np.array([1.0, np.nan, 3.0])
        assert dtmod._stereotype_tau(y, "max") == 3.0
        assert dtmod._stereotype_tau(y, "min") == 1.0
        assert dtmod._stereotype_tau(y, 2) == 2.0

    def test_an_ordered_categorical_column_survives_a_formative_fit(self, make_tabular):
        """It was accepted by fit() but crashed the formative conversion."""
        df = make_tabular(30, 4, seed=3)
        df["grade"] = pd.Categorical(np.take(["low", "mid", "high"], np.arange(30) % 3),
                                     categories=["low", "mid", "high"], ordered=True)
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                         n_prototypes=4, stereotype_column="grade", stereotype_target="max",
                         random_state=0)
        out = _quiet(dt.fit_transform, df)
        assert out["stereotypical_shapley_rank"].notna().all()


# ---------------------------------------------------------------------------
# Early stopping, and saying what sampling will cost
# ---------------------------------------------------------------------------
class TestEarlyStopping:
    @pytest.mark.parametrize("patience", [0, 1, 3, 10, 12])
    def test_the_stated_earliest_stop(self, patience):
        engine = dtmod.ShapleySignificanceEngine(early_stopping_patience=patience)
        assert engine.earliest_early_stop() == max(50, 20 + 10 * patience)

    @pytest.mark.parametrize("patience", [1, 3, 10])
    def test_it_is_the_actual_earliest_stop(self, patience):
        """With a tolerance nothing can fail, the run ends exactly there."""
        X = np.random.default_rng(0).random((6, 3))
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=400, early_stopping_patience=patience,
            early_stopping_tolerance=1e12, random_state=0, n_jobs=1)
        _, info = _quiet(engine.compute_feature_shapley_values, X,
                         lambda Xs, idx, ctx=None: float(np.mean(Xs)), "mean")
        assert info["n_permutations_used"] == engine.earliest_early_stop()

    def test_at_the_defaults_it_never_fires(self):
        X = np.random.default_rng(0).random((6, 3))
        engine = dtmod.ShapleySignificanceEngine(early_stopping_tolerance=1e12, n_jobs=1)
        _, info = _quiet(engine.compute_feature_shapley_values, X,
                         lambda Xs, idx, ctx=None: float(np.mean(Xs)), "mean")
        assert info["n_permutations_used"] == 100

    def test_verbose_states_the_cost_and_the_limit(self, capsys):
        X = np.random.default_rng(0).random((4, 5))
        engine = dtmod.ShapleySignificanceEngine(verbose=True, n_jobs=1)
        _quiet(engine.compute_feature_shapley_values, X,
               lambda Xs, idx, ctx=None: float(np.mean(Xs)), "mean")
        out = capsys.readouterr().out
        assert "Coalition values to compute, at most: 2,000" in out
        assert "cannot fire below 120 permutations" in out

    def test_verbose_exact_explanations_say_so(self, df_small, capsys):
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=3,
                         n_prototypes=4, verbose=True)
        _quiet(dt.fit, df_small)
        assert "Exact explanations for" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Visualisation: which fit types each plot serves
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def three_fits():
    rng = np.random.default_rng(11)
    tab = pd.DataFrame(rng.normal(size=(30, 5)), columns=["a", "b", "c", "d", "e"])
    tab["y"] = rng.normal(size=30)
    kw = dict(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
              n_prototypes=4, random_state=0)
    fits = {}
    dt = DataTypical(stereotype_column="y", **kw)
    fits["tabular"] = (dt, _quiet(dt.fit_transform, tab))
    corpus = [" ".join(rng.choice(["alpha", "beta", "gamma", "delta", "eps",
                                   "zeta", "eta", "theta"], size=12)) for _ in range(20)]
    dt = DataTypical(**kw)
    fits["text"] = (dt, _quiet(dt.fit_transform, corpus))
    nodes = pd.DataFrame(rng.normal(size=(15, 3)), columns=["x0", "x1", "x2"])
    edges = np.array([[i, (i + 1) % 15] for i in range(15)] + [[0, 7], [3, 11]])
    dt = DataTypical(graph_topology_features=["degree", "pagerank"], **kw)
    fits["graph"] = (dt, _quiet(dt.fit_transform, nodes, edges=edges))
    return fits


class TestEveryPlotServesEveryFitType:
    @pytest.mark.parametrize("kind", ["tabular", "text", "graph"])
    def test_significance_plot(self, three_fits, kind):
        _, results = three_fits[kind]
        assert viz.significance_plot(results, "archetypal") is not None

    @pytest.mark.parametrize("kind", ["tabular", "text", "graph"])
    def test_heatmap(self, three_fits, kind):
        dt, results = three_fits[kind]
        ax = viz.heatmap(dt, results, top_n=5)
        names = {t.get_text() for t in ax.get_xticklabels()}
        assert names and names <= {str(n) for n in viz._feature_names(dt, "heatmap")}

    @pytest.mark.parametrize("kind", ["tabular", "text", "graph"])
    def test_profile_plot(self, three_fits, kind):
        dt, results = three_fits[kind]
        top = results["archetypal_rank"].idxmax()
        ax = viz.profile_plot(dt, top)
        assert len(ax.patches) == len(viz._feature_names(dt, "profile_plot"))

    def test_the_graph_fit_names_its_topology_columns(self, three_fits):
        dt, _ = three_fits["graph"]
        assert {"degree", "pagerank"} <= set(viz._feature_names(dt, "heatmap"))

    def test_text_after_tabular_names_the_terms(self, df_small, corpus):
        """The hazard the old guard protected against: stale tabular names."""
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                         n_prototypes=3, random_state=0)
        _quiet(dt.fit_transform, df_small)
        out = _quiet(dt.fit_transform, corpus)
        names = viz._feature_names(dt, "heatmap", dt.Phi_archetypal_explanations_.shape[1])
        assert names == list(dt.vectorizer_.get_feature_names_out())
        assert not set(names) & set(df_small.columns)
        ax = viz.profile_plot(dt, out["archetypal_rank"].idxmax())
        assert {t.get_text() for t in ax.get_xticklabels()} <= set(names)

    @pytest.mark.parametrize("plot", ["heatmap", "profile_plot"])
    def test_an_unfitted_estimator_is_refused_with_the_reason(self, plot):
        dt = DataTypical(shapley_mode=True)
        dt.Phi_archetypal_explanations_ = np.zeros((2, 3))
        dt.train_index_ = pd.RangeIndex(2)
        with pytest.raises(RuntimeError, match="needs feature names"):
            if plot == "heatmap":
                viz.heatmap(dt, samples=[0, 1])
            else:
                viz.profile_plot(dt, 0)

    def test_inconsistent_state_is_refused(self, three_fits, monkeypatch):
        dt, _ = three_fits["tabular"]
        with pytest.raises(RuntimeError, match="cannot tell which column"):
            viz._feature_names(dt, "heatmap", 99)

    @pytest.mark.parametrize("plot", ["heatmap", "profile_plot"])
    def test_a_fit_without_shapley_is_refused(self, df_small, plot):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3)
        out = _quiet(dt.fit_transform, df_small)
        with pytest.raises(RuntimeError, match="Shapley mode not enabled"):
            if plot == "heatmap":
                viz.heatmap(dt, out)
            else:
                viz.profile_plot(dt, out.index[0])

    def test_profile_needs_some_way_to_colour_the_bars(self, three_fits, monkeypatch):
        dt, results = three_fits["text"]
        monkeypatch.setattr(dt, "_text_scaled_fit_", None)
        with pytest.raises(RuntimeError, match="requires original tabular data"):
            viz.profile_plot(dt, results.index[0])


# ---------------------------------------------------------------------------
# Visualisation: plot size, fonts, markers, colour bars
# ---------------------------------------------------------------------------
def _axes_size(ax):
    ax.figure.canvas.draw()
    box = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    return box.width, box.height


class TestPlotSize:
    @pytest.mark.parametrize("size", [(5.0, 5.0), (3.0, 2.0)])
    def test_the_scatter_gets_exactly_the_plot_area(self, three_fits, size):
        _, results = three_fits["tabular"]
        ax = viz.significance_plot(results, "archetypal", color_by="y", plot_size=size)
        np.testing.assert_allclose(_axes_size(ax), size, atol=0.01)
        cbar_axes = [a for a in ax.figure.axes if a is not ax]
        assert len(cbar_axes) == 1
        assert _axes_size(cbar_axes[0])[1] == pytest.approx(size[1], abs=0.01)

    def test_the_heatmap_gets_exactly_the_plot_area(self, three_fits):
        dt, results = three_fits["tabular"]
        ax = viz.heatmap(dt, results, top_n=6, plot_size=(4.0, 3.0))
        np.testing.assert_allclose(_axes_size(ax), (4.0, 3.0), atol=0.01)
        cax = [a for a in ax.figure.axes if a is not ax][0]
        assert _axes_size(cax)[1] == pytest.approx(3.0, abs=0.01)

    def test_the_profile_gets_exactly_the_plot_area(self, three_fits):
        dt, results = three_fits["tabular"]
        ax = viz.profile_plot(dt, results.index[0], plot_size=(6.0, 2.5))
        np.testing.assert_allclose(_axes_size(ax), (6.0, 2.5), atol=0.01)

    def test_nothing_spills_outside_the_figure(self, three_fits):
        dt, results = three_fits["tabular"]
        ax = viz.heatmap(dt, results, top_n=6, plot_size=(4.0, 3.0))
        fig = ax.figure
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        W, H = fig.get_size_inches()
        assert tight.x0 >= -0.01 and tight.y0 >= -0.01
        assert tight.x1 <= W + 0.01 and tight.y1 <= H + 0.01

    def test_text_and_markers_scale_with_the_area(self, three_fits):
        _, results = three_fits["tabular"]
        w0, h0 = viz._REFERENCE_PLOT_AREA["significance"]
        ax = viz.significance_plot(results, "archetypal", plot_size=(2 * w0, 2 * h0))
        assert ax.xaxis.label.get_size() == pytest.approx(14 * 2)
        assert ax.get_xticklabels()[0].get_size() == pytest.approx(12 * 2)
        assert ax.collections[0].get_sizes()[0] == pytest.approx(50 * 4)

    def test_the_reference_area_reproduces_the_defaults(self, three_fits):
        _, results = three_fits["tabular"]
        ax = viz.significance_plot(results, "archetypal",
                                   plot_size=viz._REFERENCE_PLOT_AREA["significance"])
        assert ax.xaxis.label.get_size() == pytest.approx(14)
        assert ax.collections[0].get_sizes()[0] == pytest.approx(50)

    def test_plot_size_cannot_be_combined_with_an_axes(self, three_fits):
        _, results = three_fits["tabular"]
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="cannot be combined with ax"):
            viz.significance_plot(results, "archetypal", ax=ax, plot_size=(3, 3))

    @pytest.mark.parametrize("bad", [(0, 3), (3,), "big", (3, float("nan"))])
    def test_a_nonsensical_plot_size_is_refused(self, three_fits, bad):
        _, results = three_fits["tabular"]
        with pytest.raises(ValueError, match="plot_size"):
            viz.significance_plot(results, "archetypal", plot_size=bad)

    def test_the_placeholders_accept_a_plot_size(self, fitted_shapley_fast):
        dt, results = fitted_shapley_fast
        ax = viz.significance_plot(results, "archetypal", plot_size=(3, 3))
        np.testing.assert_allclose(_axes_size(ax), (3, 3), atol=0.01)


@pytest.fixture(scope="module")
def fitted_shapley_fast():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
    dt = DataTypical(fast_mode=True, shapley_mode=True, nmf_rank=2, n_prototypes=3,
                     random_state=0)
    return dt, _quiet(dt.fit_transform, df)


class TestHouseStyle:
    """One size per role across the three plots, no bold, no legend frame."""

    def _plots(self, three_fits):
        dt, results = three_fits["tabular"]
        yield viz.significance_plot(results, "archetypal", color_by="y")
        yield viz.heatmap(dt, results, top_n=5)
        yield viz.profile_plot(dt, results.index[0])

    def test_one_size_per_role(self, three_fits):
        for ax in self._plots(three_fits):
            assert ax.xaxis.label.get_size() == 14
            assert ax.yaxis.label.get_size() == 14
            assert ax.title.get_size() == 14
            assert {t.get_size() for t in ax.get_xticklabels()} == {12}

    def test_nothing_is_bold(self, three_fits):
        for ax in self._plots(three_fits):
            for text in ax.figure.findobj(matplotlib.text.Text):
                assert text.get_fontweight() in ("normal", 400), text.get_text()

    def test_legends_have_no_frame(self, three_fits):
        _, results = three_fits["tabular"]
        results = results.copy()
        results["g"] = np.arange(len(results)) % 2
        ax = viz.significance_plot(results, "archetypal", color_by="g")
        assert ax.get_legend().get_frame_on() is False

    def test_importing_the_module_leaves_rcparams_alone(self):
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            before = dict(matplotlib.rcParams)
            importlib.reload(viz)
            assert dict(matplotlib.rcParams) == before

    def test_layout_acts_on_the_plot_s_own_figure(self, three_fits):
        """plt.tight_layout() acted on the current figure, whichever it was."""
        _, results = three_fits["tabular"]
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax2.set_ylabel("a long label " * 4)
        before = ax2.get_position().bounds
        viz.significance_plot(results, "archetypal", ax=ax1)
        assert ax2.get_position().bounds == before

    @pytest.mark.parametrize("name, as_title, as_label", [
        ("fe_concentration", "Fe concentration", "fe concentration"),
        ("pH", "pH", "pH"),
        ("mRNA_level", "mRNA level", "mRNA level"),
        ("Age", "Age", "Age"),
        (3, "3", "3"),
    ])
    def test_labels_are_written_as_names(self, name, as_title, as_label):
        assert viz._display_name(name, capitalise=True) == as_title
        assert viz._display_name(name) == as_label

    def test_labels_turn_only_when_they_would_collide(self):
        assert viz._label_rotation(["F1", "F22"], 0.5, 12) == 0
        assert viz._label_rotation(["a_long_feature_name"], 0.5, 12) == 90
        assert viz._label_rotation([], 0.5, 12) == 0

    @pytest.mark.parametrize("plot", ["heatmap", "profile_plot"])
    def test_short_feature_names_stay_upright(self, three_fits, plot):
        dt, results = three_fits["tabular"]
        if plot == "heatmap":
            ax = viz.heatmap(dt, results, top_n=5)
        else:
            ax = viz.profile_plot(dt, results.index[0])
        assert {t.get_rotation() for t in ax.get_xticklabels()} == {0.0}

    def test_crowded_feature_names_are_turned(self, wide_text_fit):
        dt, results = wide_text_fit
        ax = viz.profile_plot(dt, results.index[0], top_features=40)
        assert {t.get_rotation() for t in ax.get_xticklabels()} == {90.0}

    def test_a_heatmap_without_a_colour_bar(self, three_fits):
        dt, results = three_fits["tabular"]
        ax = viz.heatmap(dt, results, top_n=5, cbar=False)
        assert ax.collections[0].colorbar is None

    @staticmethod
    def _drawn_text_boxes(fig):
        """Every visible, non-empty text with its box, skipping ticks outside the view."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        undrawn = set()
        for each in fig.axes:
            for axis in (each.xaxis, each.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                slack = 1e-9 * (hi - lo)
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not lo - slack <= tick.get_loc() <= hi + slack:
                        undrawn.update({id(tick.label1), id(tick.label2)})
        return [(t.get_text(), t.get_window_extent(renderer))
                for t in fig.findobj(matplotlib.text.Text)
                if t.get_visible() and t.get_text().strip() and id(t) not in undrawn]

    @pytest.mark.parametrize("case", ["scatter continuous", "scatter categories",
                                      "scatter small", "heatmap", "heatmap text",
                                      "profile", "profile graph"])
    def test_no_text_overlaps_other_text_or_the_plot(self, three_fits, case):
        """Measured on the rendered boxes, with a pixel of tolerance."""
        dt, results = three_fits["tabular"]
        if case == "scatter continuous":
            ax = viz.significance_plot(results, "archetypal", color_by="y")
        elif case == "scatter categories":
            results = results.copy()
            results["site"] = np.take(["north", "south", "east", "west"],
                                      np.arange(len(results)) % 4)
            ax = viz.significance_plot(results, "archetypal", color_by="site")
        elif case == "scatter small":
            ax = viz.significance_plot(results, "archetypal", color_by="y",
                                       plot_size=(2.5, 2.5))
        elif case == "heatmap":
            ax = viz.heatmap(dt, results, top_n=10)
        elif case == "heatmap text":
            dt, results = three_fits["text"]
            ax = viz.heatmap(dt, results, top_n=10)
        elif case == "profile":
            ax = viz.profile_plot(dt, results.index[0])
        else:
            dt, results = three_fits["graph"]
            ax = viz.profile_plot(dt, results.index[0])
        boxes = self._drawn_text_boxes(ax.figure)
        overlaps = [(a, b) for (a, ba), (b, bb) in itertools.combinations(boxes, 2)
                    if min(ba.x1, bb.x1) - max(ba.x0, bb.x0) > 1
                    and min(ba.y1, bb.y1) - max(ba.y0, bb.y0) > 1]
        assert not overlaps
        area = ax.get_window_extent(ax.figure.canvas.get_renderer())
        inside = [a for a, b in boxes
                  if min(area.x1, b.x1) - max(area.x0, b.x0) > 1
                  and min(area.y1, b.y1) - max(area.y0, b.y0) > 1]
        assert not inside


# ---------------------------------------------------------------------------
# Visualisation: wide data and long labels
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wide_text_fit():
    rng = np.random.default_rng(0)
    words = ["w%03d" % i for i in range(300)]
    corpus = [" ".join(rng.choice(words, size=40)) for _ in range(25)]
    dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                     n_prototypes=4, random_state=0)
    return dt, _quiet(dt.fit_transform, corpus)


class TestWideData:
    def test_a_wide_heatmap_is_capped_and_says_so(self, wide_text_fit):
        dt, results = wide_text_fit
        with pytest.warns(UserWarning, match="more than can be labelled legibly"):
            ax = viz.heatmap(dt, results, top_n=8)
        shown = ax.collections[0].get_array().reshape(8, -1).shape[1]
        assert 5 <= shown < 300
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(labels) == shown, "every column shown is labelled"

    def test_an_explicit_count_is_honoured_silently(self, wide_text_fit):
        dt, results = wide_text_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ax = viz.heatmap(dt, results, top_n=8, top_features=6)
        assert ax.collections[0].get_array().reshape(8, -1).shape[1] == 6

    def test_a_wide_profile_is_capped_and_says_so(self, wide_text_fit):
        dt, results = wide_text_fit
        with pytest.warns(UserWarning, match="top_features="):
            ax = viz.profile_plot(dt, results.index[0])
        assert 10 <= len(ax.patches) < 300
        assert ax.get_xlabel().startswith("Top %d Features" % len(ax.patches))

    def test_a_defaulted_row_count_is_capped(self, wide_text_fit, monkeypatch):
        dt, results = wide_text_fit
        monkeypatch.setattr(dt, "shapley_top_n", len(results))
        with pytest.warns(UserWarning, match="samples are more than"):
            ax = viz.heatmap(dt, results, top_features=5, figsize=(6, 2))
        assert len(ax.get_yticklabels()) < len(results)

    def test_separators_are_dropped_when_cells_are_thin(self, wide_text_fit):
        dt, results = wide_text_fit
        ax = viz.heatmap(dt, results, top_n=4, top_features=80, figsize=(30, 5))
        assert ax.collections[0].get_linewidths()[0] == 0.0


class TestLongLabels:
    def test_elision_keeps_both_ends(self):
        label = "SA3902.8_NP591.2_F0.15_ix44801"
        short = viz._elide(label, 24)
        assert len(short) == 24 and short.startswith("SA3902") and short.endswith("ix44801")
        assert viz._elide(label, None) == label
        assert viz._elide("short", 24) == "short"
        assert viz._elide(label, 3) == "SA3"

    def test_heatmap_rows_are_elided(self, three_fits):
        dt, results = three_fits["tabular"]
        long_index = ["sample_with_a_very_long_identifier_%02d" % i for i in range(len(results))]
        dt.train_index_ = pd.Index(long_index)
        try:
            ax = viz.heatmap(dt, samples=[0, 1, 2])
            assert all(len(t.get_text()) <= 24 for t in ax.get_yticklabels())
            ax = viz.heatmap(dt, samples=[0, 1, 2], max_label_length=None)
            assert ax.get_yticklabels()[0].get_text() == long_index[0]
        finally:
            dt.train_index_ = pd.RangeIndex(len(results))

    def test_sample_labels_by_mapping_or_function(self, three_fits):
        dt, results = three_fits["tabular"]
        ax = viz.heatmap(dt, results, top_n=3, sample_labels=lambda i: "row %s" % i)
        assert all(t.get_text().startswith("row ") for t in ax.get_yticklabels())
        names = pd.Series(["n%d" % i for i in range(len(results))], index=results.index)
        ax = viz.heatmap(dt, results, top_n=3, sample_labels=names)
        assert all(t.get_text().startswith("n") for t in ax.get_yticklabels())

    def test_the_heatmap_axis_label_is_short(self, three_fits):
        dt, results = three_fits["tabular"]
        assert viz.heatmap(dt, results, top_n=3).get_ylabel() == "Samples (by Actual Rank)"
        ax = viz.heatmap(dt, results, top_n=3, order="formative")
        assert ax.get_ylabel() == "Samples (by Formative Rank)"

    def test_the_profile_title_elides_the_sample(self, three_fits):
        dt, results = three_fits["tabular"]
        ax = viz.profile_plot(dt, results.index[0])
        assert ax.get_title() == "Archetypal Explanations: %s" % results.index[0]


class TestMissingValuesInTheProfile:
    def test_only_the_sample_that_was_missing_loses_its_colour(self):
        """A missing value used to make the whole feature colourless for everyone."""
        rng = np.random.default_rng(9)
        df = pd.DataFrame(rng.normal(size=(25, 4)), columns=list("abcd"))
        df.loc[3, "b"] = np.nan
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                         n_prototypes=3, random_state=0)
        _quiet(dt.fit_transform, df)

        def alpha_of_b(sample):
            ax = viz.profile_plot(dt, sample, order="global")
            names = [t.get_text() for t in ax.get_xticklabels()]
            return ax.patches[names.index("b")].get_facecolor()[3]

        assert alpha_of_b(3) == 0.0
        assert alpha_of_b(5) == 1.0
