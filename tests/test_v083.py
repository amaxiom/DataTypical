"""
Tests for v0.8.3: the archetypal and prototypical explanation games on the
fitted model.

The 0.8.2 games were defined feature by feature without the archetypes or
prototypes, and their values did not follow the ranks they accompany; the
prototypical one, the variance of a sample's own values, ran against
prototypical_rank. These tests pin the replacement: each fitted game is the
significance score with the fitted model held fixed, its closed form matches
enumeration of every coalition, the archetypal attributions sum to
archetypal_rank, sampling estimates the same games, and explanation_game='legacy'
still gives the 0.8.2 values.
"""
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import pytest

import datatypical as dtmod
from datatypical import (ConfigError, DataTypical,
                         exact_explanations_archetypal,
                         exact_explanations_archetypal_fitted,
                         exact_explanations_prototypical,
                         exact_explanations_prototypical_fitted)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


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


def _scaled(dt):
    """The training data as the fit scaled it."""
    return dt.scaler_.transform(
        dt._df_original_fit[dt.feature_columns_].to_numpy(dtype=np.float64)
    )[:, dt.keep_mask_].astype(np.float64)


def _archetypal_game(x, H, take):
    """The fitted archetypal game for one sample, written out from the score."""
    B = H.T @ np.linalg.pinv(H @ H.T + 1e-6 * np.eye(H.shape[0]))
    w = np.maximum(x @ B, 0.0)
    S, k = w.sum(), int(w.argmax())
    m = np.minimum(x[take], 1.0 - x[take])
    root = math.sqrt(len(take))

    def v(F):
        if not F:
            return 0.0
        # no weight at all (the origin) leaves only the corner term
        value = 0.7 * sum(x[f] * B[f, k] for f in F) / S if S > 0 else 0.0
        held = [i for i, t in enumerate(take) if t in F]
        if held:
            value += 0.3 * (1.0 - math.sqrt(sum(m[i] ** 2 for i in held)) / root)
        return value

    wmax = w[k] / S if S > 0 else 0.0
    score = 0.7 * wmax + 0.3 * (1.0 - math.sqrt(float(np.sum(m * m))) / root)
    return v, score


def _prototypical_game(x, P, scale):
    """The fitted prototypical game for one sample."""
    p = P[((P - x) ** 2).sum(axis=1).argmin()]
    xh = x / np.linalg.norm(x)
    Ph = P / np.linalg.norm(P, axis=1, keepdims=True)
    q = Ph[(Ph @ xh).argmax()]

    def v(F):
        if not F:
            return 0.0
        return (0.5 * (1.0 - sum((x[f] - p[f]) ** 2 for f in F) / scale ** 2)
                + 0.5 * sum(xh[f] * q[f] for f in F))
    return v


# ---------------------------------------------------------------------------
# The closed forms against enumeration
# ---------------------------------------------------------------------------
class TestFittedGamesMatchEnumeration:
    @pytest.mark.parametrize("d", [2, 3, 4, 6, 7])
    def test_archetypal(self, d):
        rng = np.random.default_rng(d)
        X = rng.random((12, d))
        H = X[:3] * 0.9 + 0.05
        take = dtmod._corner_feature_indices(X)
        phi = exact_explanations_archetypal_fitted(X, H)
        for r in range(6):
            v, score = _archetypal_game(X[r], H, take)
            np.testing.assert_allclose(phi[r], _brute(d, v), atol=1e-12)
            assert phi[r].sum() == pytest.approx(score, abs=1e-12)

    def test_archetypal_on_one_feature(self):
        X = np.array([[0.0], [0.2], [0.7], [1.0]])
        H = np.array([[0.1], [0.9]])
        phi = exact_explanations_archetypal_fitted(X, H)
        for r in range(4):
            v, score = _archetypal_game(X[r], H, np.array([0]))
            np.testing.assert_allclose(phi[r], _brute(1, v), atol=1e-12)
            assert phi[r, 0] == pytest.approx(score, abs=1e-12)

    @pytest.mark.parametrize("d", [1, 2, 3, 5, 7])
    def test_prototypical(self, d):
        rng = np.random.default_rng(d + 20)
        X = rng.random((10, d)) + 0.01
        P = X[[1, 4, 7]]
        scale = 0.37
        phi = exact_explanations_prototypical_fitted(X, P, distance_scale=scale)
        for r in range(10):
            np.testing.assert_allclose(phi[r], _brute(d, _prototypical_game(X[r], P, scale)),
                                       atol=1e-12)

    def test_prototypical_efficiency_and_defaults(self):
        rng = np.random.default_rng(3)
        X = rng.random((30, 5)) + 0.01
        P = X[[0, 9, 18]]
        nearest, d2 = dtmod._nearest_prototype(X, P)
        scale = float(np.percentile(np.sqrt(d2), 95))
        cos = (X / np.linalg.norm(X, axis=1, keepdims=True)
               @ (P / np.linalg.norm(P, axis=1, keepdims=True)).T).max(axis=1)
        phi = exact_explanations_prototypical_fitted(X, P)
        np.testing.assert_allclose(phi.sum(axis=1),
                                   0.5 * (1.0 - d2 / scale ** 2) + 0.5 * cos, atol=1e-12)
        # the normalised prototypes, given or computed, are the same
        Pl2 = P / np.linalg.norm(P, axis=1, keepdims=True)
        np.testing.assert_allclose(exact_explanations_prototypical_fitted(X, P, Pl2), phi,
                                   atol=1e-15)

    def test_a_zero_distance_scale_is_replaced_by_one(self):
        X = np.array([[0.2, 0.4], [0.2, 0.4]])
        np.testing.assert_allclose(exact_explanations_prototypical_fitted(X, X[:1]),
                                   exact_explanations_prototypical_fitted(X, X[:1],
                                                                          distance_scale=1.0))

    def test_an_exact_tie_goes_to_the_first_prototype(self):
        """Count data puts samples midway between prototypes; the choice must not be the last bit's."""
        X = np.array([[1.0, 2.0], [0.0, 2.0], [2.0, 2.0]])
        nearest, d2 = dtmod._nearest_prototype(X, X[[1, 2]])
        assert nearest[0] == 0
        np.testing.assert_allclose(d2, [1.0, 0.0, 0.0])


class TestTheArchetypeWeightTerms:
    def test_they_sum_to_the_scores_membership_share(self, df_small):
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, n_prototypes=4,
                         dtype="float64", random_state=0)
        _quiet(dt.fit, df_small)
        X = _scaled(dt)
        terms, wmax = dtmod._archetype_weight_terms(X, dt.H_)
        np.testing.assert_allclose(terms.sum(axis=1), wmax, atol=1e-12)
        HHT_inv = np.linalg.pinv(dt.H_ @ dt.H_.T + 1e-6 * np.eye(dt.H_.shape[0]))
        W = np.maximum(X @ dt.H_.T @ HHT_inv, 0.0)
        np.testing.assert_allclose(wmax, W.max(axis=1) / W.sum(axis=1), atol=1e-12)

    def test_a_sample_with_no_weight_gets_no_terms(self):
        """At the origin of the scaled space every weight is zero, and so is wmax."""
        X = np.array([[0.0, 0.0], [0.5, 0.8]])
        H = np.array([[1.0, 0.0], [0.0, 1.0]])
        terms, wmax = dtmod._archetype_weight_terms(X, H)
        assert wmax[0] == 0.0 and np.all(terms[0] == 0.0)
        assert terms[1].sum() == pytest.approx(wmax[1])

    def test_nmf_terms_sum_to_the_solvers_weights(self, df_small):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=4,
                         dtype="float64", random_state=0)
        _quiet(dt.fit, df_small)
        X = _scaled(dt)
        terms, wmax = dtmod._archetype_weight_terms(X, dt.H_, dt.nmf_model_)
        W = dt.nmf_model_.transform(X)
        np.testing.assert_allclose(terms.sum(axis=1), W.max(axis=1) / W.sum(axis=1), atol=1e-12)
        np.testing.assert_allclose(terms.sum(axis=1), wmax, atol=1e-12)

    def test_nmf_rows_far_from_the_optimum_fall_back_to_the_archetype_profile(self):
        class FarFromOptimum:
            """A solver that stopped where the active-set projection is negative."""
            def transform(self, X):
                return np.array([[0.3, 0.0], [0.0, 0.0], [0.5, 0.2]])

        H = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        X = np.array([[0.0, 1.0, 0.0], [0.1, 0.1, 0.1], [0.4, 0.3, 0.5]])
        terms, wmax = dtmod._archetype_weight_terms(X, H, FarFromOptimum())
        np.testing.assert_allclose(terms[0], [0.5, 0.0, 0.5])      # profile of archetype 0
        assert np.all(terms[1] == 0.0) and wmax[1] == 0.0          # no weight at all
        np.testing.assert_allclose(terms.sum(axis=1), wmax, atol=1e-12)
        assert wmax[2] == pytest.approx(0.5 / 0.7)

    def test_an_archetype_with_an_empty_profile_spreads_nothing(self):
        class FarFromOptimum:
            def transform(self, X):
                return np.array([[0.4, 0.0]])

        H = np.array([[0.0, 0.0], [1.0, 1.0]])
        terms, wmax = dtmod._archetype_weight_terms(np.array([[0.2, 0.6]]), H, FarFromOptimum())
        np.testing.assert_allclose(terms, [[0.0, 0.0]])
        assert wmax[0] == 1.0


# ---------------------------------------------------------------------------
# In the fit
# ---------------------------------------------------------------------------
class TestFittedGamesInTheFit:
    KW = dict(shapley_mode=True, archetypal_method="nmf", nmf_rank=3, n_prototypes=4,
              stereotype_column="Age", stereotype_target="max", random_state=0)

    def test_they_are_the_default(self, df_small):
        dt = DataTypical(**self.KW)
        _quiet(dt.fit, df_small)
        assert dt.explanation_game == "fitted"
        for kind in ("archetypal", "prototypical", "stereotypical"):
            assert dt.shapley_info_["%s_explanations" % kind]["game"] == "fitted"

    @pytest.mark.parametrize("method", ["aa", "nmf"])
    def test_the_archetypal_attributions_sum_to_the_rank(self, df_small, method):
        dt = DataTypical(**dict(self.KW, archetypal_method=method))
        out = _quiet(dt.fit_transform, df_small)
        np.testing.assert_allclose(dt.Phi_archetypal_explanations_.sum(axis=1),
                                   out["archetypal_rank"].values, atol=1e-6)

    def test_so_they_do_on_the_convex_hull(self, df_small, monkeypatch):
        monkeypatch.setattr(dtmod, "PCHA", None)
        dt = DataTypical(**dict(self.KW, archetypal_method="auto", dtype="float64"))
        out = _quiet(dt.fit_transform, df_small)
        assert dt.archetypal_backend_ == "convexhull"
        np.testing.assert_allclose(dt.Phi_archetypal_explanations_.sum(axis=1),
                                   out["archetypal_rank"].values, atol=1e-9)

    def test_the_prototypical_attributions_sum_to_the_squared_distance_score(self, df_small):
        dt = DataTypical(**dict(self.KW, dtype="float64"))
        out = _quiet(dt.fit_transform, df_small)
        X = _scaled(dt)
        P = dt.prototype_features_
        _, d2 = dtmod._nearest_prototype(X, P)
        d95 = np.percentile(np.sqrt(d2), 95)
        Xl2 = X / np.linalg.norm(X, axis=1, keepdims=True)
        cos = (Xl2 @ dt.prototype_features_l2_.T).max(axis=1)
        game = 0.5 * (1.0 - d2 / d95 ** 2) + 0.5 * cos
        np.testing.assert_allclose(dt.Phi_prototypical_explanations_.sum(axis=1), game,
                                   atol=1e-9)
        # and the score itself, with the distance unsquared and clipped
        score = 0.5 * (1.0 - np.clip(np.sqrt(d2) / d95, 0.0, 1.0)) + 0.5 * cos
        np.testing.assert_allclose(out["prototypical_rank"].values, score, atol=1e-9)
        rho = pd.Series(game).corr(out["prototypical_rank"], method="spearman")
        assert rho > 0.95

    def test_they_depend_on_the_seed_only_through_the_fitted_model(self, df_small):
        """Bit for bit under one seed; under another, as far as the archetypes move."""
        a = DataTypical(**dict(self.KW, random_state=1))
        a2 = DataTypical(**dict(self.KW, random_state=1))
        b = DataTypical(**dict(self.KW, random_state=99))
        for dt in (a, a2, b):
            _quiet(dt.fit, df_small)
        for kind in ("archetypal", "prototypical", "stereotypical"):
            np.testing.assert_array_equal(getattr(a, "Phi_%s_explanations_" % kind),
                                          getattr(a2, "Phi_%s_explanations_" % kind))
        assert np.abs(a.H_ - b.H_).max() < 1e-12
        np.testing.assert_allclose(a.Phi_archetypal_explanations_,
                                   b.Phi_archetypal_explanations_, atol=1e-12)
        np.testing.assert_array_equal(a.Phi_prototypical_explanations_,
                                      b.Phi_prototypical_explanations_)

    def test_the_ranks_do_not_depend_on_the_game(self, df_small):
        a = _quiet(DataTypical(**self.KW).fit_transform, df_small)
        b = _quiet(DataTypical(explanation_game="legacy", **self.KW).fit_transform, df_small)
        pd.testing.assert_frame_equal(a, b)

    def test_legacy_gives_the_082_games(self, df_small):
        dt = DataTypical(explanation_game="legacy", **dict(self.KW, dtype="float64"))
        _quiet(dt.fit, df_small)
        X = _scaled(dt)
        np.testing.assert_allclose(dt.Phi_archetypal_explanations_,
                                   exact_explanations_archetypal(X), atol=1e-12)
        np.testing.assert_allclose(dt.Phi_prototypical_explanations_,
                                   exact_explanations_prototypical(X), atol=1e-12)
        assert dt.shapley_info_["archetypal_explanations"]["game"] == "legacy"

    def test_the_stereotypical_explanations_are_the_same_under_both(self, df_small):
        a, b = DataTypical(**self.KW), DataTypical(explanation_game="legacy", **self.KW)
        _quiet(a.fit, df_small)
        _quiet(b.fit, df_small)
        np.testing.assert_array_equal(a.Phi_stereotypical_explanations_,
                                      b.Phi_stereotypical_explanations_)

    def test_sampling_estimates_the_fitted_games(self, df_small):
        exact = DataTypical(**self.KW)
        _quiet(exact.fit, df_small)
        errors = []
        for m in (10, 300):
            sampled = DataTypical(explanation_method="monte_carlo",
                                  shapley_n_permutations=m, **self.KW)
            _quiet(sampled.fit, df_small)
            info = sampled.shapley_info_["archetypal_explanations"]
            assert info["method"] == "monte_carlo" and info["game"] == "fitted"
            assert info["additivity_error"] < 1e-9
            # the prototypical game is additive, so every permutation is exact
            np.testing.assert_allclose(sampled.Phi_prototypical_explanations_,
                                       exact.Phi_prototypical_explanations_, atol=1e-12)
            errors.append(np.abs(sampled.Phi_archetypal_explanations_
                                 - exact.Phi_archetypal_explanations_).max())
        assert errors[1] < errors[0]
        assert errors[1] < 0.01

    def test_sampling_estimates_the_legacy_games(self, df_small):
        """explanation_game='legacy' with 'monte_carlo' is the pre-0.8.2 computation."""
        exact = DataTypical(explanation_game="legacy", **self.KW)
        _quiet(exact.fit, df_small)
        sampled = DataTypical(explanation_game="legacy", explanation_method="monte_carlo",
                              shapley_n_permutations=300, **self.KW)
        _quiet(sampled.fit, df_small)
        assert sampled.shapley_info_["prototypical_explanations"]["game"] == "legacy"
        for k in ("archetypal", "prototypical"):
            a = getattr(sampled, "Phi_%s_explanations_" % k)
            b = getattr(exact, "Phi_%s_explanations_" % k)
            assert np.abs(a - b).max() / np.abs(b).max() < 0.1

    def test_sampling_works_on_one_feature(self, make_tabular):
        df = make_tabular(15, 1, seed=4, with_age=False)
        kw = dict(shapley_mode=True, archetypal_method="nmf", nmf_rank=1, n_prototypes=3,
                  random_state=0)
        exact = DataTypical(**kw)
        _quiet(exact.fit, df)
        sampled = DataTypical(explanation_method="monte_carlo", shapley_n_permutations=3, **kw)
        _quiet(sampled.fit, df)
        np.testing.assert_allclose(sampled.Phi_archetypal_explanations_,
                                   exact.Phi_archetypal_explanations_, atol=1e-12)

    def test_subsampling_keeps_the_whole_training_sets_constants(self, df_small):
        """The corner features and d95 come from every row, not only those explained."""
        for method in ("exact", "monte_carlo"):
            sub = DataTypical(shapley_top_n=5, explanation_method=method,
                              shapley_n_permutations=20, **self.KW)
            _quiet(sub.fit, df_small)
            full = DataTypical(explanation_method="exact", **self.KW)
            _quiet(full.fit, df_small)
            rows = np.flatnonzero(np.any(sub.Phi_prototypical_explanations_ != 0, axis=1))
            assert 5 <= rows.size < len(df_small)
            np.testing.assert_allclose(sub.Phi_prototypical_explanations_[rows],
                                       full.Phi_prototypical_explanations_[rows], atol=1e-12)
            if method == "exact":
                np.testing.assert_allclose(sub.Phi_archetypal_explanations_[rows],
                                           full.Phi_archetypal_explanations_[rows], atol=1e-12)

    def test_a_prototype_has_something_to_say(self, df_small):
        """Under closeness alone a prototype, matching itself everywhere, would be flat."""
        dt = DataTypical(**self.KW)
        _quiet(dt.fit, df_small)
        phi = dt.Phi_prototypical_explanations_[dt.prototype_indices_]
        assert np.all(phi.std(axis=1) > 1e-6)

    def test_identical_samples_get_identical_explanations(self, df_small):
        df = pd.concat([df_small, df_small.iloc[[3, 3, 17]]], ignore_index=True)
        dt = DataTypical(**self.KW)
        _quiet(dt.fit, df)
        for kind in ("archetypal", "prototypical"):
            phi = getattr(dt, "Phi_%s_explanations_" % kind)
            np.testing.assert_array_equal(phi[3], phi[40])
            np.testing.assert_array_equal(phi[3], phi[41])
            np.testing.assert_array_equal(phi[17], phi[42])

    @pytest.mark.parametrize("selected", ["archetypal", "prototypical"])
    def test_one_significance_type_at_a_time(self, df_small, selected):
        dt = DataTypical(selected_significance=selected, **self.KW)
        out = _quiet(dt.fit_transform, df_small)
        other = "prototypical" if selected == "archetypal" else "archetypal"
        assert getattr(dt, "Phi_%s_explanations_" % other) is None
        phi = getattr(dt, "Phi_%s_explanations_" % selected)
        if selected == "archetypal":
            np.testing.assert_allclose(phi.sum(axis=1), out["archetypal_rank"].values,
                                       atol=1e-6)
        else:
            assert np.all(np.isfinite(phi))

    def test_text_fits(self, corpus):
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                         n_prototypes=3, random_state=0)
        out = _quiet(dt.fit_transform_text, corpus)
        np.testing.assert_allclose(dt.Phi_archetypal_explanations_.sum(axis=1),
                                   out["archetypal_rank"].values, atol=1e-6)
        assert np.all(np.isfinite(dt.Phi_prototypical_explanations_))

    def test_an_unknown_game_is_rejected(self, df_small):
        with pytest.raises(ConfigError, match="explanation_game='bogus' is not recognised"):
            DataTypical(explanation_game="bogus").fit(df_small)

    def test_the_configuration_round_trips(self):
        cfg = DataTypical(explanation_game="legacy").to_config()
        assert cfg["explanation_game"] == "legacy"
        assert cfg["version"] == "0.8.3"
        assert DataTypical.from_config(cfg).explanation_game == "legacy"


class TestTheEngineCanPassTheCoalition:
    def test_a_value_function_that_asks_receives_the_features(self):
        X = np.random.default_rng(5).random((3, 4))
        seen = []

        def value(Xs, idx, ctx, features):
            seen.append(len(features))
            return float(np.sum(X[idx[0], features] * np.arange(1, 5)[features]))

        engine = dtmod.ShapleySignificanceEngine(n_permutations=5, random_state=0, n_jobs=1)
        phi, info = _quiet(engine.compute_feature_shapley_values, X, value, "test",
                           {"with_features": True})
        np.testing.assert_allclose(phi, X * np.arange(1, 5), atol=1e-12)
        assert max(seen) == 4 and info["additivity_error"] < 1e-12


# ---------------------------------------------------------------------------
# Defects found in the pre-release audit, fixed in 0.8.3
# ---------------------------------------------------------------------------
NOT_FITTED = "not samples the model was fitted on"
FORMATIVE = ["archetypal_shapley_rank", "prototypical_shapley_rank",
             "stereotypical_shapley_rank"]


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


class TestFormativeRanksOnTransform:
    """Formative values belong to the training samples, so only they get one."""

    KW = dict(shapley_mode=True, archetypal_method="nmf", nmf_rank=3, n_prototypes=4,
              stereotype_column="Age", random_state=0)

    @pytest.fixture
    def fitted(self, df_small):
        dt = DataTypical(**self.KW)
        return dt, _quiet(dt.fit_transform, df_small)

    def test_new_rows_get_none(self, fitted, make_tabular):
        """They used to get the training rows' at the same index labels."""
        dt, train = fitted
        fresh = make_tabular(10, 6, seed=99)
        out, messages = _caught(dt.transform, fresh)
        assert out[FORMATIVE].isna().all().all()
        assert np.isfinite(out["archetypal_rank"]).all()
        assert any(NOT_FITTED in m and "10 of 10" in m for m in messages)
        assert not np.allclose(train["archetypal_shapley_rank"].iloc[:10],
                               out["archetypal_shapley_rank"].fillna(-1))

    def test_the_training_rows_keep_theirs_without_a_warning(self, fitted, df_small):
        dt, train = fitted
        out, messages = _caught(dt.transform, df_small)
        pd.testing.assert_frame_equal(out[FORMATIVE], train[FORMATIVE])
        assert not any(NOT_FITTED in m for m in messages)

    def test_a_reordered_subset_of_them_keeps_theirs(self, fitted, df_small):
        dt, train = fitted
        part = df_small.iloc[::-1].iloc[:10]
        out, messages = _caught(dt.transform, part)
        pd.testing.assert_frame_equal(out[FORMATIVE], train.loc[part.index, FORMATIVE])
        assert not any(NOT_FITTED in m for m in messages)

    def test_a_changed_row_loses_its_own(self, fitted, df_small):
        dt, train = fitted
        changed = df_small.copy()
        changed.iloc[3, 0] += 1.0
        out, messages = _caught(dt.transform, changed)
        assert out[FORMATIVE].iloc[3].isna().all()
        keep = np.arange(len(df_small)) != 3
        pd.testing.assert_frame_equal(out[FORMATIVE].iloc[keep], train[FORMATIVE].iloc[keep])
        assert any("1 of 40" in m for m in messages)

    def test_missing_values_in_the_same_places_still_match(self, df_small):
        holed = df_small.copy()
        holed.iloc[2, 1] = np.nan
        dt = DataTypical(**self.KW)
        train = _quiet(dt.fit_transform, holed)
        out, messages = _caught(dt.transform, holed)
        assert np.isfinite(out["archetypal_shapley_rank"]).all()
        pd.testing.assert_frame_equal(out[FORMATIVE], train[FORMATIVE])

    def test_text_documents_are_matched_by_position_and_text(self, corpus):
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                         n_prototypes=3, random_state=0)
        train = _quiet(dt.fit_transform_text, corpus)
        assert np.isfinite(train["archetypal_shapley_rank"]).all()
        out, messages = _caught(dt.transform_text, [corpus[1], corpus[0]] + corpus[2:])
        assert out["archetypal_shapley_rank"].iloc[:2].isna().all()
        pd.testing.assert_series_equal(out["archetypal_shapley_rank"].iloc[2:],
                                       train["archetypal_shapley_rank"].iloc[2:])
        assert any(NOT_FITTED in m and "2 of 10" in m for m in messages)

    def test_a_text_fit_saved_before_083_recognises_no_document(self, corpus):
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=2,
                         n_prototypes=3, random_state=0)
        _quiet(dt.fit_text, corpus)
        del dt._text_doc_hashes_
        out = _quiet(dt.transform_text, corpus)
        assert out["archetypal_shapley_rank"].isna().all()


class TestTextTransformScaling:
    @pytest.fixture
    def fitted(self, corpus):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3, random_state=0)
        _quiet(dt.fit_text, corpus)
        return dt

    def test_a_document_is_scaled_as_the_fit_scaled_it(self, fitted, corpus):
        """One document alone used to be scaled by its own maxima: every term 1.0."""
        alone, _ = fitted._preprocess_text_transform([corpus[0]])
        np.testing.assert_allclose(alone.toarray(), fitted._text_scaled_fit_[0].toarray(),
                                   atol=1e-12)
        assert alone.data.min() < 1.0

    def test_the_training_corpus_is_scaled_bit_for_bit_as_before(self, fitted, corpus):
        before = dtmod._sparse_minmax_0_1_nonneg(fitted.vectorizer_.transform(corpus))
        after, _ = fitted._preprocess_text_transform(corpus)
        np.testing.assert_array_equal(after.toarray(), before.toarray())

    def test_a_weight_above_the_training_maximum_is_clipped(self, fitted):
        with pytest.warns(RuntimeWarning, match="clipped"):
            scaled, _ = fitted._preprocess_text_transform(["protein", "galaxy rotation"])
        assert scaled.data.max() == 1.0

    def test_a_text_fit_saved_before_083_scales_the_old_way(self, fitted, corpus):
        del fitted._text_col_max_
        alone, _ = fitted._preprocess_text_transform([corpus[0]])
        np.testing.assert_array_equal(alone.data, np.ones(alone.nnz))


class TestTextStereotypeOnTransform:
    @pytest.fixture
    def fitted(self, corpus, text_metadata):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         stereotype_column="Year", random_state=0)
        train = _quiet(dt.fit_transform_text, corpus, text_metadata=text_metadata)
        return dt, train

    def test_the_training_corpus_uses_the_fit_metadata(self, fitted, corpus):
        dt, train = fitted
        out = _quiet(dt.transform_text, corpus)
        pd.testing.assert_series_equal(out["stereotypical_rank"], train["stereotypical_rank"])

    @pytest.mark.parametrize("which", ["shorter", "same length"])
    def test_other_documents_need_their_own(self, fitted, corpus, which):
        """A corpus of the same length used to be ranked by the training values."""
        dt, _ = fitted
        docs = corpus[:2] if which == "shorter" else list(reversed(corpus))
        with pytest.raises(ConfigError, match="text_metadata="):
            _quiet(dt.transform_text, docs)

    def test_and_are_ranked_by_it(self, fitted, corpus):
        dt, _ = fitted
        meta = pd.DataFrame({"Year": [2001.0, 2020.0, 2010.0]})
        out = _quiet(dt.transform_text, corpus[:3], text_metadata=meta)
        np.testing.assert_allclose(out["stereotypical_rank"].to_numpy(), [0.0, 1.0, 9 / 19])

    def test_through_the_unified_entry_point(self, fitted, corpus):
        dt, _ = fitted
        dt.fit(corpus, text_metadata=pd.DataFrame({"Year": np.arange(10.0)}))
        out = _quiet(dt.transform, corpus[:2],
                     text_metadata=pd.DataFrame({"Year": [5.0, 7.0]}))
        np.testing.assert_allclose(out["stereotypical_rank"].to_numpy(), [0.0, 1.0])

    def test_the_metadata_is_checked(self, fitted, corpus):
        dt, _ = fitted
        with pytest.raises(ValueError, match="must match"):
            dt.transform_text(corpus[:2], text_metadata=pd.DataFrame({"Year": [1.0]}))
        with pytest.raises(ValueError, match="not found in text_metadata"):
            dt.transform_text(corpus[:1], text_metadata=pd.DataFrame({"Month": [1.0]}))


class TestFormativeAttributionsForOneType:
    @pytest.mark.parametrize("selected", ["prototypical", "stereotypical"])
    def test_they_are_returned(self, df_small, selected):
        """They used to raise, blaming fast_mode, whenever archetypal was not selected."""
        dt = DataTypical(shapley_mode=True, selected_significance=selected, n_prototypes=4,
                         stereotype_column="Age", random_state=0)
        _quiet(dt.fit, df_small)
        assert set(dt.get_formative_attributions(df_small.index[0])) == {selected}


class TestTheConvergedFlag:
    @staticmethod
    def _value(Xs, idx):
        return float(np.sum(Xs))

    def test_a_run_that_used_every_permutation_has_not_converged(self):
        engine = dtmod.ShapleySignificanceEngine(n_permutations=30, random_state=0, n_jobs=1)
        X = np.random.default_rng(1).random((25, 3))
        _, info = _quiet(engine.compute_shapley_values, X, self._value, "test")
        assert info["n_permutations_used"] == 30 and info["converged"] is False
        _, info = _quiet(engine.compute_feature_shapley_values, X[:3], self._value, "test")
        assert info["n_permutations_used"] == 30 and info["converged"] is False

    def test_one_that_early_stopping_ended_has(self):
        engine = dtmod.ShapleySignificanceEngine(
            n_permutations=200, random_state=0, n_jobs=1,
            early_stopping_patience=1, early_stopping_tolerance=1e9)
        X = np.random.default_rng(1).random((25, 3))
        _, info = _quiet(engine.compute_shapley_values, X, self._value, "test")
        assert info["n_permutations_used"] == 50 and info["converged"] is True


class TestDensityWeighting:
    @pytest.mark.parametrize("clip_neg", [True, False])
    def test_the_density_is_computed_when_it_weights_the_selection(self, df_small, clip_neg):
        """And takes its settings, which were read and ignored."""
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=4, random_state=0)
        dt.density_weighted_fl, dt.density_k, dt.density_clip_neg = True, 3, clip_neg
        out = _quiet(dt.fit_transform, df_small)
        assert len(dt.prototype_indices_) == 4
        assert np.isfinite(out["prototypical_rank"]).all()


class TestFacilityLocationSelectorIdentity:
    def test_selectors_differ_by_their_settings_and_hash(self):
        """A field-less @dataclass made every selector equal every other."""
        a = dtmod.FacilityLocationSelector(n_prototypes=3)
        assert a != dtmod.FacilityLocationSelector(n_prototypes=5)
        assert isinstance(hash(a), int)


class TestHeatmapRowsOfZeros:
    """An explained row of zeros is a result; the heatmap used to drop it."""

    @pytest.fixture
    def at_target(self, df_small):
        target = float(df_small["Age"].iloc[0])
        dt = DataTypical(shapley_mode=True, archetypal_method="nmf", nmf_rank=3,
                         n_prototypes=4, stereotype_column="Age",
                         stereotype_target=target, shapley_top_n=len(df_small),
                         random_state=0)
        results = _quiet(dt.fit_transform, df_small)
        top = results.nlargest(3, "stereotypical_rank").index
        zero = [(dt.get_shapley_explanations(i)["stereotypical"] == 0).all() for i in top]
        assert any(zero)
        return dt, results

    def test_they_are_shown(self, at_target):
        from datatypical_viz import heatmap
        dt, results = at_target
        ax = heatmap(dt, results=results, significance="stereotypical", top_n=3)
        assert len(ax.get_yticklabels()) == 3

    def test_a_fit_saved_before_083_falls_back_to_the_old_test(self, at_target):
        from datatypical_viz import heatmap
        dt, results = at_target
        del dt._explained_rows_
        ax = heatmap(dt, results=results, significance="stereotypical", top_n=3)
        assert len(ax.get_yticklabels()) < 3

    def test_unexplained_formative_rows_are_reported_as_left_out(self, at_target, capsys):
        from datatypical_viz import heatmap
        dt, results = at_target
        top = results.nlargest(4, "archetypal_shapley_rank").index
        dt._explained_rows_ = ~np.isin(np.arange(len(results)),
                                        [results.index.get_loc(top[0])])
        heatmap(dt, results=results, order="formative", top_n=4)
        out = capsys.readouterr().out
        assert "1/4 top formative archetypal instances have no explanations" in out
        assert "CREATE structure" in out
