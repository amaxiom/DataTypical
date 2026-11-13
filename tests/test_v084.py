"""
0.8.4: transform() on the training set's scales, two opt-in options for the
archetypal score and its explanations, and proof that the defaults reproduce
0.8.3's fit_transform() exactly.

transform()
    0.8.3 took the corner features, the prototype distance scale d95 and the
    stereotypical rank's normaliser from the rows it was given, so a row's
    ranks depended on which other rows were in the call. The fit now records
    the training set's, and transform() scores every row on them.

corner_features
    The corner term of archetypal_rank reads the two highest-variance features
    that span the scaled range. Logged counts with zeros win that choice on
    the gap below their smallest positive value, so the term reads how a count
    is recorded. The default rule is kept and now warns when it picks a
    coarsely recorded feature; 'archetypes' and explicit names choose the
    features instead.

archetypal_explanation_baseline
    The fitted archetypal explanation game measures each feature from the
    bottom of its scaled range, so a feature with a negative projection
    coefficient on the dominant archetype can contribute only zero or less.
    'mean' measures each feature from the training mean; ranks do not move.

Parity: every default output of fit_transform() (ranks, weights, archetypes,
explanations, formative values) is compared bit for bit with the released
0.8.3 module, a copy saved as datatypical_083.py and named by the environment
variable DATATYPICAL_083_MODULE (or kept in baseline_083/ beside the tree).
Without one the parity tests are skipped.
"""
import importlib.util
import itertools
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

import datatypical as D
from datatypical import ConfigError, DataTypical

BASELINE = os.environ.get("DATATYPICAL_083_MODULE") or os.path.join(
    os.path.dirname(__file__), os.pardir, "baseline_083", "datatypical_083.py")


def _released():
    if not os.path.exists(BASELINE):
        pytest.skip("no copy of the released 0.8.3 module (set DATATYPICAL_083_MODULE)")
    if "datatypical_083" in sys.modules:
        return sys.modules["datatypical_083"]
    spec = importlib.util.spec_from_file_location("datatypical_083", BASELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["datatypical_083"] = mod          # dataclasses look the module up
    spec.loader.exec_module(mod)
    assert mod.__version__ == "0.8.3"
    return mod


def wine():
    data = load_wine()
    return pd.DataFrame(data.data, columns=data.feature_names)


def counts_frame(seed=0, n=240):
    """Four continuous features on one latent axis, and two logged counts with zeros."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    df = pd.DataFrame({
        "albumin": 42.0 - 4.0 * z + rng.normal(0, 2.0, n),
        "rdw": 13.0 + 1.2 * z + rng.normal(0, 0.6, n),
        "mpv": 10.0 - 1.0 * z + rng.normal(0, 0.7, n),
        "hgb": 130.0 - 8.0 * z + rng.normal(0, 6.0, n),
    })
    for name, lam in (("baso", 0.4), ("eo", 1.0)):
        count = rng.poisson(lam, n).astype(float)
        offset = 0.5 * count[count > 0].min()
        df[name] = np.log10(count + offset)
    return df


def fit(df, quiet=True, **kw):
    params = dict(shapley_mode=True, archetypal_method="aa", nmf_rank=2, n_prototypes=6,
                  random_state=7)
    params.update(kw)
    dt = DataTypical(**params)
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        res = dt.fit_transform(df, return_ranks_only=True)
    return dt, res


def scaled(dt, df):
    return np.clip(dt.scaler_.transform(df[dt.feature_columns_].to_numpy(float)), 0.0, 1.0)


# ---------------------------------------------------------------- parity

PARITY = {
    "wine_aa": (wine, dict(nmf_rank=3)),
    "wine_nmf": (wine, dict(archetypal_method="nmf", nmf_rank=3)),
    "counts_aa": (counts_frame, {}),
    "wine_subsample": (wine, dict(nmf_rank=3, shapley_top_n=0.3)),
    "wine_mc_explanations": (wine, dict(nmf_rank=3, explanation_method="monte_carlo",
                                        shapley_n_permutations=20)),
    "wine_legacy_game": (wine, dict(nmf_rank=3, explanation_game="legacy")),
    "wine_ranks_only": (wine, dict(nmf_rank=3, shapley_mode=False)),
}


@pytest.mark.parametrize("case", sorted(PARITY))
def test_defaults_reproduce_released_083_bit_for_bit(case):
    old = _released()
    make, kw = PARITY[case]
    df = make()
    params = dict(shapley_mode=True, archetypal_method="aa", nmf_rank=2, n_prototypes=6,
                  random_state=7)
    params.update(kw)
    out = {}
    for label, mod in (("new", D), ("old", old)):
        dt = mod.DataTypical(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = dt.fit_transform(df, return_ranks_only=True)
        out[label] = (dt, res)
    (dn, rn), (do, ro) = out["new"], out["old"]
    pd.testing.assert_frame_equal(rn, ro, check_exact=True)
    for name in ("W_", "H_", "Phi_archetypal_explanations_", "Phi_prototypical_explanations_",
                 "Phi_stereotypical_explanations_", "Phi_archetypal_formative_",
                 "Phi_prototypical_formative_", "Phi_stereotypical_formative_",
                 "prototype_indices_", "assignments_"):
        a, b = getattr(dn, name), getattr(do, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), name
    assert dn.shapley_info_.keys() == do.shapley_info_.keys()
    # and transform on new rows: 0.8.4 scores them on the training set's
    # scales, so it is compared with 0.8.3 only for a fit that kept none
    # (one saved before 0.8.4), which is scored the 0.8.3 way.
    rng = np.random.default_rng(3)
    new_rows = df.sample(20, random_state=1) * (1 + 0.01 * rng.normal(size=(20, df.shape[1])))
    del dn._transform_reference_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd.testing.assert_frame_equal(dn.transform(new_rows, return_ranks_only=True),
                                      do.transform(new_rows, return_ranks_only=True), check_exact=True)


def test_defaults_reproduce_released_083_on_text():
    old = _released()
    corpus = ["the cat sat on the mat", "a dog barked at the cat", "the mat was red",
              "dogs and cats are pets", "the red dog sat", "pets sleep on mats",
              "a cat and a dog", "the pet shop sells mats"] * 2
    res = {}
    for label, mod in (("new", D), ("old", old)):
        dt = mod.DataTypical(shapley_mode=True, nmf_rank=2, n_prototypes=3, random_state=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res[label] = (dt, dt.fit_transform(corpus, return_ranks_only=True))
    pd.testing.assert_frame_equal(res["new"][1], res["old"][1], check_exact=True)
    assert np.array_equal(res["new"][0].Phi_archetypal_explanations_,
                          res["old"][0].Phi_archetypal_explanations_)


# ---------------------------------------------------------------- corner features: before

def test_default_rule_picks_the_logged_counts_and_warns():
    df = counts_frame()
    with pytest.warns(UserWarning, match="coarsely recorded"):
        dt, _ = fit(df, quiet=False)
    assert sorted(dt.corner_features_) == ["baso", "eo"]
    assert dt.corner_feature_indices_ is None


def test_default_rule_does_not_warn_on_continuous_features():
    dt = DataTypical(shapley_mode=False, archetypal_method="aa", nmf_rank=3, random_state=7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dt.fit_transform(wine(), return_ranks_only=True)
    assert not [w for w in caught if "corner term" in str(w.message)]
    assert len(dt.corner_features_) == 2


# ---------------------------------------------------------------- corner features: after

def _manual_archetypal_score(dt, Xs, take):
    B = dt.H_.T @ np.linalg.pinv(dt.H_ @ dt.H_.T + 1e-6 * np.eye(dt.H_.shape[0]))
    W = np.maximum(Xs @ B, 0.0)
    S = W.sum(axis=1)
    S[S == 0] = 1.0
    wmax = (W / S[:, None]).max(axis=1)
    m = np.minimum(Xs[:, take], 1.0 - Xs[:, take])
    corner = 1.0 - np.clip(np.sqrt((m * m).sum(axis=1)) / math.sqrt(len(take)), 0, 1)
    return 0.7 * wmax + 0.3 * corner


def test_archetypes_rule_reads_the_features_the_archetypes_differ_on():
    df = counts_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dt, res = fit(df, quiet=False, corner_features="archetypes")
    assert not [w for w in caught if "corner term" in str(w.message)]
    spread = dt.H_.var(axis=0)
    want = np.argsort(-spread, kind="stable")[:2]
    assert list(dt.corner_feature_indices_) == list(want)
    assert set(dt.corner_features_) <= {"albumin", "rdw", "mpv", "hgb"}
    Xs = scaled(dt, df)
    assert np.allclose(res["archetypal_rank"], _manual_archetypal_score(dt, Xs, want), atol=1e-9)
    # explanations follow the score, and the counts are no longer corner players
    E = dt.Phi_archetypal_explanations_
    assert np.allclose(E.sum(axis=1), res["archetypal_rank"], atol=1e-9)


def test_named_and_positional_corner_features():
    df = counts_frame()
    dt, res = fit(df, corner_features=["albumin", "mpv"])
    assert dt.corner_features_ == ["albumin", "mpv"]
    assert list(dt.corner_feature_indices_) == [0, 2]
    dt2, res2 = fit(df, corner_features=[0, 2])
    pd.testing.assert_frame_equal(res, res2, check_exact=True)
    dt1, res1 = fit(df, corner_features=["rdw"])
    Xs = scaled(dt1, df)
    assert np.allclose(res1["archetypal_rank"], _manual_archetypal_score(dt1, Xs, [1]), atol=1e-9)


def test_fixed_corner_features_are_used_by_transform():
    df = counts_frame()
    dt, _ = fit(df, corner_features="archetypes")
    new_rows = counts_frame(seed=5, n=40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dt.transform(new_rows, return_ranks_only=True)
    Xs = scaled(dt, new_rows)
    want = _manual_archetypal_score(dt, Xs, dt.corner_feature_indices_)
    assert np.allclose(out["archetypal_rank"], want, atol=1e-9)


def test_corner_option_moves_only_the_archetypal_rank():
    df = counts_frame()
    _, base = fit(df)
    _, moved = fit(df, corner_features="archetypes")
    assert not np.allclose(base["archetypal_rank"], moved["archetypal_rank"])
    for col in ("prototypical_rank", "archetypal_shapley_rank", "prototypical_shapley_rank"):
        assert np.array_equal(base[col].to_numpy(), moved[col].to_numpy(), equal_nan=True), col


@pytest.mark.parametrize("value", ["foo", ["a", "b", "c"], [], ["albumin", "albumin"], [True],
                                   3.5, [1.5]])
def test_bad_corner_features_are_refused_before_any_work(value):
    # Settings are validated when a fit starts, like every other option.
    with pytest.raises(ConfigError):
        fit(counts_frame(), corner_features=value)


@pytest.mark.parametrize("value", [["nope"], [99], [0, 99]])
def test_unknown_corner_features_are_refused_at_fit(value):
    with pytest.raises(ConfigError):
        fit(counts_frame(), corner_features=value)


# ---------------------------------------------------------------- baseline: before

def _dominant(dt, Xs):
    B = dt.H_.T @ np.linalg.pinv(dt.H_ @ dt.H_.T + 1e-6 * np.eye(dt.H_.shape[0]))
    W = np.maximum(Xs @ B, 0.0)
    return B, W.argmax(axis=1), W.sum(axis=1)


def test_minimum_baseline_lets_a_negative_coefficient_contribute_only_zero_or_less():
    df = counts_frame()
    dt, _ = fit(df, corner_features="archetypes")
    Xs = scaled(dt, df)
    B, k, _ = _dominant(dt, Xs)
    E = dt.Phi_archetypal_explanations_
    corner = set(dt.corner_feature_indices_)
    checked = 0
    for f in range(Xs.shape[1]):
        if f in corner:
            continue
        for j in range(dt.H_.shape[0]):
            rows = (k == j)
            if B[f, j] < 0 and rows.any():
                assert (E[rows, f] <= 1e-15).all()
                checked += 1
    assert checked > 0
    assert dt.archetypal_explanation_base_ is None


# ---------------------------------------------------------------- baseline: after

def test_mean_baseline_measures_each_feature_from_the_training_mean():
    df = counts_frame()
    dt, res = fit(df, corner_features="archetypes", archetypal_explanation_baseline="mean", dtype="float64")
    Xs = scaled(dt, df)
    mu = Xs.mean(axis=0)
    B, k, S = _dominant(dt, Xs)
    E = dt.Phi_archetypal_explanations_
    corner = set(dt.corner_feature_indices_)
    for f in range(Xs.shape[1]):
        if f in corner:
            continue
        want = 0.7 * (Xs[:, f] - mu[f]) * B[f, k] / S
        assert np.allclose(E[:, f], want, atol=1e-12)
        for j in range(dt.H_.shape[0]):
            rows = (k == j) & (Xs[:, f] < mu[f] - 1e-9)
            if B[f, j] < 0 and rows.any():
                assert (E[rows, f] > 0).all()
    base = dt.archetypal_explanation_base_
    assert base is not None and base.shape == (len(df),)
    assert np.allclose(E.sum(axis=1) + base, res["archetypal_rank"], atol=1e-9)
    assert dt.shapley_info_["archetypal_explanations"]["baseline"] == "mean"


def test_mean_baseline_corner_values_are_the_two_player_shapley_values():
    df = counts_frame()
    dt, _ = fit(df, corner_features="archetypes", archetypal_explanation_baseline="mean", dtype="float64")
    Xs = scaled(dt, df)
    mu = Xs.mean(axis=0)
    a, b = dt.corner_feature_indices_
    S_minus = D._archetype_weight_terms(Xs, dt.H_, baseline=mu)[0] * 0.7
    corner_part = dt.Phi_archetypal_explanations_ - S_minus

    def u(i, members):
        z = mu[[a, b]].copy()
        for p in members:
            z[p] = Xs[i, [a, b][p]]
        m = np.minimum(z, 1 - z)
        return 1 - math.sqrt((m * m).sum()) / math.sqrt(2)

    for i in range(0, len(df), 17):
        players = (0, 1)
        phi = [0.0, 0.0]
        for p in players:
            for r in range(2):
                for coal in itertools.combinations([q for q in players if q != p], r):
                    w = math.factorial(len(coal)) * math.factorial(1 - len(coal)) / 2
                    phi[p] += w * (u(i, coal + (p,)) - u(i, coal))
        assert abs(corner_part[i, a] - 0.3 * phi[0]) < 1e-12
        assert abs(corner_part[i, b] - 0.3 * phi[1]) < 1e-12


def test_mean_baseline_moves_no_rank():
    df = counts_frame()
    d0, r0 = fit(df)
    d1, r1 = fit(df, archetypal_explanation_baseline="mean")
    pd.testing.assert_frame_equal(r0, r1, check_exact=True)
    assert np.array_equal(d0.W_, d1.W_)
    assert np.array_equal(d0.Phi_prototypical_explanations_, d1.Phi_prototypical_explanations_)


def test_mean_baseline_with_nmf_and_subsampling_keeps_efficiency():
    df = wine()
    dt, res = fit(df, archetypal_method="nmf", nmf_rank=3, archetypal_explanation_baseline="mean",
                  shapley_top_n=0.4)
    rows = dt._explained_rows_
    E, base = dt.Phi_archetypal_explanations_, dt.archetypal_explanation_base_
    assert np.allclose(E[rows].sum(axis=1) + base[rows], res["archetypal_rank"].to_numpy()[rows], atol=1e-6)
    assert (E[~rows] == 0).all() and (base[~rows] == 0).all()


def test_public_function_matches_the_fit():
    df = counts_frame()
    dt, _ = fit(df, corner_features="archetypes", archetypal_explanation_baseline="mean", dtype="float64")
    Xs = scaled(dt, df)
    phi, base = D.exact_explanations_archetypal_fitted(
        Xs, dt.H_, corner_features=dt.corner_feature_indices_, baseline=Xs.mean(axis=0),
        return_base=True)
    assert np.allclose(phi, dt.Phi_archetypal_explanations_, atol=1e-12)
    assert np.allclose(base, dt.archetypal_explanation_base_, atol=1e-12)
    # without a baseline the function is the 0.8.3 one, and its base is zero
    phi0, base0 = D.exact_explanations_archetypal_fitted(
        Xs, dt.H_, corner_features=dt.corner_feature_indices_, return_base=True)
    assert np.array_equal(phi0, D.exact_explanations_archetypal_fitted(
        Xs, dt.H_, corner_features=dt.corner_feature_indices_))
    assert (base0 == 0).all()


@pytest.mark.parametrize("kw", [dict(archetypal_explanation_baseline="median"),
                                dict(archetypal_explanation_baseline="mean", explanation_game="legacy"),
                                dict(archetypal_explanation_baseline="mean",
                                     explanation_method="monte_carlo")])
def test_bad_baseline_settings_are_refused(kw):
    with pytest.raises(ConfigError):
        fit(counts_frame(), **kw)


# ---------------------------------------------------------------- config and state

def test_new_parameters_round_trip_through_config():
    dt = DataTypical(corner_features=["albumin", "mpv"], archetypal_explanation_baseline="mean")
    cfg = dt.to_config()
    assert cfg["corner_features"] == ["albumin", "mpv"]
    assert cfg["archetypal_explanation_baseline"] == "mean"
    back = DataTypical.from_config(cfg)
    assert back.corner_features == ["albumin", "mpv"]
    assert back.archetypal_explanation_baseline == "mean"


def test_refit_clears_the_new_state():
    df = counts_frame()
    dt, _ = fit(df, corner_features="archetypes", archetypal_explanation_baseline="mean", dtype="float64")
    assert dt.corner_feature_indices_ is not None and dt.archetypal_explanation_base_ is not None
    dt.set_params(corner_features=None, archetypal_explanation_baseline="minimum")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt.fit_transform(df, return_ranks_only=True)
    assert dt.corner_feature_indices_ is None
    assert dt.archetypal_explanation_base_ is None


# ---------------------------------------------------------------- transform on the training set's scales
#
# 0.8.3's transform() took three scales from the rows it was given rather than
# from the training set: the corner features of the default rule, the
# prototype distance scale d95, and the stereotypical rank's target and
# largest distance. A row's ranks therefore depended on which other rows were
# in the call; scored alone, its stereotypical_rank was 0.0 for a numeric
# target and 1.0 for 'max' or 'min', and the distance half of its
# prototypical_rank was zero. 0.8.4 records the three at the end of the fit.

RANKS = ["archetypal_rank", "prototypical_rank", "stereotypical_rank"]


def wine_frame():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["cultivar"] = data.target
    return df


def _fit_ranks(df, module=None, **kw):
    params = dict(shapley_mode=False, archetypal_method="aa", nmf_rank=3, n_prototypes=6,
                  random_state=7)
    params.update(kw)
    dt = (module or D).DataTypical(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = dt.fit_transform(df, return_ranks_only=True)
    return dt, res


def _quiet_transform(dt, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dt.transform(X, return_ranks_only=True)


TRANSFORM_CASES = {
    "numeric_target": dict(label_columns=["cultivar"], stereotype_column="alcohol",
                           stereotype_target=13.499),
    "max_target": dict(label_columns=["cultivar"], stereotype_column="alcohol",
                       stereotype_target="max"),
    "min_target": dict(label_columns=["cultivar"], stereotype_column="alcohol",
                       stereotype_target="min"),
    "extremeness": dict(label_columns=["cultivar"]),
    "nmf": dict(label_columns=["cultivar"], archetypal_method="nmf", stereotype_column="alcohol",
                stereotype_target="max"),
    "explained": dict(label_columns=["cultivar"], stereotype_column="alcohol",
                      stereotype_target=13.499, shapley_mode=True, shapley_top_n=0.2),
}


@pytest.mark.parametrize("case", sorted(TRANSFORM_CASES))
def test_transform_scores_each_row_whatever_else_is_in_the_call(case):
    df = wine_frame()
    dt, fitted = _fit_ranks(df, **TRANSFORM_CASES[case])
    pd.testing.assert_frame_equal(_quiet_transform(dt, df), fitted, check_exact=True)
    batch = _quiet_transform(dt, df.iloc[:10])
    alone = pd.concat([_quiet_transform(dt, df.iloc[[i]]) for i in range(0, len(df), 7)])
    for col in RANKS:
        # scikit-learn's NMF.transform solves a batch to tol_nmf, so under the
        # NMF backend the membership weights agree across batches only to
        # about that; every other term is computed row by row.
        atol = 1e-4 if (case == "nmf" and col == "archetypal_rank") else 1e-9
        np.testing.assert_allclose(batch[col].to_numpy(), fitted[col].iloc[:10].to_numpy(),
                                   rtol=0, atol=atol, err_msg=col)
        np.testing.assert_allclose(alone[col].to_numpy(), fitted[col].iloc[::7].to_numpy(),
                                   rtol=0, atol=atol, err_msg=col)


def test_released_083_ranked_a_row_by_its_company():
    old = _released()
    df = wine_frame()
    dt, fitted = _fit_ranks(df, module=old, label_columns=["cultivar"],
                            stereotype_column="alcohol", stereotype_target=13.499)
    alone = _quiet_transform(dt, df.iloc[[0]])
    assert fitted["stereotypical_rank"].iloc[0] > 0.5
    assert alone["stereotypical_rank"].iloc[0] == 0.0
    assert abs(alone["prototypical_rank"].iloc[0] - fitted["prototypical_rank"].iloc[0]) > 0.05


def test_default_corner_features_are_the_training_rows_choice():
    dt, _ = fit(counts_frame())
    take = dt._transform_reference_["corner_take"]
    names = dt._feature_names_for(dt.H_.shape[1])
    assert [names[j] for j in take] == dt.corner_features_


def test_fixed_corner_features_still_win_over_the_recorded_choice():
    dt, _ = fit(counts_frame(), corner_features=["albumin", "mpv"])
    names = dt._feature_names_for(dt.H_.shape[1])
    assert [names[j] for j in dt._corner_take(np.zeros((1, len(names))))] == ["albumin", "mpv"]


@pytest.mark.parametrize("target, values, want", [
    ("max", "beyond", [1.0, 0.0]),
    ("min", "beyond", [0.0, 1.0]),
    (13.0, "beyond", [0.0, 0.0]),
])
def test_stereotype_values_beyond_the_training_range_are_bounded_and_said(target, values, want):
    df = wine_frame()
    dt, _ = _fit_ranks(df, label_columns=["cultivar"], stereotype_column="alcohol",
                       stereotype_target=target)
    new = df.iloc[:2].copy()
    new["alcohol"] = [df["alcohol"].max() + 5.0, df["alcohol"].min() - 5.0]
    with pytest.warns(RuntimeWarning, match="outside what the training rows spanned"):
        out = dt.transform(new, return_ranks_only=True)
    assert out["stereotypical_rank"].tolist() == want


def test_a_fit_saved_before_084_is_scored_the_083_way_and_says_so():
    df = wine_frame()
    dt, fitted = _fit_ranks(df, label_columns=["cultivar"], stereotype_column="alcohol",
                            stereotype_target=13.499)
    del dt._transform_reference_
    with pytest.warns(RuntimeWarning, match="fitted before DataTypical 0.8.4"):
        alone = dt.transform(df.iloc[[0]], return_ranks_only=True)
    assert alone["stereotypical_rank"].iloc[0] == 0.0


def test_the_training_scales_survive_pickling():
    import pickle
    df = wine_frame()
    dt, fitted = _fit_ranks(df, label_columns=["cultivar"], stereotype_column="alcohol",
                            stereotype_target="max")
    back = pickle.loads(pickle.dumps(dt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message="This model was fitted before")
        alone = back.transform(df.iloc[[3]], return_ranks_only=True)
    np.testing.assert_allclose(alone[RANKS].to_numpy(), fitted[RANKS].iloc[[3]].to_numpy(),
                               rtol=0, atol=1e-9)


def test_the_training_scales_are_not_a_parameter_and_a_refit_replaces_them():
    import dataclasses
    assert "_transform_reference_" not in {f.name for f in dataclasses.fields(DataTypical)}
    df = wine_frame()
    dt, _ = _fit_ranks(df, label_columns=["cultivar"], stereotype_column="alcohol",
                       stereotype_target="max")
    assert dt._transform_reference_["stereotype"]["tau"] == float(df["alcohol"].max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt.fit_transform(df.iloc[:60], return_ranks_only=True)
    assert dt._transform_reference_["stereotype"]["tau"] == float(df["alcohol"].iloc[:60].max())


def test_new_rows_without_a_held_out_stereotype_column_are_still_scored():
    df = wine_frame()
    dt, fitted = _fit_ranks(df, label_columns=["cultivar"], stereotype_column="cultivar",
                            stereotype_target="max")
    new = df.drop(columns="cultivar").iloc[:5]
    with pytest.warns(UserWarning, match="is not a column of these rows"):
        out = dt.transform(new, return_ranks_only=True)
    assert out["stereotypical_rank"].isna().all()
    np.testing.assert_allclose(out[["archetypal_rank", "prototypical_rank"]].to_numpy(),
                               fitted[["archetypal_rank", "prototypical_rank"]].iloc[:5].to_numpy(),
                               rtol=0, atol=1e-9)


def test_corner_feature_names_survive_a_dropped_constant_column():
    df = counts_frame()
    df.insert(0, "flat", 1.0)
    dt, _ = fit(df, corner_features=["albumin", "mpv"])
    assert dt.corner_features_ == ["albumin", "mpv"]
    dt, _ = fit(df)
    assert all(name in df.columns for name in dt.corner_features_)


def test_text_transform_scores_each_document_whatever_else_is_in_the_call():
    corpus = ["the cat sat on the mat", "a dog barked at the cat", "the mat was red",
              "dogs and cats are pets", "the red dog sat", "pets sleep on mats",
              "a cat and a dog", "the pet shop sells mats", "a red cat slept",
              "dogs bark at night", "the shop is shut", "mats and pets and cats"]
    for kw in (dict(stereotype_keywords=["cat", "dog"]), {}):
        dt = DataTypical(shapley_mode=False, nmf_rank=2, n_prototypes=3, random_state=7, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = dt.fit_transform(corpus, return_ranks_only=True)
            alone = pd.concat([dt.transform([doc]) for doc in corpus[:5]])
        np.testing.assert_allclose(alone[RANKS].to_numpy(), fitted[RANKS].iloc[:5].to_numpy(),
                                   rtol=0, atol=1e-9)


# ---------------------------------------------------------------- edge cases of the new code

def test_one_feature_data_reads_its_one_feature():
    assert D._corner_features_from_archetypes(np.array([[0.2], [0.9]])).tolist() == [0]
    X = np.array([[0.1], [0.5], [0.95]])
    phi, base = D._corner_explanations_from_baseline(X, np.array([0]), np.array([0.4]))
    m = np.minimum(X[:, 0], 1.0 - X[:, 0])
    np.testing.assert_allclose(phi[:, 0] + base, 1.0 - m, rtol=0, atol=1e-15)


def test_a_baseline_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="baseline has 3 entries for 2 features"):
        D._archetype_weight_terms(np.ones((1, 2)), np.eye(2), None, baseline=np.zeros(3))


class _FixedNMF:
    """A stand-in for a fitted NMF whose transform returns chosen weights."""

    def __init__(self, W):
        self.W = np.asarray(W, dtype=np.float64)

    def transform(self, X):
        return self.W


def test_nmf_baseline_terms_handle_rows_the_solver_left_unexplained():
    H = np.eye(2)
    X = np.array([[0.0, 1.0], [0.0, 0.0], [0.6, 0.2]])
    # row 0: weight on archetype 0 although its projection there is 0, so the
    # profile spread applies; row 1: no weight at all; row 2: an ordinary row.
    W = [[1.0, 0.5], [0.0, 0.0], [0.6, 0.2]]
    terms, wmax, base = D._archetype_weight_terms(X, H, _FixedNMF(W), baseline=np.full(2, 0.3))
    np.testing.assert_allclose(terms.sum(axis=1) + base, wmax, rtol=0, atol=1e-12)
    assert base[0] == 0.0 and wmax[1] == 0.0 and not terms[1].any()


def test_unfitted_names_are_positions_and_archetypes_need_archetypes():
    dt = DataTypical(corner_features="archetypes")
    assert dt._feature_names_for(3) == ["0", "1", "2"]
    dt.H_ = None
    with pytest.raises(ConfigError, match="needs fitted archetypes"):
        dt._resolve_corner_features(np.zeros((4, 2)))


def test_a_feature_named_and_numbered_is_refused_as_twice():
    with pytest.raises(ConfigError, match="twice"):
        fit(counts_frame(), corner_features=["albumin", 0])


def test_degenerate_stereotype_scales_with_and_without_the_training_scales():
    dt = DataTypical(stereotype_column="s", stereotype_target=1.0)
    X = np.full((3, 2), 0.5)
    flat = {"mode": "extremeness", "s_min": 0.5, "s_max": 0.5}
    assert not dt._compute_stereotypical_rank(X, pd.RangeIndex(3), None, reference=flat).any()
    assert not dt._compute_stereotypical_rank(X, pd.RangeIndex(3), None).any()
    with pytest.warns(RuntimeWarning, match="lies at the target"):
        out = dt._compute_stereotypical_rank(X, pd.RangeIndex(3), pd.Series([1.0, 1.0, 1.0]))
    assert out.tolist() == [1.0, 1.0, 1.0]
    at_target = {"mode": "target", "tau": 1.0, "max_dist": 0.0, "resolution": 1e-15}
    with pytest.warns(RuntimeWarning):
        out = dt._compute_stereotypical_rank(X, pd.RangeIndex(3), pd.Series([1.0, 1.0, 3.0]),
                                             reference=at_target)
    assert out.tolist() == [1.0, 1.0, 0.0]
