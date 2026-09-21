"""
Parameters that were accepted and then ignored.

Through v0.7.7 a group of constructor parameters had no effect at all. Some were
never read anywhere in the module; others recognised exactly one spelling and
silently ignored everything else. In each case a caller could ask for something
and get something different, with nothing in the output to show it.

None of the numerics changed in the fix. The implemented behaviour is now the
only one accepted, so the parameter can no longer claim what the code does not
do.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import ConfigError, DataTypical

KW = dict(archetypal_method="nmf", nmf_rank=3, n_prototypes=5, random_state=0)


def _frame(n=30, d=5, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, d)),
                      columns=["f%d" % i for i in range(d)])
    df["Age"] = rng.integers(30, 85, n).astype(float)
    return df


class TestScaleAndMetrics:
    """Never read anywhere in the module through v0.7.7."""

    @pytest.mark.parametrize(
        "name,value",
        [
            ("scale", "standard"),
            ("scale", "none"),
            ("scale", "robust"),
            ("scale", "wibble"),
            ("distance_metric", "cosine"),
            ("distance_metric", "manhattan"),
            ("similarity_metric", "dot"),
            ("similarity_metric", "euclidean"),
        ],
    )
    def test_an_unimplemented_value_is_rejected(self, name, value):
        with pytest.raises(ConfigError, match="not implemented"):
            DataTypical(random_state=0, **{name: value}).fit(_frame())

    @pytest.mark.parametrize(
        "name,value",
        [("scale", "minmax"), ("distance_metric", "euclidean"),
         ("similarity_metric", "cosine")],
    )
    def test_the_implemented_value_is_accepted(self, name, value):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0,
                         **{name: value})
        dt.fit(_frame())
        assert dt.W_ is not None

    def test_the_error_says_what_the_code_actually_does(self):
        with pytest.raises(ConfigError) as excinfo:
            DataTypical(scale="standard", **KW).fit(_frame())
        message = str(excinfo.value)
        assert "MinMax" in message
        assert "silently" in message

    def test_the_defaults_are_the_implemented_values(self):
        """So nobody using defaults sees any change."""
        dt = DataTypical()
        assert dt.scale == "minmax"
        assert dt.distance_metric == "euclidean"
        assert dt.similarity_metric == "cosine"


class TestAutoNPrototypes:
    """Only the exact string 'kneedle' did anything."""

    @pytest.mark.parametrize("value", ["knee", "elbow", "auto", "Kneedle", ""])
    def test_an_unrecognised_value_is_rejected(self, value):
        with pytest.raises(ConfigError, match="auto_n_prototypes"):
            DataTypical(auto_n_prototypes=value, **KW).fit(_frame())

    def test_none_is_accepted_and_silent(self):
        dt = DataTypical(auto_n_prototypes=None, n_prototypes=8,
                         archetypal_method="nmf", nmf_rank=3, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.fit(_frame(40, 5))
        assert not [w for w in caught if "auto_n_prototypes" in str(w.message)]
        assert len(dt.prototype_indices_) == 8

    def test_kneedle_is_accepted(self):
        dt = DataTypical(auto_n_prototypes="kneedle", n_prototypes=10,
                         archetypal_method="nmf", nmf_rank=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(_frame(40, 5))
        assert dt.prototype_indices_ is not None


class TestRegisterIdeal:
    """
    Stores a vector that no scoring path reads. Kept for compatibility, but it
    no longer implies an effect it does not have.
    """

    def _fitted(self):
        dt = DataTypical(**KW)
        dt.fit(_frame())
        return dt

    def test_it_warns_that_it_has_no_effect(self):
        dt = self._fitted()
        with pytest.warns(DeprecationWarning, match="no effect"):
            dt.register_ideal("target", np.ones(dt.H_.shape[1]))

    def test_the_warning_points_at_the_supported_mechanism(self):
        dt = self._fitted()
        with pytest.warns(DeprecationWarning) as record:
            dt.register_ideal("target", np.ones(dt.H_.shape[1]))
        message = " ".join(str(w.message) for w in record)
        assert "stereotype_column" in message

    def test_it_still_stores_the_vector(self):
        dt = self._fitted()
        dim = dt.H_.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.register_ideal("target", np.arange(dim, dtype=float))
        np.testing.assert_allclose(dt.ideals_["target"],
                                   np.arange(dim, dtype=float))

    def test_it_still_rejects_a_mismatched_dimension(self):
        dt = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="dim"):
                dt.register_ideal("target", np.ones(999))

    def test_it_still_requires_a_fit_first(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="Call fit"):
                DataTypical().register_ideal("target", [1.0, 2.0])

    def test_registering_changes_no_rank(self):
        """Pinned so the documented no-op stays documented."""
        df = _frame()
        dt = DataTypical(**KW)
        before = dt.fit_transform(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.register_ideal("target", np.ones(dt.H_.shape[1]))
        after = dt.transform(df)
        for col in ["archetypal_rank", "prototypical_rank",
                    "stereotypical_rank"]:
            np.testing.assert_allclose(before[col].to_numpy(),
                                       after[col].to_numpy(), atol=1e-12)


class TestParametersThatDoWork:
    """The other side of the ledger, so the sweep is not one-sided."""

    def test_nmf_rank_moves_the_archetypal_ranks(self):
        df = _frame(40, 6)
        a = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=5,
                        random_state=0).fit_transform(df)
        b = DataTypical(archetypal_method="nmf", nmf_rank=5, n_prototypes=5,
                        random_state=0).fit_transform(df)
        assert np.max(np.abs(a["archetypal_rank"] - b["archetypal_rank"])) > 1e-9

    def test_nmf_rank_leaves_the_prototypical_ranks_alone(self):
        """The two lenses are meant to be independent."""
        df = _frame(40, 6)
        a = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=5,
                        random_state=0).fit_transform(df)
        b = DataTypical(archetypal_method="nmf", nmf_rank=5, n_prototypes=5,
                        random_state=0).fit_transform(df)
        np.testing.assert_allclose(a["prototypical_rank"].to_numpy(),
                                   b["prototypical_rank"].to_numpy(),
                                   atol=1e-12)

    def test_n_prototypes_moves_the_prototypical_ranks(self):
        df = _frame(40, 6)
        a = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=3,
                        random_state=0).fit_transform(df)
        b = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=12,
                        random_state=0).fit_transform(df)
        assert np.max(np.abs(
            a["prototypical_rank"] - b["prototypical_rank"])) > 1e-9

    def test_feature_weights_move_the_ranks(self):
        df = _frame(40, 5)
        plain = DataTypical(**KW).fit_transform(df)
        weighted = DataTypical(
            feature_weights=np.array([9.0, 1.0, 1.0, 1.0, 1.0, 0.1]), **KW
        ).fit_transform(df)
        assert np.max(np.abs(
            plain["archetypal_rank"] - weighted["archetypal_rank"])) > 1e-9

    def test_random_state_moves_a_sampled_shapley_fit(self):
        """
        v0.8.0: the archetypal formative values are exact by default and so no
        longer depend on the seed. random_state is still a live parameter for
        the sampled path, which is what this pins.
        """
        df = _frame(40, 5)
        common = dict(shapley_mode=True, shapley_n_permutations=6,
                      shapley_compute_formative=True,
                      formative_method="monte_carlo",
                      archetypal_method="nmf", nmf_rank=3, n_prototypes=5)
        a = DataTypical(random_state=0, **common).fit_transform(df)
        b = DataTypical(random_state=999, **common).fit_transform(df)
        assert np.max(np.abs(
            a["archetypal_shapley_rank"].fillna(0)
            - b["archetypal_shapley_rank"].fillna(0))) > 1e-12

    def test_random_state_does_not_move_the_default_archetypal_formative(self):
        """The other half of the same fact: exact means seed-independent."""
        df = _frame(40, 5)
        common = dict(shapley_mode=True, shapley_n_permutations=6,
                      shapley_compute_formative=True,
                      archetypal_method="nmf", nmf_rank=3, n_prototypes=5)
        a = DataTypical(random_state=0, **common).fit_transform(df)
        b = DataTypical(random_state=999, **common).fit_transform(df)
        np.testing.assert_allclose(
            a["archetypal_shapley_rank"].fillna(0).to_numpy(),
            b["archetypal_shapley_rank"].fillna(0).to_numpy(), atol=1e-12)
