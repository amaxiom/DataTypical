"""
Regression tests for the transform and configuration defects found in the
v0.8.0 bug sweep.

Two of the three were silent: a config round trip that quietly dropped
`feature_weights` and produced a different fit, and a transform that quietly
collapsed out-of-range rows onto the training boundary.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import ConfigError, DataTypical

KW = dict(archetypal_method="nmf", nmf_rank=3, n_prototypes=5, random_state=0)
RANKS = ["archetypal_rank", "prototypical_rank"]


def _frame(n=30, d=5, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, d)),
                      columns=["f%d" % i for i in range(d)])
    df["Age"] = rng.integers(30, 85, n).astype(float)
    return df


class TestConfigRoundTrip:
    def test_feature_weights_survives_the_round_trip(self):
        """
        Dropping this silently changed archetypal_rank by 0.33 and
        prototypical_rank by 0.50 on a five-feature frame.
        """
        df = _frame()
        weights = np.array([5.0, 1.0, 1.0, 1.0, 0.1, 1.0])
        dt = DataTypical(feature_weights=weights, **KW)
        original = dt.fit_transform(df)

        clone = DataTypical.from_config(dt.to_config())
        assert clone.feature_weights is not None, \
            "feature_weights was dropped by the config round trip"
        np.testing.assert_allclose(
            np.asarray(clone.feature_weights, dtype=float), weights)

        rebuilt = clone.fit_transform(df)
        for col in RANKS:
            np.testing.assert_allclose(
                original[col].to_numpy(), rebuilt[col].to_numpy(), atol=1e-12,
                err_msg="%s differs after a config round trip" % col,
            )

    def test_to_config_covers_every_init_parameter(self):
        """A parameter missing here is a parameter a round trip loses."""
        from dataclasses import fields as dc_fields

        captured = set(DataTypical().to_config())
        init = {f.name for f in dc_fields(DataTypical) if f.init}
        missing = init - captured
        assert not missing, "to_config omits init parameters: %s" % sorted(missing)

    def test_round_trip_without_weights_still_matches(self):
        df = _frame()
        dt = DataTypical(**KW)
        original = dt.fit_transform(df)
        rebuilt = DataTypical.from_config(dt.to_config()).fit_transform(df)
        for col in RANKS:
            np.testing.assert_allclose(original[col].to_numpy(),
                                       rebuilt[col].to_numpy(), atol=1e-12)


class TestMemoryBudgetValidation:
    @pytest.mark.parametrize("budget", [0, -1, -100])
    def test_a_nonpositive_budget_is_rejected_at_fit(self, budget):
        """
        It used to reach _chunk_len only on the chunked path, so a nonsensical
        budget passed unnoticed on small data and raised much later on large.
        """
        with pytest.raises(ConfigError, match="max_memory_mb"):
            DataTypical(max_memory_mb=budget, **KW).fit(_frame())

    def test_a_sensible_budget_is_accepted(self):
        dt = DataTypical(max_memory_mb=1, **KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(_frame())
        assert dt.settings_["max_memory_mb"] == 1

    def test_the_budget_does_not_change_the_answer(self):
        df = _frame(60, 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = DataTypical(max_memory_mb=4096, **KW).fit_transform(df)
            b = DataTypical(max_memory_mb=1, **KW).fit_transform(df)
        for col in RANKS:
            np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                       atol=1e-9)


class TestTransformClipping:
    """
    The fitted scaler clips to the training range. That is kept, because
    unclipped values break the [0, 1] geometry the archetypal scores assume.
    What was wrong is that it happened in silence.
    """

    def test_out_of_range_rows_are_announced(self):
        df = _frame(40, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        far = df.iloc[:3] * 50.0
        with pytest.warns(RuntimeWarning, match="clipped"):
            dt.transform(far)

    def test_the_warning_names_the_worst_column_and_the_excess(self):
        df = _frame(40, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        far = df.iloc[:2].copy()
        far["f0"] = df["f0"].max() * 500.0
        with pytest.warns(RuntimeWarning) as record:
            dt.transform(far)
        message = " ".join(str(w.message) for w in record)
        assert "f0" in message
        assert "training ranges beyond" in message

    def test_the_warning_counts_the_affected_rows(self):
        df = _frame(40, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        far = df.iloc[:3].copy()
        far.iloc[0] = df.max() * 10.0
        with pytest.warns(RuntimeWarning) as record:
            dt.transform(far)
        message = " ".join(str(w.message) for w in record)
        assert "of 3 row(s)" in message

    def test_in_range_data_does_not_warn(self):
        df = _frame(40, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.transform(df)
        assert not [w for w in caught if "clipped" in str(w.message)]

    def test_clipping_still_happens(self):
        """The numeric contract is unchanged; only the reporting is new."""
        df = _frame(40, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        far = df.iloc[:2].copy()
        far.iloc[0] = df.max() * 100.0
        far.iloc[1] = df.max() * 1000.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = dt.transform(far)
        for col in RANKS:
            assert out[col].iloc[0] == pytest.approx(out[col].iloc[1]), (
                "clipping was changed; both rows should pin to the boundary"
            )

    def test_missing_values_do_not_trigger_the_warning(self):
        """NaN is imputed, not clipped, so it must not be counted."""
        df = _frame(40, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        holed = df.copy()
        holed.iloc[0, 0] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.transform(holed)
        assert not [w for w in caught if "clipped" in str(w.message)]


class TestTransformAlignment:
    def test_column_order_does_not_matter(self):
        df = _frame(30, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        a = dt.transform(df)
        shuffled = df[list(np.random.default_rng(1).permutation(list(df.columns)))]
        b = dt.transform(shuffled)
        for col in RANKS:
            np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                       atol=1e-12)

    def test_extra_columns_are_ignored(self):
        df = _frame(30, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        extended = df.copy()
        extended["not_in_the_fit"] = 1.23
        assert len(dt.transform(extended)) == len(df)

    def test_a_single_row_transforms(self):
        df = _frame(30, 5)
        dt = DataTypical(**KW)
        dt.fit(df)
        out = dt.transform(df.iloc[[0]])
        assert len(out) == 1
        assert np.isfinite(out["archetypal_rank"]).all()


class TestTextEdges:
    def test_an_entirely_unseen_vocabulary_still_scores(self):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        dt.fit_text(["alpha beta gamma", "beta gamma delta",
                     "gamma delta epsilon", "delta epsilon zeta"])
        out = dt.transform_text(["zzz yyy", "www vvv"])
        assert len(out) == 2
        for col in RANKS:
            assert np.isfinite(out[col]).all()

    def test_an_empty_document_is_handled(self):
        corpus = ["alpha beta", "", "beta gamma", "gamma delta", "delta alpha"]
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        out = dt.fit_transform_text(corpus)
        assert len(out) == len(corpus)
        for col in RANKS:
            assert np.isfinite(out[col]).all()

    def test_duplicate_documents_tie_exactly(self):
        corpus = ["alpha beta gamma", "alpha beta gamma", "delta epsilon zeta",
                  "eta theta iota", "kappa lambda mu"]
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        out = dt.fit_transform_text(corpus)
        for col in RANKS:
            assert out[col].iloc[0] == pytest.approx(out[col].iloc[1])
