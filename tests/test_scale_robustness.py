"""
Regression tests for the precision defects found in the v0.8.0 bug sweep.

Through v0.7.7 the raw feature matrix was cast to the working `dtype` (float32
by default) *before* scaling. Since the scaled matrix lives in [0, 1], that cast
bought nothing and cost a great deal: large values overflowed to infinity, small
values underflowed to zero, and any feature carrying its signal beyond the 7th
significant digit lost it outright and was then dropped as constant.

DataTypical's ranks are scale-free by construction, so the tests below assert
exact invariance, not approximate.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import DataTypical, DataTypicalError

RANKS = ["archetypal_rank", "prototypical_rank", "stereotypical_rank"]


def _frame(n=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, d)),
                      columns=["f%d" % i for i in range(d)])
    df["Age"] = rng.integers(30, 85, n).astype(float)
    return df


def _ranks(df, **kw):
    params = dict(archetypal_method="nmf", nmf_rank=3, n_prototypes=5,
                  random_state=0)
    params.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DataTypical(**params).fit_transform(df)


class TestScaleInvariance:
    """The ranks are minmax-normalised, so units must not matter."""

    @pytest.mark.parametrize("factor", [1e-6, 1e-3, 1e3, 1e6, 1e12])
    def test_rescaling_one_feature_leaves_the_ranks_alone(self, factor):
        df = _frame()
        base = _ranks(df)
        scaled = df.copy()
        scaled["f0"] = scaled["f0"] * factor
        out = _ranks(scaled)
        for col in RANKS:
            np.testing.assert_allclose(
                base[col].to_numpy(), out[col].to_numpy(), atol=1e-12,
                err_msg="%s moved when f0 was scaled by %g" % (col, factor),
            )

    @pytest.mark.parametrize(
        "offset,tol",
        [
            (1e3, 1e-12),
            (1e6, 1e-12),
            # Past about 1e8 the offset itself eats into float64's 16 digits, so
            # the spread of an N(0,1) feature is no longer exactly recoverable.
            # That is the representation limit, not the algorithm.
            (1e9, 1e-6),
        ],
    )
    def test_offsetting_one_feature_leaves_the_ranks_alone(self, offset, tol):
        df = _frame()
        base = _ranks(df)
        shifted = df.copy()
        shifted["f0"] = shifted["f0"] + offset
        out = _ranks(shifted)
        for col in RANKS:
            np.testing.assert_allclose(
                base[col].to_numpy(), out[col].to_numpy(), atol=tol,
                err_msg="%s moved under a +%g offset" % (col, offset),
            )

    def test_rescaling_the_whole_frame_leaves_the_ranks_alone(self):
        df = _frame()
        base = _ranks(df)
        out = _ranks(df * 1e8)
        for col in ["archetypal_rank", "prototypical_rank"]:
            np.testing.assert_allclose(base[col].to_numpy(),
                                       out[col].to_numpy(), atol=1e-12)

    def test_signal_beyond_the_float32_mantissa_survives(self):
        """
        A feature whose values differ only in the 8th significant digit, on top
        of a large offset. In float32 the spread collapses to exactly zero and
        the column is dropped; in float64 it is preserved.
        """
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
        # Shuffled, because a strictly monotonic column is dropped earlier as a
        # suspected row index, which would mask what this test is checking.
        df["fine"] = 1000.0 + rng.permutation(np.linspace(0, 3e-6, 30))

        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, n_prototypes=5,
                         dtype="float32", random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df)

        kept = [c for c, k in zip(dt.feature_columns_, dt.keep_mask_) if k]
        assert "fine" in kept, (
            "the fine-grained column was dropped as constant; its signal was "
            "lost to the working dtype before scaling"
        )

    def test_the_working_dtype_is_a_memory_knob_not_a_results_knob(self):
        """
        Scaling happens in float64 and only the scaled result is cast, so the
        two dtype settings must agree. Before v0.8.0 they did not: float32 threw
        away the input's precision before anything was computed.
        """
        df = _frame()
        a = _ranks(df, dtype="float32")
        b = _ranks(df, dtype="float64")
        for col in RANKS:
            np.testing.assert_allclose(
                a[col].to_numpy(), b[col].to_numpy(), atol=1e-6,
                err_msg="%s differs between float32 and float64" % col,
            )


class TestExtremeMagnitudes:
    def test_very_large_values_no_longer_overflow(self):
        """float32 tops out near 3.4e38; these used to become inf."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(np.abs(rng.normal(size=(25, 5))) * 1e300,
                          columns=list("abcde"))
        out = _ranks(df)
        for col in ["archetypal_rank", "prototypical_rank"]:
            assert np.isfinite(out[col]).all()

    def test_a_mix_of_extreme_and_ordinary_features(self):
        df = _frame(30, 4)
        df["tiny"] = df["f0"] * 1e-200
        df["huge"] = df["f1"] * 1e300
        out = _ranks(df)
        for col in RANKS:
            values = out[col].to_numpy()
            assert np.isfinite(values).all()
            assert values.min() >= -1e-9 and values.max() <= 1 + 1e-9

    def test_values_below_the_scaler_resolution_raise_a_clear_error(self):
        """
        MinMaxScaler treats a range under about 2e-15 as constant. Every column
        is then dropped. Before v0.8.0 this surfaced as a complaint about
        nmf_rank, which points at the wrong thing entirely.
        """
        rng = np.random.default_rng(0)
        df = pd.DataFrame(np.abs(rng.normal(size=(25, 5))) * 1e-300,
                          columns=list("abcde"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(DataTypicalError) as excinfo:
                DataTypical(archetypal_method="nmf", nmf_rank=3,
                            random_state=0).fit(df)
        message = str(excinfo.value)
        assert "below the scaler" in message
        assert "rescale" in message.lower()
        for name in "abcde":
            assert name in message


class TestDroppedColumnReporting:
    """Dropping a feature changes the analysis, so it is never silent."""

    def test_a_truly_constant_column_is_reported(self):
        df = _frame(30, 4)
        df["flat"] = 7.0
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        with pytest.warns(UserWarning, match="Dropped constant feature columns"):
            dt.fit(df)
        assert "flat" in dt.dropped_columns_

    def test_reporting_does_not_depend_on_verbose(self):
        """The v0.7.7 warning only fired under verbose, which hid it."""
        df = _frame(30, 4)
        df["flat"] = 7.0
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, verbose=False,
                         random_state=0)
        with pytest.warns(UserWarning, match="Dropped constant"):
            dt.fit(df)

    def test_a_below_resolution_column_is_named_separately(self):
        """It varies, so calling it constant would be a lie."""
        df = _frame(30, 4)
        rng = np.random.default_rng(1)
        df["subtle"] = rng.permutation(np.linspace(0.0, 1e-200, 30))
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        with pytest.warns(UserWarning, match="less than the scaler can resolve"):
            dt.fit(df)
        assert "subtle" in dt.dropped_columns_

    def test_the_two_kinds_are_distinguished(self):
        df = _frame(30, 4)
        rng = np.random.default_rng(2)
        df["flat"] = 7.0
        df["subtle"] = rng.permutation(np.linspace(0.0, 1e-200, 30))
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.fit(df)
        messages = [str(w.message) for w in caught]
        constant = [m for m in messages if "Dropped constant" in m]
        subtle = [m for m in messages if "less than the scaler can resolve" in m]
        assert constant and "flat" in constant[0] and "subtle" not in constant[0]
        assert subtle and "subtle" in subtle[0] and "flat" not in subtle[0]

    def test_a_clean_frame_warns_about_nothing(self):
        df = _frame(30, 4)
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.fit(df)
        assert not [w for w in caught if "Dropped" in str(w.message)]
        assert dt.dropped_columns_ == []
