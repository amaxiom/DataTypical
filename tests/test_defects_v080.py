"""
Regression tests for the three defects fixed in v0.8.0.

Each test reproduces the failure as it was originally observed, so that a
future change that reintroduces it fails here rather than in someone's
analysis.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import datatypical as dtmod
from datatypical import ConfigError, DataTypical


# ---------------------------------------------------------------------------
# Defect 1: UnboundLocalError in stereotypical Shapley explanations
#
# explain_stereotypical_features was defined inside the core-samples branch of
# _fit_shapley_explanations but also called from the secondary-samples branch.
# core_samples comes from self._union_core_samples when subsample_indices is
# supplied, and need not intersect the requested samples, so it can be empty
# while secondary_samples is not.
# ---------------------------------------------------------------------------
class TestStereotypeShapleyScoping:
    def test_shapley_top_n_below_n_samples(self, make_tabular):
        """The original reproduction: top_n=3 on 160 rows used to raise."""
        df = make_tabular(160, 24, seed=0)
        dt = DataTypical(
            stereotype_column="Age",
            stereotype_target=55,
            shapley_mode=True,
            shapley_top_n=3,
            shapley_compute_formative=True,
            archetypal_method="aa",
            shapley_n_permutations=4,
            random_state=0,
        )
        out = dt.fit_transform(df)
        assert "stereotypical_rank" in out.columns
        assert dt.Phi_stereotypical_explanations_ is not None

    @pytest.mark.parametrize("top_n", [1, 2, 5, 0.25])
    def test_a_range_of_subsample_sizes(self, make_tabular, top_n):
        df = make_tabular(40, 6, seed=3)
        dt = DataTypical(
            stereotype_column="Age",
            stereotype_target="max",
            shapley_mode=True,
            shapley_top_n=top_n,
            shapley_compute_formative=False,
            shapley_n_permutations=4,
            archetypal_method="auto",
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        assert dt.Phi_stereotypical_explanations_ is not None

    def test_top_n_equal_to_n_samples_still_works(self, df_small):
        """The configuration that masked the bug must keep working."""
        dt = DataTypical(
            stereotype_column="Age",
            stereotype_target=55,
            shapley_mode=True,
            shapley_top_n=len(df_small),
            shapley_compute_formative=False,
            shapley_n_permutations=4,
            archetypal_method="auto",
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df_small)
        assert dt.Phi_stereotypical_explanations_ is not None

    def test_explanations_are_finite_for_both_tiers(self, make_tabular):
        """Core and secondary tiers must both produce usable numbers."""
        df = make_tabular(60, 8, seed=4)
        dt = DataTypical(
            stereotype_column="Age",
            stereotype_target=55,
            shapley_mode=True,
            shapley_top_n=4,
            shapley_compute_formative=False,
            shapley_n_permutations=4,
            archetypal_method="auto",
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit_transform(df)
        phi = dt.Phi_stereotypical_explanations_
        assert np.isfinite(phi).all()
        assert np.abs(phi).sum() > 0


# ---------------------------------------------------------------------------
# Defect 2: silent substitution of NMF for archetypal analysis
# ---------------------------------------------------------------------------
@pytest.fixture
def no_pcha(monkeypatch):
    """Simulate an environment where py_pcha is not installed."""
    monkeypatch.setattr(dtmod, "PCHA", None)


class TestArchetypalBackend:
    def test_aa_records_pcha_backend(self, df_wide):
        pytest.importorskip("py_pcha")
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        dt.fit(df_wide)
        assert dt.archetypal_backend_ == "pcha"

    def test_backend_is_serialised_into_settings(self, df_wide):
        pytest.importorskip("py_pcha")
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        dt.fit(df_wide)
        assert dt.settings_["archetypal_backend"] == "pcha"
        assert dt.settings_["archetypal_method"] == "aa"

    def test_aa_raises_when_pcha_missing(self, df_wide, no_pcha):
        """The headline fix: no silent downgrade."""
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        with pytest.raises(ConfigError) as excinfo:
            dt.fit(df_wide)
        message = str(excinfo.value)
        assert "py_pcha" in message
        assert "auto" in message  # the error points at the way forward

    def test_aa_raises_when_pcha_fails(self, df_wide, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic PCHA failure")

        monkeypatch.setattr(dtmod, "PCHA", boom)
        dt = DataTypical(archetypal_method="aa", nmf_rank=3, random_state=0)
        with pytest.raises(ConfigError) as excinfo:
            dt.fit(df_wide)
        assert "synthetic PCHA failure" in str(excinfo.value)

    def test_aa_raises_when_k_eff_below_two(self, make_tabular):
        pytest.importorskip("py_pcha")
        df = make_tabular(20, 5, seed=0, with_age=False)
        dt = DataTypical(archetypal_method="aa", nmf_rank=1, random_state=0)
        with pytest.raises(ConfigError) as excinfo:
            dt.fit(df)
        assert "at least 2 archetypes" in str(excinfo.value)

    def test_auto_warns_and_falls_back_to_nmf(self, df_wide, no_pcha):
        """24 features, so ConvexHull is out of reach and NMF is the landing."""
        dt = DataTypical(archetypal_method="auto", nmf_rank=3, random_state=0)
        with pytest.warns(RuntimeWarning, match="py_pcha is not installed"):
            dt.fit(df_wide)
        assert dt.archetypal_backend_ == "nmf"

    def test_auto_reaches_convexhull_in_low_dimensions(self, df_tiny, no_pcha):
        dt = DataTypical(archetypal_method="auto", nmf_rank=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.fit(df_tiny)
        assert dt.archetypal_backend_ == "convexhull"

    def test_auto_warns_when_k_eff_below_two(self, make_tabular):
        pytest.importorskip("py_pcha")
        df = make_tabular(20, 5, seed=0, with_age=False)
        dt = DataTypical(archetypal_method="auto", nmf_rank=1, random_state=0)
        with pytest.warns(RuntimeWarning, match="at least 2 archetypes"):
            dt.fit(df)
        assert dt.archetypal_backend_ in {"convexhull", "nmf"}

    def test_auto_warns_when_pcha_itself_fails(self, df_wide, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic PCHA failure")

        monkeypatch.setattr(dtmod, "PCHA", boom)
        dt = DataTypical(archetypal_method="auto", nmf_rank=3, random_state=0)
        with pytest.warns(RuntimeWarning, match="PCHA failed"):
            dt.fit(df_wide)
        assert dt.archetypal_backend_ == "nmf"

    def test_nmf_is_quiet_and_records_its_backend(self, df_tiny):
        dt = DataTypical(archetypal_method="nmf", nmf_rank=3, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt.fit(df_tiny)
        assert dt.archetypal_backend_ == "nmf"
        assert not [w for w in caught if "fell back to NMF" in str(w.message)]

    def test_convexhull_failure_falls_through_to_nmf(
        self, df_tiny, no_pcha, monkeypatch
    ):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic hull failure")

        monkeypatch.setattr(dtmod, "ConvexHull", boom)
        dt = DataTypical(archetypal_method="auto", nmf_rank=3, random_state=0)
        with pytest.warns(RuntimeWarning, match="fell back to NMF"):
            dt.fit(df_tiny)
        assert dt.archetypal_backend_ == "nmf"

    def test_convexhull_absent_falls_through_to_nmf(
        self, df_tiny, no_pcha, monkeypatch
    ):
        monkeypatch.setattr(dtmod, "ConvexHull", None)
        dt = DataTypical(archetypal_method="auto", nmf_rank=3, random_state=0)
        with pytest.warns(RuntimeWarning, match="fell back to NMF"):
            dt.fit(df_tiny)
        assert dt.archetypal_backend_ == "nmf"

    @pytest.mark.parametrize("bad", ["pcha", "AA", "convexhull", ""])
    def test_unknown_method_is_rejected(self, df_tiny, bad):
        with pytest.raises(ValueError, match="archetypal_method"):
            DataTypical(archetypal_method=bad, random_state=0).fit(df_tiny)

    def test_fast_mode_defaults_to_nmf(self, df_tiny):
        dt = DataTypical(fast_mode=True, nmf_rank=3, random_state=0)
        dt.fit(df_tiny)
        assert dt.archetypal_method == "nmf"
        assert dt.archetypal_backend_ == "nmf"

    def test_publication_mode_defaults_to_aa(self, df_wide):
        pytest.importorskip("py_pcha")
        dt = DataTypical(fast_mode=False, nmf_rank=3, random_state=0)
        dt.fit(df_wide)
        assert dt.archetypal_method == "aa"
        assert dt.archetypal_backend_ == "pcha"

    def test_backend_is_none_before_fitting(self):
        assert DataTypical().archetypal_backend_ is None

    def test_verbose_labels_each_method(self, df_tiny, capsys):
        DataTypical(
            archetypal_method="auto", nmf_rank=3, verbose=True, random_state=0
        ).fit(df_tiny)
        out = capsys.readouterr().out
        assert "Archetypal Analysis" in out


# ---------------------------------------------------------------------------
# Defect 5: a categorical stereotype column raised deep inside pandas
# ---------------------------------------------------------------------------
class TestStereotypeColumnValidation:
    def test_yes_no_column_names_itself(self, df_small):
        df = df_small.copy()
        df["Menopause"] = np.where(np.arange(len(df)) % 2 == 0, "Yes", "No")
        with pytest.raises(ConfigError) as excinfo:
            DataTypical(stereotype_column="Menopause", random_state=0).fit(df)
        message = str(excinfo.value)
        assert "Menopause" in message
        assert "'Yes'" in message or "'No'" in message

    def test_error_arrives_at_fit_not_at_transform(self, df_small):
        """Before v0.8.0 this surfaced only once the fit was under way."""
        df = df_small.copy()
        df["Grade"] = ["low", "high"] * (len(df) // 2)
        dt = DataTypical(stereotype_column="Grade", random_state=0)
        with pytest.raises(ConfigError):
            dt.fit(df)

    def test_ordered_categorical_is_encoded_by_category_order(self, df_small):
        df = df_small.copy()
        stages = ["I", "II", "III", "IV"]
        df["Stage"] = pd.Categorical(
            np.take(stages, np.arange(len(df)) % 4),
            categories=stages,
            ordered=True,
        )
        dt = DataTypical(
            stereotype_column="Stage", stereotype_target="max", random_state=0
        )
        out = dt.fit_transform(df)
        ranks = out["stereotypical_rank"]
        assert ranks.notna().all()
        # Stage IV is the target, so those rows must rank top
        assert ranks[df["Stage"] == "IV"].min() == pytest.approx(1.0)

    def test_ordered_categorical_with_missing_values(self, df_small):
        df = df_small.copy()
        values = np.take(["low", "mid", "high"], np.arange(len(df)) % 3).astype(object)
        values[:3] = None
        df["Grade"] = pd.Categorical(
            values, categories=["low", "mid", "high"], ordered=True
        )
        out = DataTypical(stereotype_column="Grade", random_state=0).fit_transform(df)
        assert out["stereotypical_rank"].iloc[:3].eq(0.0).all()

    def test_unordered_categorical_is_rejected(self, df_small):
        df = df_small.copy()
        df["Site"] = pd.Categorical(np.take(["a", "b"], np.arange(len(df)) % 2))
        with pytest.raises(ConfigError, match="unordered"):
            DataTypical(stereotype_column="Site", random_state=0).fit(df)

    def test_numeric_strings_are_accepted(self, df_small):
        df = df_small.copy()
        df["AgeStr"] = [str(v) for v in df["Age"]]
        out = DataTypical(
            stereotype_column="AgeStr", stereotype_target=55.0, random_state=0
        ).fit_transform(df)
        assert out["stereotypical_rank"].notna().all()

    def test_boolean_column_is_accepted(self, df_small):
        df = df_small.copy()
        df["Flag"] = np.arange(len(df)) % 2 == 0
        out = DataTypical(
            stereotype_column="Flag", stereotype_target="max", random_state=0
        ).fit_transform(df)
        assert out["stereotypical_rank"].max() == pytest.approx(1.0)

    def test_missing_column_names_the_available_ones(self, df_small):
        with pytest.raises(ValueError, match="not found"):
            DataTypical(stereotype_column="NoSuchColumn", random_state=0).fit(df_small)

    def test_all_nan_column_warns_and_returns_zeros(self, df_small):
        df = df_small.copy()
        df["Age"] = np.nan
        dt = DataTypical(stereotype_column="Age", random_state=0)
        with pytest.warns(UserWarning, match="All stereotype values are NaN"):
            out = dt.fit_transform(df)
        assert out["stereotypical_rank"].eq(0.0).all()


# ---------------------------------------------------------------------------
# Defect 4 was withdrawn: stereotypical_rank matches the published equation.
# These tests pin that behaviour so it is not "fixed" by mistake later.
# ---------------------------------------------------------------------------
class TestStereotypicalRankSemantics:
    def test_rank_is_a_monotone_function_of_distance_to_target(self, df_small):
        """Methods 2.3.1, eq. 26: s = 1 - |y - tau| / max|y - tau|."""
        from scipy.stats import spearmanr

        dt = DataTypical(
            stereotype_column="Age", stereotype_target=55, random_state=0
        )
        out = dt.fit_transform(df_small)
        distance = -np.abs(df_small["Age"].to_numpy() - 55)
        rho = spearmanr(out["stereotypical_rank"], distance).statistic
        assert rho == pytest.approx(1.0)

    def test_equal_ages_receive_equal_ranks(self, df_small):
        dt = DataTypical(
            stereotype_column="Age", stereotype_target=55, random_state=0
        )
        out = dt.fit_transform(df_small)
        grouped = out.groupby(df_small["Age"].values)["stereotypical_rank"].nunique()
        assert grouped.eq(1).all()

    def test_exact_formula(self, df_small):
        dt = DataTypical(
            stereotype_column="Age", stereotype_target=55, random_state=0
        )
        out = dt.fit_transform(df_small)
        values = df_small["Age"].to_numpy(dtype=float)
        distance = np.abs(values - 55)
        expected = 1.0 - distance / distance.max()
        np.testing.assert_allclose(out["stereotypical_rank"].to_numpy(), expected)

    @pytest.mark.parametrize("target", ["min", "max"])
    def test_string_targets(self, df_small, target):
        dt = DataTypical(
            stereotype_column="Age", stereotype_target=target, random_state=0
        )
        out = dt.fit_transform(df_small)
        extreme = df_small["Age"].min() if target == "min" else df_small["Age"].max()
        top = out["stereotypical_rank"].idxmax()
        assert df_small.loc[top, "Age"] == extreme

    def test_constant_column_gives_uniform_rank(self, df_small):
        df = df_small.copy()
        df["Age"] = 50.0
        out = DataTypical(
            stereotype_column="Age", stereotype_target=50.0, random_state=0
        ).fit_transform(df)
        assert out["stereotypical_rank"].eq(1.0).all()
