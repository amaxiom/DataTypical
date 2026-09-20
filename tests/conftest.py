"""
Shared fixtures for the DataTypical test suite.

Kept deliberately small and fast: most tests want a handful of rows and a
handful of features, because the expensive paths (Shapley permutations,
formative walks) are exercised by a few targeted tests rather than by every
test in the file.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The package ships as two flat modules at the project root, not a package
# directory, so the root has to be importable from tests/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # never open a window during tests


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test leaves behind, so runs stay memory-flat."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def _make_tabular(n=40, d=6, seed=0, with_age=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n, d)),
        columns=["f%d" % i for i in range(d)],
    )
    if with_age:
        df["Age"] = rng.integers(30, 85, size=n).astype(float)
    return df


@pytest.fixture
def df_small():
    """40 rows, 6 numeric features plus a numeric Age column."""
    return _make_tabular(40, 6, seed=0)


@pytest.fixture
def df_tiny():
    """12 rows, 4 features. For paths where speed matters more than realism."""
    return _make_tabular(12, 4, seed=1)


@pytest.fixture
def df_wide():
    """More than 20 features, so the ConvexHull fallback is out of reach."""
    return _make_tabular(30, 24, seed=2)


@pytest.fixture
def make_tabular():
    """Factory, for tests that need a specific shape."""
    return _make_tabular


@pytest.fixture
def corpus():
    """A small text corpus with two clear topics."""
    return [
        "protein folding kinetics in aqueous solution",
        "protein structure prediction with deep learning",
        "enzyme catalysis and protein binding affinity",
        "galaxy rotation curves and dark matter haloes",
        "stellar nucleosynthesis in massive stars",
        "cosmic microwave background anisotropy maps",
        "protein ligand docking free energy",
        "supernova remnants and interstellar shocks",
        "crystal structure of a membrane protein",
        "redshift surveys of distant galaxies",
    ]


@pytest.fixture
def text_metadata(corpus):
    return pd.DataFrame(
        {
            "Year": np.arange(2010, 2010 + len(corpus)).astype(float),
            "Citations": np.linspace(1, 100, len(corpus)),
        }
    )


@pytest.fixture
def graph_data():
    """Node features plus an edge list for a 12-node graph."""
    rng = np.random.default_rng(7)
    n_nodes = 12
    features = pd.DataFrame(
        rng.normal(size=(n_nodes, 4)),
        columns=["x%d" % i for i in range(4)],
    )
    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0], [0, 2],
            [4, 5], [5, 6], [6, 4], [7, 8], [8, 9],
            [9, 10], [10, 11], [11, 7], [3, 4], [6, 7],
        ]
    )
    return features, edges


@pytest.fixture
def fitted_shapley(df_small):
    """A fitted estimator with explanations and formative attributions."""
    from datatypical import DataTypical

    dt = DataTypical(
        shapley_mode=True,
        shapley_n_permutations=6,
        shapley_compute_formative=True,
        nmf_rank=3,
        n_prototypes=4,
        archetypal_method="auto",
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = dt.fit_transform(df_small)
    return dt, results
