"""
Edge-list handling for graph topology features.

Through v0.7.7 an edge naming a node outside `range(n_nodes)` was accepted in
silence: NetworkX simply added the extra node, so every global measure was
normalised over a larger graph than the node features described. Pagerank over
the requested nodes then summed to less than one, and betweenness and closeness
were computed against phantom nodes, while the returned columns looked
perfectly ordinary.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from datatypical import DataTypical


def _dt(features):
    return DataTypical(graph_topology_features=features, random_state=0)


class TestTopologyValues:
    """Checked against graphs whose values can be worked out by hand."""

    def test_degree_on_a_path_with_an_isolate(self):
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        topo = _dt(["degree"])._compute_graph_topology_features(edges, 5)
        assert list(topo["degree"]) == [1, 2, 2, 1, 0]

    def test_clustering_on_a_triangle_with_a_pendant(self):
        edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3]])
        topo = _dt(["clustering"])._compute_graph_topology_features(edges, 5)
        got = [round(v, 6) for v in topo["clustering"]]
        # node 0 has degree 3 and one of its three neighbour pairs is joined
        assert got == [round(1 / 3, 6), 1.0, 1.0, 0.0, 0.0]

    def test_triangles_on_a_triangle_with_a_pendant(self):
        edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3]])
        topo = _dt(["triangles"])._compute_graph_topology_features(edges, 5)
        assert list(topo["triangles"]) == [1, 1, 1, 0, 0]

    def test_pagerank_sums_to_one_over_the_requested_nodes(self):
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        topo = _dt(["pagerank"])._compute_graph_topology_features(edges, 4)
        assert float(np.sum(topo["pagerank"])) == pytest.approx(1.0)

    def test_isolated_nodes_get_defined_values(self):
        edges = np.array([[0, 1]])
        topo = _dt(["degree", "clustering", "closeness"]) \
            ._compute_graph_topology_features(edges, 4)
        assert list(topo["degree"]) == [1, 1, 0, 0]
        assert np.isfinite(topo["closeness"]).all()


class TestEdgeOrientation:
    def test_both_orientations_agree(self):
        tall = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
        wide = tall.T
        dt = _dt(["degree"])
        a = list(dt._compute_graph_topology_features(tall, 5)["degree"])
        b = list(dt._compute_graph_topology_features(wide, 5)["degree"])
        assert a == b == [2, 2, 2, 2, 2]

    def test_an_ambiguous_two_by_two_array_says_which_reading_it_used(self):
        """Two edges as rows and two nodes as columns are the same array."""
        with pytest.warns(RuntimeWarning, match="ambiguous"):
            _dt(["degree"])._compute_graph_topology_features(
                np.array([[0, 1], [0, 2]]), 3)

    def test_a_larger_array_is_not_called_ambiguous(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _dt(["degree"])._compute_graph_topology_features(
                np.array([[0, 1], [1, 2], [2, 0]]), 3)
        assert not [w for w in caught if "ambiguous" in str(w.message)]

    @pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((3, 3, 2)), np.zeros((3, 5))])
    def test_a_wrongly_shaped_edge_array_is_rejected(self, bad):
        with pytest.raises(ValueError, match="2D array"):
            _dt(["degree"])._compute_graph_topology_features(bad, 5)


class TestNodeIndexValidation:
    def test_an_index_past_the_end_is_rejected(self):
        edges = np.array([[0, 1], [1, 2], [2, 99]])
        with pytest.raises(ValueError, match="outside range"):
            _dt(["pagerank"])._compute_graph_topology_features(edges, 3)

    def test_the_offending_index_is_named(self):
        edges = np.array([[0, 1], [1, 42]])
        with pytest.raises(ValueError) as excinfo:
            _dt(["degree"])._compute_graph_topology_features(edges, 3)
        assert "42" in str(excinfo.value)

    def test_a_negative_index_is_rejected(self):
        edges = np.array([[0, 1], [-3, 2]])
        with pytest.raises(ValueError, match="outside range"):
            _dt(["degree"])._compute_graph_topology_features(edges, 3)

    def test_the_last_valid_index_is_accepted(self):
        """Off-by-one guard: n_nodes - 1 must still be allowed."""
        edges = np.array([[0, 1], [1, 4]])
        topo = _dt(["degree"])._compute_graph_topology_features(edges, 5)
        assert list(topo["degree"]) == [1, 2, 0, 0, 1]

    def test_the_public_graph_api_surfaces_the_error(self):
        rng = np.random.default_rng(0)
        features = pd.DataFrame(rng.normal(size=(5, 3)), columns=list("xyz"))
        edges = np.array([[0, 1], [1, 2], [2, 77]])
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        with pytest.raises(ValueError, match="outside range"):
            dt.fit_transform_graph(features, edges=edges)

    def test_a_valid_graph_still_fits_end_to_end(self):
        rng = np.random.default_rng(1)
        features = pd.DataFrame(rng.normal(size=(12, 4)), columns=list("wxyz"))
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
                          [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6]])
        dt = DataTypical(archetypal_method="nmf", nmf_rank=2, n_prototypes=3,
                         random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = dt.fit_transform_graph(features, edges=edges)
        assert len(out) == 12
        assert dt.graph_topology_df_ is not None
