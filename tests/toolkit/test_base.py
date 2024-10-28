from typing import Dict, List

import igraph as ig
import networkx as nx
import pytest

from src.toolkit.base import BaseDag


@pytest.fixture
def base_graph_toolkit() -> BaseDag:
    class TestDag(BaseDag):

        def is_valid_dict(self, graph: ig.Graph, quiet: bool = True) -> bool:
            pass

        def is_valid_graph(self, graph: ig.Graph, quiet: bool = True) -> bool:
            pass

        def graph_reindex(self, graph: ig.Graph, indexes: List[int]) -> ig.Graph:
            pass

        def from_dict_to_graph(self, pydict: Dict) -> ig.Graph:
            pass

        def from_graph_to_dict(self, graph: ig.Graph) -> Dict:
            pass

        def from_graph_to_nxgraph(self, graph: ig.Graph) -> nx.DiGraph:
            pass

        def graph_equals(self, graph1: ig.Graph, graph2: ig.Graph) -> bool:
            pass

    return TestDag(num_vertices=4)


@pytest.fixture
def graph() -> ig.Graph:
    g = ig.Graph(directed=True)
    g.add_vertices(6)
    g.add_edges(
        [
            (0, 1),
            (1, 2),
            (2, 5)
        ]
    )
    return g


def test_drop_isolates(base_graph_toolkit: BaseDag, graph: ig.Graph):
    graph_without_isolates = base_graph_toolkit.graph_drop_isolates(graph, copy=True)

    assert [0, 1, 2, 3, 4, 5] == graph.vs.indices
    assert [0, 1, 2, 3] == graph_without_isolates.vs.indices
