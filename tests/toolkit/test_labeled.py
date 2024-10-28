from typing import Dict

import igraph as ig
import pytest

from src.toolkit.labeled import LabeledDag, LABEL_KEY


@pytest.fixture
def labeled_graph_toolkit() -> LabeledDag:
    return LabeledDag(num_vertices=5, label_cardinality=5)


@pytest.fixture
def graph_unlabeled() -> ig.Graph:
    g = ig.Graph(directed=True)
    g.add_vertices(5)
    g.add_edges(
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 4)
        ]
    )
    return g


@pytest.fixture
def graph_labeled() -> ig.Graph:
    g = ig.Graph(directed=True)
    g.add_vertices(5)
    g.add_edges(
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 4)
        ]
    )

    g.vs()[LABEL_KEY] = [0, 1, 2, 3, 4]

    return g


@pytest.fixture
def pydict() -> Dict:
    graph_dict = {
        'l0': 0,
        'l1': 1,
        'l2': 2,
        'l3': 3,
        'l4': 4,
        'e0': [],
        'e1': [1],
        'e2': [1, 1],
        'e3': [0, 0, 1],
        'e4': [0, 0, 0, 1],
    }

    return graph_dict


def test_is_valid_dict(
        labeled_graph_toolkit: LabeledDag,
        pydict: Dict
):
    assert True == labeled_graph_toolkit.is_valid_dict(pydict, quiet=False)


def test_is_valid_graph(
        labeled_graph_toolkit: LabeledDag,
        graph_unlabeled: ig.Graph,
        graph_labeled: ig.Graph,
):
    assert False == labeled_graph_toolkit.is_valid_graph(graph_unlabeled, quiet=True)
    assert True == labeled_graph_toolkit.is_valid_graph(graph_labeled, quiet=False)


def test_from_dict_to_graph(
        labeled_graph_toolkit: LabeledDag,
        pydict: Dict,
        graph_labeled: ig.Graph,
):
    new_graph = labeled_graph_toolkit.from_dict_to_graph(pydict)

    assert labeled_graph_toolkit.graph_equals(graph_labeled, new_graph)


def test_graph_to_dict(
        labeled_graph_toolkit: LabeledDag,
        graph_labeled: ig.Graph,
        pydict: Dict,
):
    new_pydict = labeled_graph_toolkit.from_graph_to_dict(graph_labeled)

    assert pydict == new_pydict


def test_drop_isolates(labeled_graph_toolkit: LabeledDag, graph_unlabeled: ig.Graph):
    graph_without_isolates = labeled_graph_toolkit.graph_drop_isolates(graph_unlabeled, copy=True)

    assert [0, 1, 2, 3, 4] == graph_unlabeled.vs.indices
    assert [0, 1, 2, 3, 4] == graph_without_isolates.vs.indices

def test_generate_random_graph_erdos_renyi(labeled_graph_toolkit: LabeledDag):

    graph_generated = labeled_graph_toolkit.generate_random_graph_erdos_renyi(
        num_edges=10,
        accept_isolates=False,
        seed=42,
    )

    assert 5 == graph_generated.vcount()
    assert True == labeled_graph_toolkit.is_valid_graph(graph_generated, quiet=False)