import igraph as ig
import pytest

from src.encoders.pace_utils import PaceDag, POSITION_KEY
from src.toolkit.labeled import LabeledDag, LABEL_KEY


@pytest.fixture
def labeled_graph_toolkit() -> LabeledDag:
    return LabeledDag(num_vertices=5, label_cardinality=5)


@pytest.fixture
def pace_graph_toolkit() -> PaceDag:
    return PaceDag(num_vertices=5, label_cardinality=5)


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
def graph_labeled_pace() -> ig.Graph:
    g = ig.Graph(directed=True)
    g.add_vertices(8)
    g.add_edges(
        [
            # start -> input
            (0, 1),
            # input -> in_degree == 0
            (1, 2),
            # origin graph edges + 2 in index
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (5, 6),
            # out_degree == 0 -> output
            (6, 7),
        ]
    )

    g.vs()[LABEL_KEY] = [2, 0, 3, 4, 5, 6, 7, 1]
    g.vs()[POSITION_KEY] = [0, 1, 2, 3, 4, 5, 6, 7]

    return g


def test_is_valid_graph(
        pace_graph_toolkit: PaceDag,
        graph_labeled_pace: ig.Graph
):
    assert True == pace_graph_toolkit.is_valid_graph(graph_labeled_pace, quiet=False)


def test_graph_to_labeled_graph(
        labeled_graph_toolkit: LabeledDag,
        pace_graph_toolkit: PaceDag,
        graph_labeled: ig.Graph,
        graph_labeled_pace: ig.Graph,
):
    new_labeled_graph = pace_graph_toolkit.from_graph_to_labeled_graph(graph_labeled_pace)

    assert True == labeled_graph_toolkit.graph_equals(graph_labeled, new_labeled_graph)


def test_from_graph_to_labeled_graph(
        labeled_graph_toolkit: LabeledDag,
        pace_graph_toolkit: PaceDag,
        graph_labeled: ig.Graph,
        graph_labeled_pace: ig.Graph,
):
    new_graph_labeled_pace = pace_graph_toolkit.from_labeled_graph_to_graph(graph_labeled)

    assert True == pace_graph_toolkit.graph_equals(graph_labeled_pace, new_graph_labeled_pace)
