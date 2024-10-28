import igraph as ig
import pytest

from src.problem.bn.bnlearn import BNLearnWrapper
from src.toolkit.labeled import LabeledDag


@pytest.fixture
def graph() -> ig.Graph:
    labeled_dag_toolkit = LabeledDag(num_vertices=8, label_cardinality=8)

    # [
    #  [0],
    #  [1, 1],
    #  [2, 0, 0],
    #  [3, 0, 0, 0],
    #  [4, 0, 1, 0, 0],
    #  [5, 1, 1, 0, 0, 0],
    #  [6, 0, 1, 0, 0, 1, 0],
    #  [7, 0, 0, 0, 1, 1, 1, 0]
    # ]
    graph_dict = {
        'l0': 0,
        'l1': 1,
        'l2': 2,
        'l3': 3,
        'l4': 4,
        'l5': 5,
        'l6': 6,
        'l7': 7,
        'e0': [],
        'e1': [1],
        'e2': [0, 0],
        'e3': [0, 0, 0],
        'e4': [0, 1, 0, 0],
        'e5': [1, 1, 0, 0, 0],
        'e6': [0, 1, 0, 0, 1, 0],
        'e7': [0, 0, 0, 1, 1, 1, 0],
    }

    graph = labeled_dag_toolkit.from_dict_to_graph(graph_dict)

    return graph


def test_score(
        graph: ig.Graph
):
    evaluator = BNLearnWrapper("asia", "bic")

    result = evaluator.score(graph)

    print(result)

    assert -13331.093616667435 == pytest.approx(result, abs=1e-5)
