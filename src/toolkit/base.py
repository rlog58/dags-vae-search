from abc import ABC, abstractmethod
from typing import Dict

import igraph as ig
import networkx as nx


class BaseDag(ABC):
    def __init__(
            self,
            num_vertices: int,
            validation: bool = True,
    ):
        if num_vertices <= 0:
            raise ValueError("`num_vertices` must be greater than 0, got {}".format(num_vertices))

        self._num_vertices = num_vertices
        self.validation = validation

    @property
    def num_vertices(self) -> int:
        return self._num_vertices

    @abstractmethod
    def is_valid_graph(self, graph: ig.Graph, quiet: bool = True) -> bool:
        pass

    @abstractmethod
    def is_valid_dict(self, graph: ig.Graph, quiet: bool = True) -> bool:
        pass

    @abstractmethod
    def from_dict_to_graph(self, pydict: Dict) -> ig.Graph:
        pass

    @abstractmethod
    def from_graph_to_dict(self, graph: ig.Graph) -> Dict:
        pass

    @abstractmethod
    def from_graph_to_nxgraph(self, graph: ig.Graph) -> nx.DiGraph:
        pass

    @abstractmethod
    def graph_equals(self, graph1: ig.Graph, graph2: ig.Graph) -> bool:
        pass

    def graph_drop_isolates(self, graph: ig.Graph, copy: bool = False, validation: bool = None) -> ig.Graph:
        if validation is None:
            if self.validation:
                self.is_valid_graph(graph)
        elif validation:
            self.is_valid_graph(graph)

        cur_graph = graph.copy() if copy else graph

        cur_graph.delete_vertices([v.index for v in cur_graph.vs.select(_outdegree_eq=0, _indegree_eq=0)])

        return cur_graph
