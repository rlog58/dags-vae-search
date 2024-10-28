from typing import List

import igraph as ig
import pyarrow as pa

from src.toolkit.labeled import LabeledDag, LABEL_KEY

POSITION_KEY = 'position'


class PaceDag(LabeledDag):
    def __init__(
            self,
            num_vertices: int,
            label_cardinality: int,
            graph_label_key: str = LABEL_KEY,
            graph_position_key: str = POSITION_KEY,
            graph_label_input: int = 0,
            graph_label_output: int = 1,
            graph_label_start: int = 2,
            dict_label_prefix: str = 'l',
            dict_edges_prefix: str = 'e',
            pyarrow_schema: pa.Schema = None,
            validation: bool = True
    ):
        super().__init__(
            num_vertices + 3,
            label_cardinality + 3,
            graph_label_key,
            dict_label_prefix,
            dict_edges_prefix,
            pyarrow_schema,
            validation
        )

        self._graph_position_key = graph_position_key

        self._graph_attributes = [
            self._graph_label_key,
            self._graph_position_key,
        ]

        self._graph_label_input = graph_label_input
        self._graph_label_output = graph_label_output
        self._graph_label_start = graph_label_start

    @property
    def graph_position_key(self) -> str:
        return self._graph_position_key

    @property
    def graph_label_input(self) -> int:
        return self._graph_label_input

    @property
    def graph_label_output(self) -> int:
        return self._graph_label_output

    @property
    def graph_label_start(self) -> int:
        return self._graph_label_start

    @staticmethod
    def compute_graph_positions(graph: ig.Graph) -> List[int]:

        graph_positions = graph.topological_sorting()

        return graph_positions

    def is_valid_graph(self, graph: ig.Graph, quiet=True) -> bool:

        if not super().is_valid_graph(graph, quiet):
            return False

        graph_positions = self.compute_graph_positions(graph)

        if graph_positions != graph.vs[self.graph_position_key]:
            if quiet:
                return False
            else:
                raise AssertionError(
                    f"Graph positions {graph_positions} expected, got instead {graph.vs[self.graph_position_key]}")

        n_input = 0
        n_output = 0
        n_start = 0

        for vertex in graph.vs:
            if vertex[self.graph_label_key] == self.graph_label_input:
                n_input += 1
            elif vertex[self.graph_label_key] == self.graph_label_output:
                n_output += 1
            elif vertex[self.graph_label_key] == self.graph_label_start:
                n_start += 1

        if n_input != 1:
            if quiet:
                return False
            else:
                raise AssertionError(
                    f"Only one input vertex with {self.graph_label_input} label expected, got instead {n_input}")

        if n_output != 1:
            if quiet:
                return False
            else:
                raise AssertionError(
                    f"Only one output vertex with {self.graph_label_output} label expected, got instead {n_output}")

        if n_start != 1:
            if quiet:
                return False
            else:
                raise AssertionError(
                    f"Only one start vertex with {self.graph_label_start} label expected, got instead {n_start}")

        return True

    def from_labeled_graph_to_graph(self, labeled_graph: ig.Graph) -> ig.Graph:
        if self.validation:
            assert self.num_vertices - 3 == labeled_graph.vcount(), f"Expected {self.num_vertices - 3}, got instead {labeled_graph.vcount()}"

        graph = ig.Graph(directed=True)
        graph.add_vertices(self.num_vertices)

        # Set start, input, output vertex types
        graph.vs[0][self.graph_label_key] = self.graph_label_start
        graph.vs[1][self.graph_label_key] = self.graph_label_input

        output_vertex_id = self.num_vertices - 1
        graph.vs[output_vertex_id][self.graph_label_key] = self.graph_label_output

        # Add edges from start to input
        graph.add_edge(0, 1)

        for vertex_id in range(self.num_vertices - 3):
            vertex: ig.Vertex = labeled_graph.vs[vertex_id]

            vertex_label = vertex[self.graph_label_key] + 3
            graph.vs[vertex_id + 2][self.graph_label_key] = vertex_label

            vertex_connections = [(v.index + 2, vertex_id + 2) for v in vertex.predecessors()]

            if len(vertex_connections) == 0:
                graph.add_edge(1, vertex_id + 2)
                continue

            graph.add_edges(vertex_connections)

        # Add edge from last nodes to output
        end_vertices = [vertex.index for vertex in graph.vs.select(_outdegree_eq=0) if vertex.index != output_vertex_id]

        for vertex_id in end_vertices:
            graph.add_edge(vertex_id, output_vertex_id)

        graph.vs()[POSITION_KEY] = self.compute_graph_positions(graph)

        return graph

    def from_graph_to_labeled_graph(self, graph: ig.Graph) -> ig.Graph:
        if self.validation:
            self.is_valid_graph(graph, quiet=False)

        labeled_graph = ig.Graph(directed=True)
        labeled_graph.add_vertices(self.num_vertices - 3)

        for vertex_id in range(2, self.num_vertices - 1):
            labeled_graph.vs[vertex_id - 2][self.graph_label_key] = graph.vs[vertex_id][self.graph_label_key] - 3

            if vertex_id == self.graph_label_start:
                continue

            for incoming_vertex_id in range(2, vertex_id + 2):
                if graph.are_adjacent(incoming_vertex_id, vertex_id):
                    labeled_graph.add_edge(incoming_vertex_id - 2, vertex_id - 2)

        return labeled_graph
