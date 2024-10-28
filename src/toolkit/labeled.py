import random
from typing import Dict

import igraph as ig
import networkx as nx
import pyarrow as pa

from src.toolkit.base import BaseDag

LABEL_KEY = 'type'


class LabeledDag(BaseDag):
    def __init__(
            self,
            num_vertices: int,
            label_cardinality: int,
            graph_label_key: str = LABEL_KEY,
            dict_label_prefix: str = 'l',
            dict_edges_prefix: str = 'e',
            pyarrow_schema: pa.Schema = None,
            validation: bool = True
    ):
        super().__init__(num_vertices, validation)

        if label_cardinality <= 0:
            raise ValueError("`label_cardinality` must be greater than 0, got {}".format(label_cardinality))

        self._label_cardinality = label_cardinality

        self._graph_label_key = graph_label_key

        self._graph_attributes = [
            self._graph_label_key,
        ]

        self._dict_label_prefix = dict_label_prefix
        self._dict_edges_prefix = dict_edges_prefix
        self._pyarrow_schema = pyarrow_schema

    @property
    def label_cardinality(self) -> int:
        return self._label_cardinality

    @property
    def graph_label_key(self) -> str:
        return self._graph_label_key

    @property
    def dict_label_prefix(self) -> str:
        return self._dict_label_prefix

    @property
    def dict_edges_prefix(self) -> str:
        return self._dict_edges_prefix

    def _label_field(self, vertex_id: int) -> str:
        if vertex_id < 0:
            raise ValueError("'vertex_id' must be >= 0, got {}".format(vertex_id))
        return f"{self.dict_label_prefix}{vertex_id}"

    def _edges_field(self, vertex_id: int) -> str:
        if vertex_id < 0:
            raise ValueError("'vertex_id' must be >= 0, got {}".format(vertex_id))
        return f"{self.dict_edges_prefix}{vertex_id}"

    def is_valid_dict(self, pydict: Dict, quiet: bool = True) -> bool:
        if not isinstance(pydict, dict):
            if quiet:
                return False
            else:
                raise ValueError("pydict must be a dict")

        label_fields_count = len([key for key in pydict.keys() if key.startswith(self.dict_label_prefix)])

        if self.num_vertices != label_fields_count:
            if quiet:
                return False
            else:
                raise AssertionError(f"Expected {self.num_vertices} label fields, got instead {label_fields_count}")

        edges_fields_count = len([key for key in pydict.keys() if key.startswith(self.dict_edges_prefix)])

        if self.num_vertices != edges_fields_count:
            if quiet:
                return False
            else:
                raise AssertionError(f"Expected {self.num_vertices} edges fields, got instead {edges_fields_count}")

        for vertex_id in range(self.num_vertices):
            if self._label_field(vertex_id) not in pydict:
                if quiet:
                    return False
                else:
                    raise ValueError(f"{self._label_field(vertex_id)} expected to be in pydict")
            label = pydict[self._label_field(vertex_id)]
            if not (0 <= label < self.label_cardinality):
                if quiet:
                    return False
                else:
                    raise AssertionError(
                        f"Label of vertex '{vertex_id}' expected to be in range 0 ... {self.label_cardinality - 1}, got instead {label}")
            if self._edges_field(vertex_id) not in pydict:
                if quiet:
                    return False
                else:
                    raise ValueError(f"'{self._edges_field(vertex_id)}' expected to be in pydict")
            elif len(pydict[self._edges_field(vertex_id)]) != vertex_id:
                if quiet:
                    return False
                else:
                    raise ValueError(f"{vertex_id} elements expected to be in '{self._edges_field(vertex_id)}'")

        return True

    @property
    def pyarrow_schema(self) -> pa.Schema:
        if self._pyarrow_schema is None:
            label_fields = [
                pa.field(self._label_field(i), pa.uint16(), nullable=False)
                for i in
                range(self.num_vertices)
            ]
            edges_fields = [
                pa.field(self._edges_field(i), pa.string(), nullable=False)
                for i in
                range(self.num_vertices)
            ]
            self._pyarrow_schema = pa.schema(label_fields + edges_fields)
        return self._pyarrow_schema

    def from_dict_to_graph(self, pydict: Dict, validation: bool = None) -> ig.Graph:
        if validation is None:
            if self.validation:
                self.is_valid_dict(pydict, quiet=False)
        elif validation:
            self.is_valid_dict(pydict, quiet=False)

        graph = ig.Graph(directed=True)

        graph.add_vertices(self.num_vertices)

        for vertex_id in range(self.num_vertices):

            vertex_label = pydict[self._label_field(vertex_id)]
            vertex_connections = [int(c) for c in pydict[self._edges_field(vertex_id)]]

            graph.vs[vertex_id][LABEL_KEY] = vertex_label

            for incoming_vertex_id in range(vertex_id):
                if vertex_connections[incoming_vertex_id] == 1:
                    graph.add_edge(incoming_vertex_id, vertex_id)

        return graph

    def from_graph_to_dict(self, graph: ig.Graph, validation: bool = None) -> Dict:
        if validation is None:
            if self.validation:
                self.is_valid_graph(graph, quiet=False)
        elif validation:
            self.is_valid_graph(graph, quiet=False)

        pydict = dict()

        topological_order = graph.topological_sorting()

        for i, vertex_id in enumerate(topological_order):
            vertex: ig.Vertex = graph.vs[vertex_id]
            vertex_label = vertex[self.graph_label_key]
            pydict[self._label_field(i)] = vertex_label

        for i, vertex_id in enumerate(topological_order):
            vertex: ig.Vertex = graph.vs[vertex_id]
            vertex_connections = [v.index for v in vertex.predecessors()]
            pydict[self._edges_field(i)] = [int(i in vertex_connections) for i in range(i)]

        return pydict

    def from_graph_to_dict_writable(self, graph: ig.Graph, validation: bool = None) -> Dict:
        pydict = self.from_graph_to_dict(graph, validation)

        for vertex_id in range(self.num_vertices):
            pydict[self._edges_field(vertex_id)] = "".join(str(connection) for connection in pydict[self._edges_field(vertex_id)])

        return pydict

    def is_valid_graph(self, graph: ig.Graph, quiet: bool = True) -> bool:
        if not graph.is_dag():
            if quiet:
                return False
            else:
                raise AssertionError("graph is not a dag")

        if graph.vcount() != self.num_vertices:
            if quiet:
                return False
            else:
                raise AssertionError(f"Graph expected to has {self.num_vertices} got instead {graph.vcount()}")

        graph_attributes = graph.vs().attribute_names()

        for attr in self._graph_attributes:
            if attr not in graph_attributes:
                if quiet:
                    return False
                else:
                    raise AssertionError(f"Attribute '{attr}' not found in graph")

        for i, label in enumerate(graph.vs[self.graph_label_key]):
            if not (0 <= label < self.label_cardinality):
                if quiet:
                    return False
                else:
                    raise AssertionError(
                        f"Label of vertex '{i}' expected to be in range 0 ... {self.label_cardinality - 1}, got instead {label}")

        return True

    def from_graph_to_nxgraph(self, graph: ig.Graph, validation: bool = None) -> nx.DiGraph:
        if validation is None:
            if self.validation:
                self.is_valid_graph(graph, quiet=False)
        elif validation:
            self.is_valid_graph(graph, quiet=False)

        nxgraph = nx.DiGraph()
        nxgraph.add_nodes_from(graph.vs.indices)
        nxgraph.add_edges_from(graph.get_edgelist())

        for vertex in graph.vs:
            vertex_id = vertex.index

            update_attributes = {attr: vertex[attr] for attr in self._graph_attributes}
            nxgraph.nodes[vertex_id].update(update_attributes)

        return nxgraph

    def graph_equals(
            self,
            graph1: ig.Graph,
            graph2: ig.Graph,
            attributes_match: bool = True,
            complete_isomorphism_check: bool = True,
            validation: bool = None,
    ) -> bool:
        validate = validation if validation is None else self.validation

        def check_attributes(attr_dict1: Dict, attr_dict2: Dict) -> bool:
            return all([attr_dict1[attr] == attr_dict2[attr] for attr in self._graph_attributes])

        nxgraph1 = self.from_graph_to_nxgraph(graph1, validate)
        nxgraph2 = self.from_graph_to_nxgraph(graph2, validate)

        if complete_isomorphism_check:
            if attributes_match:
                return nx.is_isomorphic(nxgraph1, nxgraph2, node_match=check_attributes)
            else:
                return nx.is_isomorphic(nxgraph1, nxgraph2)
        else:
            return nx.faster_could_be_isomorphic(nxgraph1, nxgraph2)

    def graph_sort_topological(self, graph: ig.Graph, validation: bool = None) -> ig.Graph:
        if validation is None:
            if self.validation:
                self.is_valid_graph(graph, quiet=False)
        elif validation:
            self.is_valid_graph(graph, quiet=False)

        topological_order = graph.topological_sorting()

        reindex_mapping = {vertex.index: new_index for vertex, new_index in zip(graph.vs, topological_order)}

        reindex_graph = ig.Graph(directed=True)
        reindex_graph.add_vertices(self.num_vertices)
        reindex_graph.add_edges([(reindex_mapping[v1], reindex_mapping[v2]) for v1, v2 in graph.get_edgelist()])
        for vertex in graph.vs:
            reindex_graph.vs[reindex_mapping[vertex.index]][self.graph_label_key] = vertex[self.graph_label_key]

        return reindex_graph

    def generate_random_graph_erdos_renyi(
            self,
            num_edges: int,
            label_random_method: str = "sample",
            accept_isolates: bool = True,
            accept_no_connectivity: bool = False,
            try_limit: int = 100,
            seed: int = None,
    ) -> ig.Graph:
        assert num_edges >= self.num_vertices - 1, f"Expected at least {self.num_vertices - 1} edges (connectivity condition), but got {num_edges}"

        if label_random_method not in ["sample", "choice"]:
            raise ValueError("`label_random_method` must be one of ['sample', 'choice']")

        if seed is not None:
            random.seed(seed)

        current_try = 0
        while current_try < try_limit:

            graph_generated = ig.Graph.Erdos_Renyi(
                n=self.num_vertices,
                m=num_edges,
                directed=False,
                loops=False,
            )

            graph_generated.to_directed(mode="acyclic")

            if accept_isolates:
                graph_without_isolates = self.graph_drop_isolates(graph_generated, copy=True, validation=False)
                is_weakly_connected = graph_without_isolates.is_connected(mode="weak")
            else:
                is_weakly_connected = graph_generated.is_connected(mode="weak")

            if is_weakly_connected or accept_no_connectivity:

                labels = list(range(self.label_cardinality)) if self.label_cardinality > 1 else [0]

                if label_random_method == "sample":
                    random_labels = random.sample(labels, self.num_vertices)
                else:
                    random_labels = random.choices(labels, k=self.num_vertices)

                graph_generated.vs[self.graph_label_key] = random_labels

                graph_sorted = self.graph_sort_topological(graph_generated)

                return graph_sorted

            current_try += 1

        raise Exception("try_limit exceeded with no correct dag generated")
