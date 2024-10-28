import os
import subprocess

import igraph as ig
from pgmpy.utils import get_example_model

from src.toolkit.labeled import LABEL_KEY


class BNLearnWrapper:
    def __init__(
            self,
            dataset_name: str,
            metric_name: str,
            score_script_filename: str = "bnlearn_score.R",
    ):
        self.dataset_name = dataset_name
        self.metric_name = metric_name
        self.score_script_filename = score_script_filename

        self.model = get_example_model(dataset_name)

        self.vertex_mapping = {i: label for i, label in enumerate(self.model.nodes)}

        self.scripts_path = os.path.join(os.path.dirname(__file__), "bnlearn_scripts")

    def score(self, labeled_graph: ig.Graph, label_key: str = LABEL_KEY) -> float:
        madel_n = self.model.number_of_nodes()
        n = labeled_graph.vcount()
        labels = labeled_graph.vs()[label_key]

        intersected_labels = len(set(range(madel_n)).intersection(set(labels)))

        assert madel_n == n, f"Expected {madel_n} vertices, but got {n}"
        assert madel_n == intersected_labels, f"Expected graph labels from 0 to {madel_n - 1}, but got {labels}"

        # Reindex graph vertices to require real BN problem mapping
        reindex_mapping = {vertex.index: vertex[label_key] for vertex in labeled_graph.vs}

        reindex_graph = ig.Graph(directed=True)
        reindex_graph.add_vertices(n)
        reindex_graph.add_edges([(reindex_mapping[v1], reindex_mapping[v2]) for v1, v2 in labeled_graph.get_edgelist()])

        adj_str = " ".join("".join(str(v) for v in row) for row in reindex_graph.get_adjacency().data)

        command = [
            "Rscript",
            os.path.join(self.scripts_path, self.score_script_filename),
            self.dataset_name,
            self.metric_name,
            adj_str,
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"R script failed with error: {result.stdout}")

        value = float(result.stdout.strip())

        return value
