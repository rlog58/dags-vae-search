import os
import shutil
import time
from typing import List, Tuple

import igraph as ig
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from src.toolkit.labeled import LabeledDag


def filter_non_isomorphic_graphs(toolkit: LabeledDag, graphs: List[ig.Graph]):
    n = len(graphs)

    if n < 2:
        return (_ for _ in graphs)

    isomorphic_mask = np.zeros(n, dtype=np.bool_)

    for i in range(n - 1):
        if not isomorphic_mask[i]:
            for j in range(i + 1, n):
                if not isomorphic_mask[j]:
                    isomorphic_mask[j] = toolkit.graph_equals(
                        graphs[i],
                        graphs[j],
                        attributes_match=False,
                        complete_isomorphism_check=False,
                        validation=False,
                    )
            yield graphs[i]

    if not isomorphic_mask[n - 1]:
        yield graphs[n - 1]


def get_encoder_train_schema(
        num_vertices: int,
        density_limit: float,
        steps_limit: int,
) -> List[Tuple[int, int]]:
    """
    Generates a training schema for dags based on the number of vertices, density limit, and step count.
    """

    if num_vertices < 1:
        raise ValueError("num_vertices must be at least 1.")

    if not (0 < density_limit <= 1):
        raise ValueError("density_limit must be between 0 (exclusive) and 1 (inclusive).")

    if steps_limit < 1:
        raise ValueError("steps_limit must be at least 1.")

    min_edges = num_vertices - 1
    max_edges = (num_vertices * (num_vertices - 1)) // 2  # Correct maximum edges

    max_edges_density = int(max_edges * density_limit)

    if max_edges_density < min_edges:
        raise ValueError("max_edges_density cannot be less than min_edges. "
                         "Check num_vertices and density_limit.")

    # Generate unique, sorted edge counts
    linspace = list(
        map(int, np.linspace(min_edges, max_edges_density, steps_limit))
    )
    unique_edges = sorted(set(linspace))

    schema = [(edge_count, (i + 1) ** 2) for i, edge_count in enumerate(unique_edges)]

    # TODO: replace with logger
    print(f"Minimum edges: {min_edges}")
    print(f"Maximum edges based on density: {max_edges_density}")
    print(f"Steps ratio: {(max_edges_density - min_edges) / steps_limit:.2f}")

    return schema


def generate_batch(
        toolkit: LabeledDag,
        num_edges: int,
        batch_size: int = 100,
        try_limit: int = 100,
) -> List[ig.Graph]:
    batch = []
    for i in range(batch_size):
        graph = None
        try:
            graph = toolkit.generate_random_graph_erdos_renyi(
                num_edges=num_edges,
                label_random_method='sample',
                accept_isolates=False,
                accept_no_connectivity=True,
                try_limit=try_limit,
            )
        except Exception as e:
            print(e)
        finally:
            if graph is not None:
                batch.append(graph)

    return batch


if __name__ == "__main__":

    NUM_VERTICES = 11

    train_schema = get_train_schema(
        11,
        0.4,
        20
    )

    print("train_schema:", train_schema)

    batch_size = 400

    total_records = sum(x[1] * batch_size for x in train_schema)

    print("total_records:", total_records)

    toolkit = LabeledDag(
        num_vertices=NUM_VERTICES,
        label_cardinality=NUM_VERTICES,
        validation=True,
    )

    if os.path.isdir('../tmp'):
        print("exists")
        shutil.rmtree('../tmp')
        os.makedirs('../tmp')
    else:
        os.makedirs('../tmp')

    start_time = time.time()

    progress_bar = tqdm.tqdm(train_schema)

    part = 0
    for elem in progress_bar:

        inner_progress_bar = tqdm.tqdm(range(elem[1]))

        for i in inner_progress_bar:
            new_batch = generate_batch(toolkit, elem[0], batch_size=batch_size)
            # new_batch = list(filter_non_isomorphic_graphs(toolkit, batch))

            if len(new_batch) > 0:
                dict_batch = [toolkit.from_graph_to_dict_writable(g) for g in new_batch]

                df_to_write = pd.DataFrame(dict_batch)

                table = pa.Table.from_pandas(df_to_write).cast(toolkit.pyarrow_schema)

                pq.write_table(table, f"../tmp/part-{part}.parquet")

                part += 1
