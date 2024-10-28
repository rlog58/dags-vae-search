import logging
import os
import shutil
from typing import List, Tuple

import dask.dataframe as dd
import igraph as ig
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from src.toolkit.labeled import LabeledDag

logger = logging.getLogger(__name__)


def encoder_dag_train_schema(
        num_vertices: int,
        density_limit: float,
        steps_limit: int,
) -> List[Tuple[int, int]]:
    """
    Generates a training schema for DAG encoder based on the number of vertices,
    density limit, and steps limit.
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

    train_schema = [(edge_count, (i + 1) ** 2) for i, edge_count in enumerate(unique_edges)]

    logger.info(f"Minimum edges: {min_edges}")
    logger.info(f"Maximum edges based on density: {max_edges_density}")
    logger.info(f"Steps ratio: {(max_edges_density - min_edges) / steps_limit:.2f}")

    return train_schema


def generate_encoder_graphs_batch(
        toolkit: LabeledDag,
        num_edges: int,
        batch_size: int = 100,
        try_limit: int = 100,
        label_random_method: str = 'sample',
        accept_isolates: bool = False,
        accept_no_connectivity=False,
) -> List[ig.Graph]:
    """
    Generates a batch of random Erdos-Renyi graphs.
    """
    batch = []
    attempts = 0
    while len(batch) < batch_size and attempts < batch_size * try_limit:
        attempts += 1
        try:
            graph = toolkit.generate_random_graph_erdos_renyi(
                num_edges=num_edges,
                label_random_method=label_random_method,
                accept_isolates=accept_isolates,
                accept_no_connectivity=accept_no_connectivity,
                try_limit=try_limit,
            )
            batch.append(graph)
        except Exception as e:  # Replace SpecificException with actual expected exceptions
            logging.error(f"Graph generation failed: {e}")
            break

    if len(batch) < batch_size:
        logging.warning(
            f"Requested batch size {batch_size}, but only {len(batch)} graphs were generated after {attempts} attempts.")

    return batch


def create_encoder_dataset(
        toolkit: LabeledDag,
        output_dir: str,
        batch_size: int,
        steps_limit: int,
        density_limit: float = 0.6,
        overwrite: bool = False,
        label_random_method: str = 'sample',
        accept_isolates: bool = False,
        accept_no_connectivity=False,
        npartitions: int = 4,
):
    """
    Generates an encoder dataset by creating graph batches and saving them as Parquet files.
    """
    tmp_dir = output_dir + "_tmp"

    if os.path.isdir(output_dir):
        if overwrite:
            logging.info(f"Overwriting existing directory: {output_dir}")
            try:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                if os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir)
                os.makedirs(tmp_dir)
            except Exception as e:
                logging.error(f"Failed to overwrite directory: {e}")
                raise
        else:
            raise FileExistsError(f"Directory '{output_dir}' already exists.")
    else:
        try:
            os.makedirs(output_dir)
            os.makedirs(output_dir + "_tmp")
            logging.info(f"Created directory: {output_dir}")
            logging.info(f"Created directory: {tmp_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory '{output_dir}': {e}")
            raise

    train_schema = encoder_dag_train_schema(
        toolkit.num_vertices,
        density_limit,
        steps_limit,
    )

    logging.info(f"Train schema (num edges, batches count): {train_schema}")

    progress_bar = tqdm.tqdm(train_schema, desc="Processing Train Schema")

    part = 0
    try:
        for schema_entry in progress_bar:
            num_iterations = schema_entry[1]

            for _ in range(num_iterations):
                try:
                    new_batch = generate_encoder_graphs_batch(
                        toolkit,
                        schema_entry[0],
                        batch_size=batch_size,
                        label_random_method=label_random_method,
                        accept_isolates=accept_isolates,
                        accept_no_connectivity=accept_no_connectivity,
                    )

                    if not new_batch:
                        continue

                    dict_batch = [
                        toolkit.from_graph_to_dict_writable(g) for g in new_batch
                    ]

                    # Directly create PyArrow Table without Pandas
                    table = pa.Table.from_pylist(dict_batch).cast(toolkit.pyarrow_schema)

                    pq.write_table(
                        table,
                        f"{output_dir}_tmp/part-{part}.parquet",
                    )

                    part += 1

                except Exception as batch_err:
                    logging.error(f"Error processing batch {part}: {batch_err}")
                    # Depending on requirements, decide to continue or raise
                    raise

    except Exception as e:
        logging.error(f"Error during dataset creation: {e}")
        raise

    logging.info(f"Dataset creation completed with {part * batch_size} records.")

    logging.info(f"Repartition on {npartitions} partitions")

    df = dd.read_parquet(output_dir + "_tmp")

    df \
        .repartition(npartitions=npartitions) \
        .to_parquet(
        output_dir,
        engine='pyarrow',
    )
    logging.info(f"Removing: {tmp_dir}.")
    shutil.rmtree(tmp_dir)