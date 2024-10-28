import logging
import os
import shutil

import dask.dataframe as dd
import pandas as pd
import tqdm
from torch.utils.data import DataLoader

from src.encoders.pace import PaceVaeV3

logger = logging.getLogger(__name__)


def generate_predictor_graphs_batch(
        model,
        evaluator,
        graphs,
):
    batch = []

    for graph in graphs:
        mu, _ = model.encode([graph])
        y = evaluator(graph)
        Z = mu[0].detach().numpy()

        batch.append(
            {
                'vector': Z,
                'target': y,
            }
        )

    return batch


def create_predictor_dataset(
        model: PaceVaeV3,
        graphs_dataloader: DataLoader,
        output_dir: str,
        evaluator,
        npartitions: int = 4,
):
    tmp_dir = output_dir + "_tmp"

    for batch_ind, batch in enumerate(graphs_dataloader):
        processed_batch = generate_predictor_graphs_batch(
            model,
            evaluator,
            batch
        )

        df_to_write = pd.DataFrame(processed_batch)
        df_to_write.to_parquet(
            f"{tmp_dir}/part-{batch_ind}.parquet",
            engine="pyarrow"
        )

    logging.info(f"Dataset creation completed.")
