import math
import os
import random
import time

import dask.dataframe as dd
import numpy as np
import torch
import tqdm
from dask_ml.model_selection import train_test_split
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.encoders.pace import PaceVaeV3
from src.toolkit.labeled import LabeledDag
from src.train_utils import load_model_state


def train_split():
    toolkit = LabeledDag(num_vertices=12, label_cardinality=1)

    dataset_path = "../../data/synthetic_12"

    df = dd.read_parquet(dataset_path, engine="pyarrow", dtype_backend="pyarrow")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    train_df.to_parquet("data/train", engine="pyarrow", schema=toolkit.pyarrow_schema)
    test_df.to_parquet("data/test", engine="pyarrow", schema=toolkit.pyarrow_schema)


class LabeledDagDatasetInMemory(Dataset):
    def __init__(self, dataset_dir: str, toolkit: LabeledDag, model: PaceVaeV3):
        self.df = dd.read_parquet(dataset_dir, engine="pyarrow", dtype_backend="pyarrow").compute()
        self.toolkit = toolkit
        self.model = model
        self.graphs = self.load_graphs()
        del self.df

    def load_graphs(self):
        return [
            self.model.prepare_features([self.toolkit.from_dict_to_graph(record._asdict())])
            for record in tqdm.tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Load graphs in memory")
        ]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx]

class LabeledDagDatasetInMemoryTest(Dataset):
    def __init__(self, dataset_dir: str, toolkit: LabeledDag, model: PaceVaeV3):
        self.df = dd.read_parquet(dataset_dir, engine="pyarrow", dtype_backend="pyarrow").compute()
        self.toolkit = toolkit
        self.model = model
        self.graphs = self.load_graphs()
        del self.df

    def load_graphs(self):
        return [
            self.toolkit.from_dict_to_graph(record._asdict())
            for record in tqdm.tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Load graphs in memory")
        ]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx]


def pace_collate_fn(data):
    batch_dict = {}
    keys_to_cat = [
        "vertex_label_features",
        "vertex_position_features",
        "adjacency_matrices",
        "source_masks",
        "target_masks",
        "memory_masks"
    ]

    for key in keys_to_cat:
        batch_dict[key] = torch.cat([el[key] for el in data], dim=0)

    batch_dict["num_vertices"] = [el["num_vertices"][0] for el in data]
    batch_dict["vertex_labels"] = [el["vertex_labels"][0] for el in data]

    return batch_dict


def train_batch(batch, model, optimizer, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad()

    # Origin:
    # Epoch: 1, loss: 1.0752, recon: 1.0695, kld: 1.1408: 100%|██████████| 2341/2341 [04:12<00:00,  9.29it/s]
    # ====> Epoch: 1 loss: 1.0752, compute time: 252.1214

    # Vectorized:
    # Epoch: 1, loss: 1.0752, recon: 1.0695, kld: 1.1406: 100%|██████████| 2341/2341 [02:34<00:00, 15.16it/s]
    # ====> Epoch: 1 loss: 1.0752, compute time: 154.3774

    # Vectorized V2:
    # Epoch: 1, loss: 1.0762, recon: 1.0705, kld: 1.1457: 100%|██████████| 2341/2341 [02:10<00:00, 17.89it/s]
    # ====> Epoch: 1 loss: 1.0762, compute time: 130.8899

    loss, recon, kld = model.loss_direct(batch)
    loss_value = loss.item()

    loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss_value, recon, kld


def train_model():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    toolkit = LabeledDag(num_vertices=12, label_cardinality=1)

    BATCH_SIZE = 32
    EPOCHS = 1

    model = PaceVaeV3(
        max_num_vertices=12,
        vertex_label_cardinality=1,
        vertices_embedding_size=32,
        num_heads=8,
        num_layers=3,
        ff_hidden_size=64,
        latent_layer_size=32,
        fc_hidden=32,
        dropout=0.15,
    )

    dataset = LabeledDagDatasetInMemory(
        dataset_dir="data/test/",
        toolkit=toolkit,
        model=model,
    )

    dl = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        collate_fn=pace_collate_fn,
        shuffle=True,
        #drop_last=True,
        # num_workers=1,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("device:", device)

    print("Model params count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    # Junk.
    # model_name = os.path.join("../results", 'model_checkpoint_1.pth')
    # load_model_state(model, model_name)

    loss_value = math.inf
    time_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        progress_bar = tqdm.tqdm(dl, desc=f"Epoch {epoch} Training")

        for batch in progress_bar:
            loss_value, recon, kld = train_batch(batch, model, optimizer)

            progress_bar.set_description(
                f'Epoch: {epoch}, loss: {loss_value / BATCH_SIZE:.4f}, recon: {recon / BATCH_SIZE:.4f}, kld: {kld / BATCH_SIZE:.4f}')

        scheduler.step(loss_value)

        comp_time = time.time() - time_start
        print('====> Epoch: {0} loss: {1:.4f}, compute time: {2:.4f}'.format(epoch, loss_value / BATCH_SIZE, comp_time))

        model_name = os.path.join("model", 'model_checkpoint_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)

    print("Done")

    print("Try to load model")
    loaded_model = torch.load(model_name)

def batch_test(toolkit, batch, model, encode_times=10, decode_times=10):
    n_valid = 0
    n_perfect = 0

    mu, logvar = model.encode(batch)

    _, nll, _ = model.loss(batch)

    for _ in range(encode_times):

        z = mu

        for _ in range(decode_times):
            batch_reconstructed = model.decode(z)

            n_valid += sum(toolkit.is_valid_graph(g) for g in batch_reconstructed)
            n_perfect += sum(toolkit.graph_equals(g0, g1) for g0, g1 in zip(batch, batch_reconstructed))

    return nll, n_valid, n_perfect

def model_test():

    toolkit = LabeledDag(num_vertices=12, label_cardinality=1)

    model = PaceVaeV3(
        max_num_vertices=12,
        vertex_label_cardinality=1,
        vertices_embedding_size=32,
        num_heads=8,
        num_layers=3,
        ff_hidden_size=64,
        latent_layer_size=32,
        fc_hidden=32,
        dropout=0.15,
    )

    # 20 AVG recon loss: 1.4330551624298096, valid ratio: 1.0000, recon accuracy: 0.1367:   1%|          | 16/2341 [07:21<18:52:05, 29.22s/it]
    # 76 AVG recon loss: 0.6824547052383423, valid ratio: 1.0000, recon accuracy: 0.3640:   0%|          | 7/2341 [03:02<17:02:26, 26.28s/it]
    # 77 AVG recon loss: 0.6356959939002991, valid ratio: 1.0000, recon accuracy: 0.3561:   0%|          | 8/2341 [03:24<16:26:19, 25.37s/it]
    # 78 AVG recon loss: 0.7733710408210754, valid ratio: 1.0000, recon accuracy: 0.3886:   0%|          | 7/2341 [03:03<16:58:01, 26.17s/it]

    model_name = os.path.join(
        "model",
        'model_checkpoint_78.pth'
    )
    load_model_state(model, model_name)

    model.eval()

    BATCH_SIZE = 32

    dataset = LabeledDagDatasetInMemoryTest(
        dataset_dir="data/test/",
        toolkit=toolkit,
        model=model,
    )

    dl = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda data: [g for g in data],
        shuffle=True,
        # drop_last=True,
        # num_workers=1,
    )

    progress_bar = tqdm.tqdm(dl, desc=f"Test reconstruct metrics")

    total_nll = 0
    total_n_valid = 0
    total_n_perfect = 0

    encode_times = 10
    decode_times = 10

    for i, batch in enumerate(progress_bar):
        nll, n_valid, n_perfect = batch_test(toolkit, batch, model, encode_times, decode_times)

        total_nll += nll
        total_n_valid += n_valid
        total_n_perfect += n_perfect

        progress_bar.set_description(
            f'AVG recon loss: {total_nll / (BATCH_SIZE * (i + 1))}, valid ratio: {(total_n_valid / (BATCH_SIZE * (i + 1) * encode_times * decode_times)):.4f}, recon accuracy: {(total_n_perfect / (BATCH_SIZE * (i + 1) * encode_times * decode_times)):.4f}')


if __name__ == '__main__':
    # train_split()
    # train_model()
    model_test()
