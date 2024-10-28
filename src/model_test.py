import os

import tqdm
from torch.utils.data import DataLoader

from src.datasets import LabeledDagDatasetInMemory
from src.encoders.pace import PaceVae
from src.encoders.pace_utils import PaceDag
from src.toolkit.labeled import LabeledDag
from src.train_utils import collate_graph_batch, load_model_state


def batch_test(toolkit, batch, model, encode_times=10, decode_times=10):
    n_valid = 0
    n_perfect = 0

    mu, logvar = model.encode(batch)

    _, nll, _ = model.loss(mu, logvar, batch)

    for _ in range(encode_times):

        z = mu

        for _ in range(decode_times):
            batch_reconstructed = model.decode(z)

            n_valid += sum(toolkit.is_valid_graph(g) for g in batch_reconstructed)
            n_perfect += sum(toolkit.graph_equals(g0, g1) for g0, g1 in zip(batch, batch_reconstructed))

    return nll, n_valid, n_perfect


if __name__ == "__main__":
    labeled_graph_toolkit = LabeledDag(num_vertices=8, label_cardinality=8)

    BATCH_SIZE = 8

    dataset = LabeledDagDatasetInMemory(
        dataset_dir="/experiments/00_bn_asia_200k/data/test",
        toolkit=labeled_graph_toolkit,
    )

    dl = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_graph_batch,
        shuffle=True,
    )

    # ASIA model

    model = PaceVae(
        toolkit=PaceDag(
            num_vertices=8,
            label_cardinality=8,
        ),
        ninp=32,
        nhead=8,
        nhid=64,
        nlayers=3,
        dropout=0.15,
        fc_hidden=32,
        nz=32  # 64 по умолчанию
    )

    # model = PACE_VAE(
    #     max_n=12 + 3,  # число вершин
    #     nvt=12 + 3,  # число типов вершин
    #     START_TYPE=0,  # тип входящей вершины
    #     END_TYPE=1,  # тип выходящей вершины
    #     START_SYMBOL=2,  # тип стартовой вершины
    #     ninp=32,
    #     nhead=8,
    #     nhid=64,
    #     nlayers=3,
    #     dropout=0.15,
    #     fc_hidden=32,
    #     nz=64
    # )

    model_name = os.path.join(
        "/experiments/00_bn_asia_200k/model",
        'model_checkpoint_20.pth')
    load_model_state(model, model_name)

    model.eval()

    progress_bar = tqdm.tqdm(dl, desc=f"Test reconstruct metrics")

    total_nll = 0
    total_n_valid = 0
    total_n_perfect = 0

    encode_times = 10
    decode_times = 10

    for i, batch in enumerate(progress_bar):
        nll, n_valid, n_perfect = batch_test(labeled_graph_toolkit, batch, model, encode_times, decode_times)

        total_nll += nll
        total_n_valid += n_valid
        total_n_perfect += n_perfect

        progress_bar.set_description(
            f'AVG recon loss: {total_nll / (BATCH_SIZE * (i + 1))}, valid ratio: {(total_n_valid / (BATCH_SIZE * (i + 1) * encode_times * decode_times)):.4f}, recon accuracy: {(total_n_perfect / (BATCH_SIZE * (i + 1) * encode_times * decode_times)):.4f}')
