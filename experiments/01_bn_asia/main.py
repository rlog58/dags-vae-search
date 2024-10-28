import math
import os
import random
import time

import igraph as ig
import igraph
import matplotlib.pyplot as plt
import dask.dataframe as dd
import gpytorch
import numpy as np
import torch
import tqdm
from dask_ml.model_selection import train_test_split
from igraph import Layout
from matplotlib.patches import FancyArrowPatch, Circle
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.encoders.pace import PaceVaeV3
from src.predictors.gp import GPRegressionModel
from src.predictors.utils import create_predictor_dataset
from src.problem.bn.bnlearn import BNLearnWrapper
from src.toolkit.labeled import LabeledDag, LABEL_KEY
from src.train_utils import load_model_state


toolkit = LabeledDag(num_vertices=8, label_cardinality=8)

model = PaceVaeV3(
        max_num_vertices=8,
        vertex_label_cardinality=8,
        vertices_embedding_size=32,
        num_heads=8,
        num_layers=3,
        ff_hidden_size=64,
        latent_layer_size=32,
        fc_hidden=32,
        dropout=0.15,
    )


def encoder_train_split(toolkit):

    dataset_path = "../../data/bn_asia/encoder_dataset"

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

    loss, recon, kld = model.loss_direct(batch)
    loss_value = loss.item()

    loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss_value, recon, kld

def train_model(toolkit, model):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    BATCH_SIZE = 32
    EPOCHS = 10

    dataset = LabeledDagDatasetInMemory(
        dataset_dir="data/train/",
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

    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    # Junk.
    model_name = os.path.join("model_full_vectorized", 'model_checkpoint_100.pth')
    load_model_state(model, model_name)

    loss_value = math.inf
    time_start = time.time()

    for epoch in range(100 + 1, 100 + EPOCHS + 1):

        model.train()

        progress_bar = tqdm.tqdm(dl, desc=f"Epoch {epoch} Training")

        for batch in progress_bar:
            loss_value, recon, kld = train_batch(batch, model, optimizer)

            progress_bar.set_description(f'Epoch: {epoch}, loss: {loss_value / BATCH_SIZE:.5f}, recon: {recon / BATCH_SIZE:.5f}, kld: {kld / BATCH_SIZE:.5f}')

        scheduler.step(loss_value)

        comp_time = time.time() - time_start
        print('====> Epoch: {0} loss: {1:.5f}, compute time: {2:.4f}'.format(epoch, loss_value / BATCH_SIZE, comp_time))

        model_name = os.path.join("model_full_vectorized", 'model_checkpoint_{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)

    print("Done")

    print("Try to load model")
    loaded_model = torch.load(model_name)

def batch_test(toolkit, batch, model, encode_times=10, decode_times=10):

    n_valid = 0
    n_same_structure = 0
    n_perfect = 0


    mu, logvar = model.encode(batch)

    _, nll, _ = model.loss(batch)

    for _ in range(encode_times):

        z = mu

        for _ in range(decode_times):
            batch_reconstructed = model.decode(z)

            n_valid += sum(toolkit.is_valid_graph(g) for g in batch_reconstructed)

            n_same_structure += sum(toolkit.graph_equals(g0, g1, attributes_match=False) for g0, g1 in zip(batch, batch_reconstructed))
            n_perfect += sum(toolkit.graph_equals(g0, g1) for g0, g1 in zip(batch, batch_reconstructed))

    return nll, n_valid, n_same_structure, n_perfect

def model_test(toolkit, model):

    model_name = os.path.join(
        "model_full_vectorized",
        'model_checkpoint_110.pth'
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
        drop_last=True,
        # num_workers=1,
    )

    progress_bar = tqdm.tqdm(dl, desc=f"Test reconstruct metrics")

    total_nll = 0
    total_n_valid = 0
    total_n_same_structure = 0
    total_n_perfect = 0

    encode_times = 1
    decode_times = 1

    for i, batch in enumerate(progress_bar):
        nll, n_valid, n_same_structure, n_perfect = batch_test(toolkit, batch, model, encode_times, decode_times)

        total_nll += nll
        total_n_valid += n_valid
        total_n_same_structure += n_same_structure
        total_n_perfect += n_perfect

        progress_bar.set_description(f'Avg recon loss: {total_nll / (BATCH_SIZE * (i + 1)):.3f}, valid ratio: {(total_n_valid / (BATCH_SIZE * (i + 1) * encode_times * decode_times)):.3f}, structure recon accuracy: {(total_n_same_structure / (BATCH_SIZE * (i + 1) * encode_times * decode_times)):.3f}, complete recon accuracy: {(total_n_perfect / (BATCH_SIZE * (i + 1) * encode_times * decode_times)):.3f}')


def prepare_predictor_data(toolkit, model):

    BATCH_SIZE = 64

    model_name = os.path.join(
        "model_full_vectorized",
        'model_checkpoint_110.pth'
    )
    load_model_state(model, model_name)

    model.eval()

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
        drop_last=True,
        # num_workers=1,
    )

    evaluator = BNLearnWrapper("asia", "bic").score

    create_predictor_dataset(
        model,
        dl,
        "predictor_dataset",
        evaluator,
        2
    )

def predictor_train_split():

    dataset_path = "predictor_dataset"

    df = dd.read_parquet(dataset_path, engine="pyarrow", dtype_backend="pyarrow")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    train_df.to_parquet("predictor_data/train", engine="pyarrow")
    test_df.to_parquet("predictor_data/test", engine="pyarrow")

def train_predictor(toolkit, model):
    df = dd.read_parquet("predictor_dataset")
    df = df.compute()

    X = torch.Tensor(df['vector'].array)
    y = torch.Tensor(df['target'].array)

    train_n = int(math.floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    print("Model params count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    ITERATIONS = 10000

    for i in range(ITERATIONS):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, ITERATIONS, loss.item()))
        optimizer.step()
        torch.cuda.empty_cache()

    print("Done")
    print("Testing:")

    model.eval()
    likelihood.eval()

    with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        preds = model(test_x)

    print("predictions example:")
    print(preds[:10].mean.detach().numpy())

    # 100
    # Test MAE: 3117.620849609375
    # Test MAPE: 0.22864490747451782
    # 1000
    # Test MAE: 528.623046875
    # Test MAPE: 0.03854367509484291
    # 2000
    # Test MAE: 338.1571350097656
    # Test MAPE: 0.02457481063902378
    # 5000
    # Test MAE: 204.15782165527344
    # Test MAPE: 0.014772910624742508
    print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))
    print('Test MAPE: {}'.format(torch.mean(torch.abs(preds.mean - test_y) / test_y)))

    print("Saving predictor.pth")
    model_name = os.path.join("predictor_results", 'predictor.pth')
    torch.save(model.state_dict(), model_name)
    torch.load(model_name)

    print("Saving likelihood.pth")
    model_name = os.path.join("predictor_results", 'likelihood.pth')
    torch.save(likelihood.state_dict(), model_name)
    torch.load(model_name)


def draw_dag(
        graph: ig.Graph,
        ax: plt.Axes,
        node_size=0.025,
        node_color='skyblue',
        node_edge_color='k',
        edge_color='k',
        arrowsize=15,
        label_fontsize=8,
):
    """
    Draws a Directed Acyclic Graph (DAG) using the Sugiyama layout on a given matplotlib Axes.

    Parameters:
    - graph (ig.Graph): The input igraph DAG.
    - ax (plt.Axes): The matplotlib Axes to draw the graph on.
    - node_size (int): Size of the nodes.
    - node_color (str): Fill color of the nodes.
    - node_edge_color (str): Edge color of the nodes.
    - edge_color (str): Color of the edges.
    - arrowsize (int): Size of the arrow heads.
    - labels (list or None): List of labels for the nodes. If None, node indices are used.
    - label_fontsize (int): Font size of the labels.
    """
    if not graph.is_directed():
        raise ValueError("The graph must be directed.")

    if not graph.is_dag():
        raise ValueError("The graph must be a Directed Acyclic Graph (DAG).")

    # Compute Sugiyama layout
    layout = graph.layout("sugiyama")

    # Extract coordinates and normalize them for plotting
    x_coords = [coord[0] for coord in layout]
    y_coords = [coord[1] for coord in layout]

    # Normalize coordinates to fit in [0,1] range
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    normalized_x = [(x - min_x) / (max_x - min_x + 1e-6) for x in x_coords]
    normalized_y = [(y - min_y) / (max_y - min_y + 1e-6) for y in y_coords]

    positions = list(zip(normalized_x, normalized_y))

    # Draw edges
    for edge in graph.es:
        source, target = edge.tuple
        start = positions[source]
        end = positions[target]
        # Convert to coordinates in Axes
        start_x, start_y = start
        end_x, end_y = end
        # Create an arrow
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle='-|>',
            mutation_scale=arrowsize,
            color=edge_color,
            linewidth=1,
            zorder=1
        )
        ax.add_patch(arrow)

    # Draw nodes
    for idx, (x, y) in enumerate(positions[:graph.vcount()]):
        circle = plt.Circle((x, y),
                            radius=node_size,  # Adjust radius based on node_size
                            facecolor=node_color,
                            edgecolor=node_edge_color,
                            zorder=2)
        ax.add_patch(circle)
        # Add labels
        label = graph.vs[idx]["label"]
        ax.text(x, y, label, fontsize=label_fontsize, ha='center', va='center', zorder=3,)

    # Set limits and aspect
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')



def draw_examples(toolkit, model):
    naming = {
        0: "A",
        1: "S",
        2: "T",
        3: "L",
        4: "B",
        5: "E",
        6: "X",
        7: "D",
    }

    model_name = os.path.join(
        "model_full_vectorized",
        'model_checkpoint_110.pth'
    )
    load_model_state(model, model_name)

    model.eval()

    generated_graph = toolkit.generate_random_graph_erdos_renyi(
        num_edges=11,
        label_random_method="sample",
        accept_isolates=False,
        accept_no_connectivity=False,
        try_limit=100,
    )
    generated_graph.vs["label"] = [naming[elem] for elem in generated_graph.vs[LABEL_KEY]]

    pace_graph = model.from_labeled_graph_to_pace_graph(generated_graph)

    pace_graph.vs[0]["label"] = "Start"
    pace_graph.vs[1]["label"] = "Input"
    pace_graph.vs[model.max_num_vertices - 1]["label"] = "Output"

    for i in range(toolkit.num_vertices):
        pace_graph.vs[i + 2]["label"] = naming[pace_graph.vs[i + 2][LABEL_KEY] - 3]


    mu, _  = model.encode([generated_graph])
    decoded_graph = model.decode(mu)[0]
    decoded_graph.vs["label"] = [naming[elem] for elem in decoded_graph.vs[LABEL_KEY]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    fig.suptitle("BNSL: Asia")

    ax1.set_title("Generated DAG")
    draw_dag(generated_graph, ax1)

    ax2.set_title("Generated DAG: PACE wrapping")
    draw_dag(pace_graph, ax2)

    ax3.set_title("Decoded DAG")
    draw_dag(decoded_graph, ax3)

    plt.show()


if __name__ == "__main__":

    # train_split()

    # train_model(toolkit, model)

    # loss_log_likelihood
    # epoch   2 - Avg recon loss: 2.580, valid ratio: 1.000, structure recon accuracy: 0.203, complete recon accuracy: 0.167: 100%|██████████| 688/688 [02:36<00:00,  4.40it/s]
    # epoch  10 - Avg recon loss: 0.144, valid ratio: 1.000, structure recon accuracy: 0.834, complete recon accuracy: 0.833: 100%|██████████| 688/688 [02:22<00:00,  4.82it/s]
    # epoch  20 - Avg recon loss: 0.033, valid ratio: 1.000, structure recon accuracy: 0.895, complete recon accuracy: 0.894: 100%|██████████| 688/688 [02:30<00:00,  4.58it/s]
    # epoch  30 - Avg recon loss: 0.020, valid ratio: 1.000, structure recon accuracy: 0.915, complete recon accuracy: 0.914: 100%|██████████| 688/688 [02:41<00:00,  4.26it/s]
    # epoch  40 - Avg recon loss: 0.012, valid ratio: 1.000, structure recon accuracy: 0.917, complete recon accuracy: 0.917: 100%|██████████| 688/688 [02:24<00:00,  4.75it/s]
    #
    # loss_log_likelihood_full_vectorized
    # epoch   2 - Avg recon loss: 2.577, valid ratio: 1.000, structure recon accuracy: 0.206, complete recon accuracy: 0.168: 100%|██████████| 688/688 [02:17<00:00,  5.01it/s]
    # epoch  10 - Avg recon loss: 0.139, valid ratio: 1.000, structure recon accuracy: 0.821, complete recon accuracy: 0.819: 100%|██████████| 688/688 [02:02<00:00,  5.61it/s]
    # epoch  20 - Avg recon loss: 0.034, valid ratio: 1.000, structure recon accuracy: 0.883, complete recon accuracy: 0.882: 100%|██████████| 688/688 [02:08<00:00,  5.35it/s]
    # epoch  30 - Avg recon loss: 0.017, valid ratio: 1.000, structure recon accuracy: 0.913, complete recon accuracy: 0.913: 100%|██████████| 688/688 [02:10<00:00,  5.29it/s]
    # epoch  40 - Avg recon loss: 0.015, valid ratio: 1.000, structure recon accuracy: 0.919, complete recon accuracy: 0.919: 100%|██████████| 688/688 [02:05<00:00,  5.48it/s]
    # epoch 100 - Avg recon loss: 0.007, valid ratio: 1.000, structure recon accuracy: 0.935, complete recon accuracy: 0.935: 100%|██████████| 688/688 [02:08<00:00,  5.36it/s]
    #model_test(toolkit, model)

    #prepare_predictor_data(toolkit, model)
    #predictor_train_split()

    #Test
    #MAPE: -0.05218623951077461
    #train_predictor(toolkit, model)

    draw_examples(toolkit, model)
