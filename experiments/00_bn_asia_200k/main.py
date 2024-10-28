import os
from math import floor

import dask.dataframe as dd
import gpytorch
import torch

from src.encoders.pace import PaceVae
from src.encoders.pace_utils import PaceDag
from src.predictors.gp import GPRegressionModel
from src.toolkit.labeled import LabeledDag
from src.train_utils import load_model_state


def prepare_predictor():
    df = dd.read_parquet("data_predictor/train", engine="pyarrow", dtype_backend="pyarrow")
    df = df.compute()

    X = torch.Tensor(df['vector'].array)
    y = torch.Tensor(df['target'].array)

    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    likelihood_state_dict = torch.load('predictor/likelihood32.pth')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(likelihood_state_dict)
    likelihood.eval()

    # Create a new GP model
    predictor = GPRegressionModel(train_x, train_y, likelihood)
    state_dict = torch.load('predictor/predictor32.pth')
    predictor.load_state_dict(state_dict)
    predictor.eval()

    def _predictor(x):
        with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
            value = predictor(x).mean.detach().numpy()[0]

        return value

    return _predictor


if __name__ == "__main__":
    labeled_graph_toolkit = LabeledDag(8, 8)
    pace_graph_toolkit = PaceDag(8, 8)

    graph_dict = {
        'l0': 0,
        'l1': 1,
        'l2': 2,
        'l3': 3,
        'l4': 4,
        'l5': 5,
        'l6': 6,
        'l7': 7,
        'e0': [],
        'e1': [1],
        'e2': [0, 0],
        'e3': [0, 0, 0],
        'e4': [0, 1, 0, 0],
        'e5': [1, 1, 0, 0, 0],
        'e6': [0, 1, 0, 0, 1, 0],
        'e7': [0, 0, 0, 1, 1, 1, 0],
    }

    labeled_graph = labeled_graph_toolkit.from_dict_to_graph(graph_dict)

    model = PaceVae(
        PaceDag(8, 8),
        ninp=32,
        nhead=8,
        nhid=64,
        nlayers=3,
        dropout=0.15,
        fc_hidden=32,
        nz=32
    )

    model_name = os.path.join("model", 'model_checkpoint_20.pth')
    load_model_state(model, model_name)
    model.eval()

    predictor = prepare_predictor()

    print("ENCODING TO LATENT VECTOR:")
    mu, _ = model.encode([labeled_graph])
    Z = mu[0].detach().numpy()
    print("Z.shape:", Z.shape)
    print("Z:", Z)

    expected = -13331.093616667435
    print(f"PREDICTOR RESULT: {expected} expected")
    value = predictor(mu)
    print("value:", value, "diff:", abs(value - expected) / abs(expected))

    print("DECODING:")
    graph_recon = model.decode(mu)[0]
    print(graph_recon)

    print("Decoded graph equals?")
    print(labeled_graph_toolkit.graph_equals(labeled_graph, graph_recon))
