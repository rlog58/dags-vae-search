import os
from math import floor

import dask.dataframe as dd
import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class GPRegressionModel(ExactGP):
    def __init__(
            self,
            train_x,
            train_y,
            likelihood
    ):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=train_x[:500, :],
            likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    df = dd.read_parquet("../experiments/00_bn_asia_200k/asia_200k_32", engine="pyarrow", dtype_backend="pyarrow")
    df = df.compute()

    X = torch.Tensor(df['vector'].array)
    y = torch.Tensor(df['target'].array)

    train_n = int(floor(0.8 * len(X)))
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


    def train():
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


    train()

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
    model_name = os.path.join("../tmp/00_bn_asia_200k/predictor_results", 'predictor32.pth')
    torch.save(model.state_dict(), model_name)
    torch.load(model_name)

    print("Saving likelihood.pth")
    model_name = os.path.join("../tmp/00_bn_asia_200k/predictor_results", 'likelihood32.pth')
    torch.save(likelihood.state_dict(), model_name)
    torch.load(model_name)
