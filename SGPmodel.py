import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_num):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:inducing_num, :], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    
class Sparse_GP():
    def __init__(self, X_train, y_train, device, inducing_num):
        self.X_train = X_train 
        self.y_train = y_train
        # Instantiate the model and move to cuda
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.model = GPRegressionModel(self.X_train, self.y_train, self.likelihood, inducing_num).to(device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        
    def train(self, training_iterations):
        # Find optimal model hyperparameters 
        self.model.train()
        self.likelihood.train()
        # "Loss" for GPs - the marginal log likelihood

        for i in range(training_iterations):
            # Zero backprop gradients
            self.optimizer.zero_grad()
            # Get output from model
            output = self.model(self.X_train)
            # Calc loss and backprop derivatives
            loss = -self.mll(output, self.y_train)
            loss.backward(retain_graph = True)
            if (i+1) % 20 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            self.optimizer.step()
            torch.cuda.empty_cache()