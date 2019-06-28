import pickle
import gzip
import scipy.stats as sps
import numpy as np
import os.path
import time

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from bo import sascorer
import networkx as nx

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

from jtnn import *


# from optparse import OptionParser
import argparse

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


# parser = OptionParser()
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vocab", dest="vocab_path", default="/home/ubuntu/ASAIL/jtnn_bo/jtnn/vocab.txt")
parser.add_argument("-m", "--model", dest="model_path", required=True)
parser.add_argument("-o", "--save_dir", dest="save_dir", required=True)
parser.add_argument("-n", "--n_train", dest="training_num", default=500)
parser.add_argument("-i", "--n_ind", dest="inducing_num", default=500)
parser.add_argument("-w", "--hidden", dest="hidden_size", default=200)
parser.add_argument("-l", "--latent", dest="latent_size", default=56)
parser.add_argument("-d", "--depth", dest="depth", default=3)
parser.add_argument("-r", "--seed", dest="random_seed", default=19)
parser.add_argument("-e", "--evaluate", dest="eval", default="False")
# opts,args = parser.parse_args()
args = parser.parse_args()

# We load the random seed
np.random.seed(int(args.random_seed))

# We load the data (y is minued!)
kkk = int(args.training_num)
M = int(args.inducing_num)
X = np.loadtxt('./bo/latent_features2.txt')[:kkk]
y = -np.loadtxt('./bo/targets2.txt')[:kkk]
y = y.reshape((-1, 1))
logP_values = np.loadtxt('./bo/logP_values2.txt')
SA_scores = np.loadtxt('./bo/SA_scores2.txt')
cycle_scores = np.loadtxt('./bo/cycle_scores2.txt')
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)
#y = -logP_values
#y = y[:kkk].reshape((-1, 1))
#y = (np.array(y) - np.mean(y)) / np.std(y)

device = "cuda"
# device = "cpu"

n = X.shape[0]

permutation = np.random.choice(n, n, replace=False)

# Cross Validation
CV_loss = []
for i in range(10):    
    print(np.int(np.round(i / 10.0 * n)), np.int(np.round((i+1) / 10.0 * n)))
    X1_train = X[permutation, :][0: np.int(np.round(i / 10.0 * n)), :]
    X2_train = X[permutation, :][np.int(np.round((i+1)/10.0 * n)):, :]
    y1_train = y[permutation, :][0: np.int(np.round(i / 10.0 * n)), :]
    y2_train = y[permutation, :][np.int(np.round((i+1)/10.0 * n)):, :]

    #X_train = X[permutation, :][0: np.int(np.round(0.9 * n)), :]
    X_train = np.concatenate((X1_train, X2_train))
    #X_test = X[permutation, :][np.int(np.round(0.9 * n)):, :]
    X_test = X[permutation, :][np.int(np.round(i / 10.0 * n)):np.int(np.round((i+1) / 10.0 * n)), :]

    #y_train = y[permutation][0: np.int(np.round(0.9 * n))]
    y_train = np.concatenate((y1_train, y2_train))
    #y_test = y[permutation][np.int(np.round(0.9 * n)):]
    y_test = y[permutation][np.int(np.round(i / 10.0 * n)):np.int(np.round((i+1) / 10.0 * n))]

    y_train = y_train.transpose()
    y_test = y_test.transpose()


    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Train sparse Gaussian by gpytorch:

    import gpytorch
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
    from gpytorch.distributions import MultivariateNormal

    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.base_covar_module = ScaleKernel(RBFKernel())
            self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:M, :], likelihood=likelihood)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)


    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
    print(y_train.shape, y_test.shape)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = GPRegressionModel(X_train, y_train, likelihood).cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iterations = 200
    def train():
        for i in range(training_iterations):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(X_train)
            # Calc loss and backprop derivatives
            loss = -mll(output, y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            torch.cuda.empty_cache()
    if args.eval == "False":
        with gpytorch.settings.use_toeplitz(True):
            train()
            print("Save model to {}".format(args.save_dir + '/SGPmodel_state.pth'))
            torch.save(model.state_dict(), args.save_dir + '/SGPmodel_state.pth')
        model.cpu().eval()
        likelihood.cpu().eval()
        with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
            with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
                preds = model(X_test)
        loss = -mll(preds, y_test)
        print("CrossValidation iteration - {}".format(i))
        print("Test mll:", loss)
        RMSE = np.sqrt(np.mean((preds.mean.numpy() - y_test.numpy())**2))
        print("Test RMSE: ",RMSE)
        CV_loss.append((loss, RMSE))
    else:
        state_dict = torch.load('result/SGPmodel_state.pth')
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model = GPRegressionModel(X_train.cpu(), y_train.cpu(), likelihood)  # Create a new GP model
        model.load_state_dict(state_dict) 
        model.eval()
        likelihood.eval()
        with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
            with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
                preds = model(X_test)
        loss = -mll(preds, y_test)
        print("Test mll:", loss)
        RMSE = np.sqrt(np.mean((preds.mean.numpy() - y_test.numpy())**2))
        print("Test RMSE: ",RMSE)
save_object(CV_loss, args.save_dir + "/CV_loss.dat")
        
        