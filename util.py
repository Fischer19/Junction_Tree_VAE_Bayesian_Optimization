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

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

from jtnn import *


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


def generate_train_data(X, y, device):
    # Generate training and testing data

    n = X.shape[0]

    permutation = np.random.choice(n, n, replace=False)

    X_train = X[permutation, :][np.int(np.round(0.1 * n)):, :]
    X_test = X[permutation, :][:np.int(np.round(0.1 * n)), :]

    y_train = y[permutation][np.int(np.round(0.1 * n)):]
    y_test = y[permutation][:np.int(np.round(0.1 * n))]

    y_train = y_train.transpose()
    y_test = y_test.transpose()


    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

    print("Training data shape X:{}, y:{}".format(X_train.shape, y_train.shape))
    print("Testing data shape X:{}, y:{}".format(X_test.shape, y_test.shape))
    return X_train, y_train, X_test, y_test

def make_vae_model(vocab_path, model_path, hidden_size=200, latent_size=56, depth=3, device="cuda"):
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)
    JT_model = DGLJTNNVAE(vocab, hidden_size, latent_size, depth)
    JT_model.load_state_dict(torch.load(model_path))
    return JT_model.to(device)

# Define helper function for BO

def update_posterior(model, likelihood, X_new, Y_new, iter=50):
    model.set_train_data(X_new, Y_new, strict = False)
    # optimize the GP hyperparameters using Adam with lr=0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    #train(training_iterations=iter, optimizer=optimizer)
    #model.train()
    #likelihood.train()
    train_posterior(model, likelihood)

# Define Acquisition function
# TODO: the original paper uses expected imporvement 
def lower_confidence_bound(model, likelihood, x, kappa=1):
    model.eval()
    likelihood.eval()
    #with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
    #    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
            #preds = model(x)
    with gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
        preds = model(x)
    mu, variance = preds.mean, preds.variance
    sigma = variance.sqrt()
    return mu - kappa * sigma

def find_a_candidate(model, likelihood, x_init, lb, ub):
    # transform x to an unconstrained domain
    constraint = constraints.interval(lb, ub)
    #print(x_init)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    #print(unconstrained_x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
    
    # WARNING: this is a memory intensive optimizer
    # TODO: Maybe try other gradient-based iterative methods
    minimizer = optim.LBFGS([unconstrained_x], max_iter=50)

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y = lower_confidence_bound(model, likelihood, x)
        #y = lower_confidence_bound(unconstrained_x)
        #print(autograd.grad(y, unconstrained_x))
        #print(y)
        autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    return x.detach()

# BO with expected improvement heuristic
def batched_evaluation(model, likelihood, x, i = 0, partition = 20):
    model.eval()
    likelihood.eval()
    start = i * x.shape[0]//partition
    end = (i+1) * x.shape[0]//partition
    with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
            preds = model(x[start:end])
    torch.cuda.empty_cache()
    return torch.min(preds.mean)
    
# Define Acquisition function
# TODO: the original paper uses expected imporvement
def log_expected_improvement(model, likelihood, x, previous_best, device):
    model.to(device).eval()
    likelihood.to(device).eval()
    with gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
        preds = model(x)
    mu, variance = preds.mean, preds.variance
    delta_x = previous_best - mu
    m = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    ei = torch.max(delta_x, 0)[0] + variance * torch.exp(m.log_prob(delta_x/variance)) - torch.abs(delta_x) * m.cdf(delta_x/variance)
    #log_ei = torch.log(ei)
    #print("log_ei:", log_ei.item())
    #print("ei:",ei.item(),"mu:",mu.item())
    return -ei


def find_a_candidate_ei(model, likelihood, x_init, lb, ub, previous_best, device):
    # transform x to an unconstrained domain
    constraint = constraints.interval(lb, ub)
    #print(x_init)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    #print(unconstrained_x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
    
    # WARNING: this is a memory intensive optimizer
    # TODO: Maybe try other gradient-based iterative methods
    minimizer = optim.LBFGS([unconstrained_x], max_iter=50)

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y = log_expected_improvement(model, likelihood, x, previous_best, device)
        #y = lower_confidence_bound(unconstrained_x)
        #print(autograd.grad(y, unconstrained_x))
        #print(y)
        autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    return x.detach()

def next_x_ei(model, likelihood, X_train, lb, ub, flag = True, num_candidates_each_x=5, num_x=60, device = "cuda"):
    found_x=[]
    lb = lb.to(device)
    ub = ub.to(device)
    if flag != -1:
        x_init = model.train_inputs[0][-1 - flag::flag+1].to(device)
    else:
        random_index = np.random.randint(0,len(model.train_inputs[0]))
        x_init = model.train_inputs[0][random_index:random_index + 1].to(device)
        #x_init = model.train_inputs[0][-1:].new_empty(1,56).uniform_(0, 1).mul(ub-lb).add_(lb).to(device)
    previous_best = torch.Tensor([0]).to(device)
    for i in range(20):
        previous_best = torch.min(previous_best, batched_evaluation(model.to(device), likelihood.to(device), X_train.to(device), i))
    #print("previous best:", previous_best)
    
    for j in range(num_x):
        candidates = []
        values = []
        for i in range(num_candidates_each_x):
            x = find_a_candidate_ei(model, likelihood, x_init, lb, ub, previous_best, device)
            y = log_expected_improvement(model, likelihood, x, previous_best, device)
            candidates.append(x.to("cpu"))
            values.append(y.to("cpu"))
            # require another random initialization
            random_index = np.random.randint(0,len(model.train_inputs[0]))
            #x_init = model.train_inputs[0][random_index:random_index + 1].to(device)
            x_init =  model.train_inputs[0][-1:].new_empty(1,56).uniform_(0,1).mul(ub-lb).add_(lb).to(device)
        new_values = torch.cat(values)
        new_values[torch.isnan(new_values)] = 1000.
        argmin = torch.min(new_values, dim=0)[1].item()
        min_score = torch.min(new_values, dim=0)[0].item()
        #print("min_score:", min_score)
        found_x.append(candidates[argmin].to(device))
        #x_init=found_x[-1]
        random_index = np.random.randint(0,len(model.train_inputs[0]))
        x_init = model.train_inputs[0][random_index:random_index + 1].to(device)
        #x_init = model.train_inputs[0][-1:].new_empty(1,56).uniform_(0, 1).mul(ub-lb).add_(lb).to(device)
    return found_x

# Inner BO loop
def next_x(model, likelihood, lb, ub, num_candidates_each_x=5, num_x=60, device = "cuda"):
    found_x=[]
    lb = lb.to(device)
    ub = ub.to(device)
    x_init = model.train_inputs[0][-1:].to(device)

    for j in range(num_x):
        candidates = []
        values = []
        for i in range(num_candidates_each_x):
            x = find_a_candidate(model, likelihood, x_init, lb, ub)
            y = lower_confidence_bound(model, likelihood, x)
            candidates.append(x)
            values.append(y)
            # require another random initialization
            random_index = np.random.randint(0,len(model.train_inputs[0]))
            x_init = model.train_inputs[0][random_index:random_index + 1].to(device)
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        min_score = torch.min(torch.cat(values), dim=0)[0].item()
        #print("min_score:", min_score)
        found_x.append(candidates[argmin])
        #x_init=found_x[-1]
        random_index = np.random.randint(0,len(model.train_inputs[0]))
        x_init = model.train_inputs[0][random_index:random_index + 1].to(device)
    return found_x

# Train posterior distribution using all available data
def train_posterior(model, likelihood, training_iter = 50):
    model.train()
    likelihood.train()
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    print("train with X.shape - {}, y.shape - {}".format(train_x.shape, train_y.shape))
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if ((i+1) % 10 == 0):
            print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.likelihood.noise.item()
            ))
        torch.cuda.empty_cache()
        optimizer.step()
        
def get_rdkit_score():        
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)


    logP_values = np.loadtxt('./bo/logP_values2.txt')
    SA_scores = np.loadtxt('./bo/SA_scores2.txt')
    cycle_scores = np.loadtxt('./bo/cycle_scores2.txt')
    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)
    return logP_values, SA_scores, cycle_scores, SA_scores_normalized, logP_values_normalized, cycle_scores_normalized


def compute_mol_score(s):
    logP_values, SA_scores, cycle_scores, SA_scores_normalized, logP_values_normalized, cycle_scores_normalized = get_rdkit_score()
    current_log_P_value = Descriptors.MolLogP(MolFromSmiles(s))
    current_SA_score = -sascorer.calculateScore(MolFromSmiles(s))
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(s))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6

    current_cycle_score = -cycle_length

    current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
    current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
    current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

    score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
    #y_new = -current_log_P_value_normalized
    return score


def BayesianOpt_ei(JT_model, model, likelihood, max_iteration = 50, device = "cuda"):
    lb = torch.min(model.train_inputs[0], dim = 0)[0]
    ub = torch.max(model.train_inputs[0], dim = 0)[0]
    valid_s = []
    mol_score = []
    for iteration in range(max_iteration):
        flag = 0
        if iteration == 0:
            flag = -1
        xmin = []
        for i in range(1):
            if iteration != 0:
                flag = i
            xmin += next_x_ei(model, likelihood, model.train_inputs[0], lb,ub,flag,5,1, device)        
        valid_smiles=[]
        scores=[]
        real_scores = []
        for x_new in xmin:
            tree_vec, mol_vec = x_new.chunk(2,1)
            #print(x_new.shape, tree_vec.shape, mol_vec.shape)
            #print(x_new)
            s=JT_model.decode(tree_vec, mol_vec)
            if s is not None:
                valid_smiles.append(s)
                score = compute_mol_score(s)
                y_new = score
                print("new x score:", score)
                scores.append(y_new)

                X = torch.cat((model.train_inputs[0], x_new.to("cuda")),0) # incorporate new evaluation
                y = torch.cat((model.train_targets, torch.tensor([y_new]).float().to("cuda")),0)

        if iteration < max_iteration-1:
            update_posterior(model, likelihood, X, y)
        print(len(scores)," new molecules are found. Iteration-",iteration)
        valid_s += valid_smiles
        mol_score += scores
        #save_object(valid_smiles, save_dir + "/valid_smiles{}.txt".format(iteration))
        #save_object(scores, save_dir + "/scores{}.txt".format(iteration))
    good_score = []
    good_s = []
    for i, s in enumerate(valid_s):
        if "CCCCCCCCC" not in s and "C(C)(C)C(C)(C)C(C)(C)" not in s:
            good_s.append(s)
            good_score.append(mol_score[i])
    return good_s, good_score


def BayesianOpt(JT_model, model, likelihood, max_iteration = 50, device = "cuda"):
    lb = torch.min(model.train_inputs[0], dim = 0)[0]
    ub = torch.max(model.train_inputs[0], dim = 0)[0]
    valid_s = []
    mol_score = []
    for iteration in range(max_iteration):
        xmin = next_x(model, likelihood, lb,ub,5,60, device)
        valid_smiles=[]
        scores=[]
        real_scores = []
        for x_new in xmin:
            tree_vec, mol_vec = x_new.chunk(2,1)
            #print(x_new.shape, tree_vec.shape, mol_vec.shape)
            #print(x_new)
            s=JT_model.decode(tree_vec, mol_vec)
            if s is not None:
                valid_smiles.append(s)
                score = compute_mol_score(s)
                y_new = score
                print("new x score:", score)
                scores.append(y_new)

                X = torch.cat((model.train_inputs[0], x_new.to("cuda")),0) # incorporate new evaluation
                y = torch.cat((model.train_targets, torch.tensor([y_new]).float().to("cuda")),0)

        if iteration < max_iteration-1:
            update_posterior(model, likelihood, X, y)
        print(len(scores)," new molecules are found. Iteration-",iteration)
        valid_s += valid_smiles
        mol_score += scores
        #save_object(valid_smiles, save_dir + "/valid_smiles{}.txt".format(iteration))
        #save_object(scores, save_dir + "/scores{}.txt".format(iteration))
    good_score = []
    good_s = []
    for i, s in enumerate(valid_s):
        if "CCCCCCCCC" not in s and "C(C)(C)C(C)(C)C(C)(C)" not in s:
            good_s.append(s)
            good_score.append(mol_score[i])
    return good_s, good_score

from rdkit import Chem
from rdkit.Chem import rdBase
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import RDConfig

def draw_mol(valid_s):
    print("Start drawing {} mols:".format(len(valid_s)))
    ms = []
    for smile in valid_s:
        ms.append(MolFromSmiles(smile))
    img = Draw.MolsToGridImage(ms[:],molsPerRow=3,subImgSize=(300,200),maxMols=500)
    return img