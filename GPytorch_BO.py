from util import *
import numpy as np
import argparse

# Define your own training data or
# We load the data (y is minued!)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pretrain", action = "store_true")
parser.add_argument('-t', "--train_num", default = 1000)
parser.add_argument("-e", "--ob_epoch", default = 60)
parser.add_argument("-c", "--cuda", default = "0")
parser.add_argument("-r", "--random_seed", default = 1)
parser.add_argument("-v", "--vocab_path", default = "/home/ec2-user/ASAIL/jtnn_bo/jtnn/vocab.txt")
parser.add_argument("-m", "--model_path", default = "model.iter-0-3000")
parser.add_argument("-f", "--feature_dir", default = './bo/latent_features2.txt')
parser.add_argument("-s", "--target_dir", default = './bo/targets2.txt')
parser.add_argument("-o", "--output_dir", default = 'result/')

args = parser.parse_args()

random_seed = int(args.random_seed)
training_num = int(args.train_num)
ob_epoch = int(args.ob_epoch)
feature_dir = args.feature_dir
target_dir = args.target_dir
vocab_path = args.vocab_path
model_path = args.model_path

X = np.loadtxt(feature_dir)[:training_num]
y = -np.loadtxt(target_dir)[:training_num]
y = y.reshape((-1, 1))
device = "cuda:{}".format(args.cuda)

X_train, y_train, X_test, y_test = generate_train_data(X, y, device)

#Load sparse Gaussian process regression model and fit the training data

from SGPmodel import *
SGP = Sparse_GP(X_train, y_train, device, 1000)
if args.pretrain:
    state_dict = torch.load('SGPmodel_state_dict.pth')
    SGP.model.load_state_dict(state_dict)
else:
    with gpytorch.settings.use_toeplitz(True):
        SGP.train(200)
    torch.save(SGP.model.state_dict(), 'SGPmodel_state_dict_{}.pth'.format(args.train_num))
    
# Test MLL and RMSE on test data
'''
SGP.model.eval()
SGP.likelihood.eval()
with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
        preds = SGP.model(X_test)
MLL = -SGP.mll(preds, y_test)
RMSE = np.sqrt(np.mean((preds.mean.cpu().numpy() - y_test.cpu().numpy())**2))
print("Test MLL:", MLL.item())
print("Test RMSE: ", RMSE)
'''

# Load VAE model
JT_model = make_vae_model(vocab_path, model_path, device="cuda")

# Start Bayesian optimization for 10 iterations
result = []
for epoch in range(ob_epoch):
    print("Epoch - {}:".format(epoch))
    np.random.seed(epoch * random_seed)
    state_dict = torch.load('SGPmodel_state_dict_{}.pth'.format(args.train_num))
    SGP.model.load_state_dict(state_dict)
    SGP.model.set_train_data(X_train, y_train, strict = False)
    with gpytorch.settings.cg_tolerance(10), gpytorch.settings.max_cg_iterations(1500):
        valid_s, mol_score = BayesianOpt_ei(JT_model, SGP.model, SGP.likelihood, max_iteration=10)
        result.append((valid_s, mol_score))
    save_object(result, args.output_dir + "BO_50epoch_ei_t{}_e{}.dat".format(args.train_num, ob_epoch))
    
