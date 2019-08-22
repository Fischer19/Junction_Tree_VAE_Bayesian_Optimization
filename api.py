#from dgl import model_zoo
from util import *
from SGPmodel import *

#model = model_zoo.chem.load_pretrained("JTNN_ZINC")

class LatentSmileConverter(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def toSmile(self, *args):
        return self.decoder(*args)
    
    def toLatent(self, smiles):
        return self.encoder(smiles)


class ScoreCalculator(object):
    def __init__(self, scorecalc):
        self.scorecalc = scorecalc

    def __call__(self, smiles):
        score = []
        for s in smiles:
            score.append(self.scorecalc(s))
        return score
    

# Define BO function using general API:
# LSC for LatentSmileConverter object
def BayesianOpt_ei(LSC, SC, model, likelihood, max_iteration = 50, device = "cuda"):
    lb = torch.min(model.train_inputs[0], dim = 0)[0]
    ub = torch.max(model.train_inputs[0], dim = 0)[0]
    valid_s = []
    mol_score = []
    for iteration in range(max_iteration):
        if iteration == 0:
            flag = False
        else:
            flag = True
        xmin = next_x_ei(model, likelihood, model.train_inputs[0], lb,ub,flag,5,1, device)
        valid_smiles=[]
        scores=[]
        real_scores = []
        for x_new in xmin:
            # TODO
            tree_vec, mol_vec = x_new.chunk(2,1)
            s=LSC.toSmile(tree_vec, mol_vec)
            if s is not None:
                valid_smiles.append(s)
                score = SC(s)
                y_new = score
                #print("new x score:", score)
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

def BayesianOpt_ei_one_step(LSC, SC, model, likelihood, num_mol =1, max_iteration = 50, device = "cuda"):
    lb = torch.min(model.train_inputs[0], dim = 0)[0]
    ub = torch.max(model.train_inputs[0], dim = 0)[0]
    valid_s = []
    mol_score = []
    xmin = []
    for i in range(num_mol):
        flag = i
        xmin += next_x_ei(model, likelihood, model.train_inputs[0], lb,ub,flag,5,1, device)
    valid_smiles=[]
    for x_new in xmin:
        # TODO
        tree_vec, mol_vec = x_new.chunk(2,1)
        s=LSC.toSmile(tree_vec, mol_vec)
        if s is not None:
            valid_smiles.append(s)

    return valid_smiles, xmin

class SGPOptimizer(object):
    def __init__(self, X, y, device = "cuda"):
        self.X_train, self.y_train, self.X_test, self.y_test = generate_train_data(X, y, device)
        # TODO: how to determine number of inducing points
        self.SGP = Sparse_GP(self.X_train, self.y_train, device, 1000)
        self.device = device
        
    def train_SGP(self, train_iterations = 200):
        with gpytorch.settings.use_toeplitz(True):
            self.SGP.train(train_iterations)
            
    def optimize_smiles(self, smiles_w_scores, converter):
        pass
    
    def optimize_auto(self, X_y, LSC, SC):
        if X_y is not None:
            X, y = X_y
            X = torch.cat((self.SGP.model.train_inputs[0], X.to(self.device)),0) # incorporate new evaluation
            y = torch.cat((self.SGP.model.train_targets, y.float().to(self.device)),0)
        else:
            X, y = self.SGP.model.train_inputs[0], self.SGP.model.train_targets
        result = []
        self.SGP.model.set_train_data(X, y, strict = False)
        with gpytorch.settings.cg_tolerance(10), gpytorch.settings.max_cg_iterations(1500):
            valid_s, mol_score = BayesianOpt_ei(LSC, SC, self.SGP.model, self.SGP.likelihood, max_iteration=10)
            result.append((valid_s, mol_score))
        return result
    
    def optimize_one_step(self, X_y, LSC, num_mol = 1):
        if X_y is not None:
            X, y = X_y
            X = torch.cat((self.SGP.model.train_inputs[0], torch.cat(X).to(self.device)),0) # incorporate new evaluation
            y = torch.cat((self.SGP.model.train_targets, torch.Tensor(y).float().to(self.device)),0)
            update_posterior(self.SGP.model, self.SGP.likelihood, X, y)
        else:
            X, y = self.SGP.model.train_inputs[0], self.SGP.model.train_targets
        self.SGP.model.set_train_data(X, y, strict = False)
        with gpytorch.settings.cg_tolerance(10), gpytorch.settings.max_cg_iterations(1500):
            result, latent = BayesianOpt_ei_one_step(LSC, SC, self.SGP.model, self.SGP.likelihood, 10, max_iteration=1)
        return result, latent

    
    def save(self, trained_data=True):
        torch.save(self.SGP.model.state_dict(), 'SGPoptimizer.pth')
    
    def load(self, trained_data=True):
        state_dict = torch.load('SGPoptimizer.pth')
        self.SGP.model.load_state_dict(state_dict)
    
