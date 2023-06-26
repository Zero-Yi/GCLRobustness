'''
    Modified based on deeprobust
'''
"""
    Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
        https://arxiv.org/pdf/1906.04214.pdf
    Tensorflow Implementation:
        https://github.com/KaidiXu/GCN_ADV_Train
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from attacker import utils

class PGDAttack():
    """PGD attack for graph data.

    Parameters
    ----------
    surrogate :
        model to attack. Default `None`.
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    device: str
        'cpu' or 'cuda'
    epsilon:
        tolerance during bisection
    """

    def __init__(self, surrogate=None, loss_type='CE', device='cuda', epsilon=1e-5, log=True):

        self.modified_adj = None
        self.original_adj = None
        self.adj_changes = None
        self.nnodes = None

        self.complementary = None
        self.direction = None    

        self.surrogate = surrogate
        self.device = device
        self.epsilon = epsilon
        self.log = log

    def attack(self, eval_set, mask=None, batch_size=128, attack_ratio=0.05):
        '''
        Params:
            eval_set: the clean evaluation dataset, NOT dataloader
            mask: to indicate the graphs to attack
            batch_size: batch size of returned dataloader
            attack_ratio: budget measuring in the ratio of originally existing edges of graphs
        Return:
            dataloader_eval_adv: the adversarial evaluation dataloader, which consists of adversarial samples only
        '''
        if mask==None: # by default using all samples
            mask = torch.ones(len(eval_set)).bool

        eval_set_adv = [data for data, mask_value in zip(eval_set, mask) if mask_value] # only pick out the data indicated by mask==True

        adv_datalist_eval = []
        if self.log == True:
            for one_graph in tqdm(eval_set_adv):
                one_graph.to(self.device)
                if one_graph.x is None:
                    num_nodes = one_graph.num_nodes
                    one_graph.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)
                n_perturbations = int(one_graph.edge_index.shape[1] * attack_ratio)
                _, _, adv_edge_index = self.attack_one_graph(one_graph.x, one_graph.edge_index, one_graph.batch, one_graph.y, n_perturbations)
                new_graph = Data(edge_index=adv_edge_index, x=one_graph.x, y=one_graph.y)
                adv_datalist_eval.append(new_graph)
        else:
            for one_graph in eval_set_adv:
                one_graph.to(self.device)
                if one_graph.x is None:
                    num_nodes = one_graph.num_nodes
                    one_graph.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)
                n_perturbations = int(one_graph.edge_index.shape[1] * attack_ratio)
                _, _, adv_edge_index = self.attack_one_graph(one_graph.x, one_graph.edge_index, one_graph.batch, one_graph.y, n_perturbations)
                new_graph = Data(edge_index=adv_edge_index, x=one_graph.x, y=one_graph.y)
                adv_datalist_eval.append(new_graph)

        dataloader_eval_adv = DataLoader(adv_datalist_eval, batch_size=128)

        return dataloader_eval_adv

    def attack_one_graph(self, x, edge_index, batch, y, n_perturbations, epochs=200):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        x :
            Original (unperturbed) node feature matrix
        edge_index :
            Original (unperturbed) edge_index
        y :
            graph label
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        Returns
        ----------
        fc_edge_index:
            Fully connected version of the original graph
        modified_weight:
            Final edge_weight of the adversarial fully-connected sample
        adv_edge_index:
            edge_index of the perturbed graph, in a non-fully-connected way
        """

        victim_model = self.surrogate
        victim_model.eval()
        victim_model.requires_grad_(False)

        fc_edge_index, edge_weight = self.preprocess(x, edge_index)

        # for t in tqdm(range(epochs)):
        for t in range(epochs):
            modified_weight, _ = self.get_modified_weight()
            logit = victim_model(x, fc_edge_index, batch=batch, edge_weight=modified_weight)
            loss = F.cross_entropy(logit, y)

            loss.backward() # Calculate the gradient

            lr_weight = 200 / np.sqrt(t+1)
            self.adj_changes.data.add_(lr_weight * self.adj_changes.grad) # Update the edge weight

            self.adj_changes.grad.zero_() # Clear the gradient
            self.projection(n_perturbations)

        self.random_sample(x, fc_edge_index, batch, y, n_perturbations)
        modified_weight, modified_weight_adj = self.get_modified_weight()
        modified_weight = modified_weight.detach()
        self.check_adj_tensor(modified_weight_adj)

        modified_weight_bool = modified_weight.to(torch.bool)
        adv_edge_index = fc_edge_index[:, modified_weight_bool]

        del self.original_adj, self.adj_changes, self.direction # Added by Zinuo to try to save the memory
        torch.cuda.empty_cache() # Added by Zinuo to try to save the memory

        return fc_edge_index, modified_weight, adv_edge_index

    def preprocess(self, x, edge_index):
        """
        In this function, you should set following variables:
            self.original_adj
            self.adj_changes
            self.direction
        Params:
            x: feature matrix of the input graph
            edge_index: original edge_index
        """
        self.nnodes = x.shape[0]
        self.original_adj = torch_geometric.utils.to_dense_adj(edge_index=edge_index, max_num_nodes=self.nnodes).to(self.device).squeeze()
        self.adj_changes = torch.zeros(int(self.nnodes*(self.nnodes-1)/2), device=self.device, requires_grad=True)
        self.direction = (torch.ones_like(self.original_adj).to(self.device) - torch.eye(self.nnodes).to(self.device) - self.original_adj) - self.original_adj

        fc_adj = torch.ones_like(self.original_adj).to(self.device) - torch.eye(self.nnodes).to(self.device)
        
        fc_edge_index, _ = torch_geometric.utils.dense_to_sparse(fc_adj) # The result is sorted lexicographically, with the last index changing the fastest (C-style).
        edge_weight = self.original_adj[~torch.eye(self.nnodes,dtype=bool)]

        assert len(edge_weight)==len(self.adj_changes)*2, "Shape of edge_weight and shape of adj_changes should accord."
        return fc_edge_index, edge_weight

    def random_sample(self, x, edge_index, batch, y, n_perturbations, K = 20):
        best_loss = -1000
        best_s = None
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_weight, _ = self.get_modified_weight()
                logit = victim_model(x, edge_index, batch=batch, edge_weight=modified_weight)
                loss = F.cross_entropy(logit, y)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            if best_s is None: # just don't perturb in this corner case
                print('[Warning]: cannot find a desired perturbation within budget.')
                best_s = np.zeros_like(sampled)
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, n_perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_weight(self):
        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        triu_indices = torch.triu_indices(row=self.nnodes, col=self.nnodes, offset=1)
        m[triu_indices[0], triu_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_weight_adj = self.direction * m + self.original_adj

        modified_weight = modified_weight_adj[~torch.eye(self.nnodes,dtype=bool)]

        return modified_weight, modified_weight_adj

    def bisection(self, a, b, n_perturbations, max_iterations=5000):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        iteration = 0
        while ((b-a) >= self.epsilon and iteration<=max_iterations):
            iteration += 1
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        if iteration > max_iterations:
            print('[Warning]: Max iteration during bisection')
        return miu

    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is symmetric, unweighted, all-zero diagonal.
        """
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"