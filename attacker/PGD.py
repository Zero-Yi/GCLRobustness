'''
    Copied from deeprobust
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

from attacker import utils


# class PGDAttack():
#     """PGD attack for graph data.

#     Parameters
#     ----------
#     model :
#         model to attack. Default `None`.
#     loss_type: str
#         attack loss type, chosen from ['CE', 'CW']
#     device: str
#         'cpu' or 'cuda'
#     """

#     def __init__(self, model=None, loss_type='CE', device='cuda'):

#         self.loss_type = loss_type
#         self.model = model
#         self.device = device

#         self.edge_weight_ori = None
#         self.edge_direction = None


#     def attack(self, x, edge_index, batch, y, n_perturbations, epochs=200):
#         """Generate perturbations on the input graph.

#         Parameters
#         ----------
#         x :
#             Original (unperturbed) node feature matrix
#         edge_index :
#             Original (unperturbed) edge_index
#         y :
#             graph label
#         n_perturbations : int
#             Number of perturbations on the input graph. Perturbations could
#             be edge removals/additions or feature removals/additions.
#         epochs:
#             number of training epochs

#         Returns
#         ----------
#         fc_edge_index:
#             Fully connected version of the original graph
#         edge_weight:
#             Final edge_weight of the adversarial sample
#         adv_edge_index:
#             edge_index of the perturbed graph
#         """

#         victim_model = self.model
#         victim_model.eval()
#         victim_model.requires_grad_(False)

#         fc_edge_index, edge_weight_ori, edge_direction = self.preprocess(x, edge_index)
#         # edge_weight_change = torch.zeros(int(len(edge_weight_ori)/2), device=self.device, requires_grad=True)

#         with tqdm(total=epochs, desc='(PGD)') as pbar:
#             for t in range(epochs):
#                 # adj_norm = utils.normalize_adj_tensor(modified_adj) # Why is it normalized here?
#                 edge_weight = edge_weight_ori + edge_weight_change * edge_direction
#                 logit = victim_model(x, fc_edge_index, batch, edge_weight=edge_weight)
#                 loss = F.cross_entropy(logit, y)

#                 loss.backward() # Calculate the gradient

#                 lr_weight = 200 / np.sqrt(t+1)
#                 edge_weight_change.data.add_(lr_weight * edge_weight_change.grad) # Update the edge weight

#                 edge_weight_change.grad.zero_() # Clear the gradient

#                 self.projection(edge_weight_change, n_perturbations)
#                 pbar.set_postfix({'loss': loss.item()})
#                 pbar.update()

#         self.random_sample(x, fc_edge_index, batch, y, edge_weight_change, edge_weight_ori, edge_direction, n_perturbations)

#         edge_weight = edge_weight_ori + edge_weight_change * edge_direction

#         edge_weight_bool = edge_weight.to(torch.bool)
#         adv_edge_index = fc_edge_index[:, edge_weight_bool]

#         return adv_edge_index

#     def preprocess(self, x, edge_index):
#         """
#         Params:
#             x: feature matrix of the input graph
#             edge_index: original edge_index
#         """
#         num_nodes = x.shape[0]
#         adj = torch_geometric.utils.to_dense_adj(edge_index=edge_index, max_num_nodes=num_nodes).to(self.device).squeeze()
#         fc_adj = torch.ones_like(adj).to(self.device) - torch.eye(num_nodes).to(self.device)
#         cplmt_adj = fc_adj - adj
#         cplmt_edge_index, _ = torch_geometric.utils.dense_to_sparse(cplmt_adj)
        
#         fc_edge_index = torch.cat((edge_index, cplmt_edge_index), dim=1).to(self.device)
        
#         edge_weight = torch.zeros(int(num_nodes*(num_nodes-1))).to(self.device)
#         edge_weight[:edge_index.shape[1]] = 1.0

#         edge_direction = torch.ones_like(edge_weight).to(self.device)
#         edge_direction[:edge_index.shape[1]] = -1 # For those existing edges, we get the probability of dropping; for those non-existing edges, we get the probability of adding

#         return fc_edge_index, edge_weight, edge_direction

#     def random_sample(self, x, edge_index, batch, y, edge_weight_change, edge_weight_ori, edge_direction, n_perturbations, K=20):
#         best_loss = -1000
#         victim_model = self.model
#         victim_model.eval()
#         with torch.no_grad():
#             s = edge_weight_change.cpu().detach().numpy()
#             for i in range(K):
#                 sampled = np.random.binomial(1, s)

#                 # print(sampled.sum())
#                 if sampled.sum() > n_perturbations:
#                     continue
#                 edge_weight_change.data.copy_(torch.tensor(sampled))
#                 edge_weight = edge_weight_ori + edge_weight_change * edge_direction
#                 logit = victim_model(x, edge_index, batch, edge_weight=edge_weight)
#                 loss = F.cross_entropy(logit, y)
#                 # loss = F.nll_loss(output[idx_train], labels[idx_train])
#                 # print(loss)
#                 if best_loss < loss:
#                     best_loss = loss
#                     best_s = sampled
#             edge_weight_change.data.copy_(torch.tensor(best_s))

#     def _loss(self, output, labels):
#         if self.loss_type == "CE":
#             loss = F.nll_loss(output, labels)
#         if self.loss_type == "CW":
#             onehot = utils.tensor2onehot(labels)
#             best_second_class = (output - 1000*onehot).argmax(1)
#             margin = output[np.arange(len(output)), labels] - \
#                    output[np.arange(len(output)), best_second_class]
#             k = 0
#             loss = -torch.clamp(margin, min=k).mean()
#             # loss = torch.clamp(margin.sum()+50, min=k)
#         return loss

#     def projection(self, edge_weight_change, n_perturbations):
#         # projected = torch.clamp(self.adj_changes, 0, 1)
#         if torch.clamp(edge_weight_change, 0, 1).sum() > n_perturbations:
#             left = (edge_weight_change - 1).min()
#             right = edge_weight_change.max()
#             miu = self.bisection(edge_weight_change, left, right, n_perturbations, epsilon=1e-5)
#             edge_weight_change.data.copy_(torch.clamp(edge_weight_change.data - miu, min=0, max=1))
#         else:
#             edge_weight_change.data.copy_(torch.clamp(edge_weight_change.data, min=0, max=1))

#     def get_modified_adj(self, ori_adj):

#         if self.complementary is None:
#             self.complementary = (torch.ones_like(ori_adj).to(self.device) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

#         m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
#         tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1).to(self.device)
#         print(m.device, tril_indices.device, self.adj_changes.device)
#         m[tril_indices[0], tril_indices[1]] = self.adj_changes
#         m = m + m.t()
#         modified_adj = self.complementary * m + ori_adj

#         return modified_adj

#     def bisection(self, edge_weight_change, a, b, n_perturbations, epsilon):
#         def func(x):
#             return torch.clamp(edge_weight_change-x, 0, 1).sum() - n_perturbations

#         miu = a
#         while ((b-a) >= epsilon):
#             miu = (a+b)/2
#             # Check if middle point is root
#             if (func(miu) == 0.0):
#                 break
#             # Decide the side to repeat the steps
#             if (func(miu)*func(a) < 0):
#                 b = miu
#             else:
#                 a = miu
#         # print("The value of root is : ","%.4f" % miu)
#         return miu

class PGDAttack():
    """PGD attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, surrogate=None, loss_type='CE', device='cuda'):

        self.modified_adj = None
        self.original_adj = None
        self.adj_changes = None
        self.nnodes = None

        self.complementary = None
        self.direction = None    

        self.surrogate = surrogate
        self.device = device

        # if attack_structure:
        #     assert nnodes is not None, 'Please give nnodes='
        #     self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
        #     self.adj_changes.data.fill_(0)

    def attack(self, x, edge_index, batch, y, n_perturbations, epochs=200):
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
        edge_weight:
            Final edge_weight of the adversarial sample
        adv_edge_index:
            edge_index of the perturbed graph
        """

        victim_model = self.surrogate
        victim_model.eval()
        victim_model.requires_grad_(False)

        fc_edge_index, edge_weight = self.preprocess(x, edge_index)

        for t in tqdm(range(epochs)):
            modified_weight, _ = self.get_modified_weight()
            logit = victim_model(x, fc_edge_index, batch, edge_weight=modified_weight)
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
        # cplmt_adj = fc_adj - self.original_adj
        # cplmt_edge_index, _ = torch_geometric.utils.dense_to_sparse(cplmt_adj)
        
        fc_edge_index, _ = torch_geometric.utils.dense_to_sparse(fc_adj) # The result is sorted lexicographically, with the last index changing the fastest (C-style).
        edge_weight = self.original_adj[~torch.eye(self.nnodes,dtype=bool)]

        assert len(edge_weight)==len(self.adj_changes)*2, "Shape of edge_weight and shape of adj_changes should accord."

        # edge_weight = torch.zeros(int(self.nnodes*(self.nnodes-1))).to(self.device)
        # edge_weight[:edge_index.shape[1]] = 1.0

        # edge_direction = torch.ones_like(edge_weight).to(self.device)
        # edge_direction[:edge_index.shape[1]] = -1 # For those existing edges, we get the probability of dropping; for those non-existing edges, we get the probability of adding

        return fc_edge_index, edge_weight

    def random_sample(self, x, edge_index, batch, y, n_perturbations, K = 20):
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_weight, _ = self.get_modified_weight()
                logit = victim_model(x, edge_index, batch, edge_weight=modified_weight)
                loss = F.cross_entropy(logit, y)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
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
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
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

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
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