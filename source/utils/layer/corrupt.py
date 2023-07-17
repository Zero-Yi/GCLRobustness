import torch
import torch.nn as nn
import numpy as np

class FeatureShuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_weight=None, *args, **kwargs):
        n_nodes = x.shape[0]
        idx = np.random.permutation(n_nodes)
        shuf_x = x[idx, :]
        return shuf_x, edge_index

