import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, x, *args, **kwargs):
        '''
        Params:
            x: Features of nodes. Tensor of shape (n_nodes, n_features)
            msk: Mask of averaging features. Binary tensor of shape (n_nodes,)
        '''
        return F.sigmoid(torch.mean(x, dim=0))
