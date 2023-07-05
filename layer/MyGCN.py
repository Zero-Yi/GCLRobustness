from torch_geometric.nn.models import GCN
import torch

class MyGCN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gcn = GCN(*args, **kwargs)
        
    def forward(self, x, edge_index, edge_weight = None):
        return self.gcn(x, edge_index, edge_weight=edge_weight)