from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature
import copy
import torch

class MyFeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(MyFeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

    def mask_feature(self, x, pf):
        node_num = x.shape[0]
        mask_num = int(node_num * pf)
        node_idx = [i for i in range(node_num)]
        mask_idx = random.sample(node_idx, mask_num)
        aug_feature = copy.deepcopy(input_feature)
        zeros = torch.zeros_like(x[0])
        for j in mask_idx:
            aug_feature[j] = zeros
        return aug_feature
