from torch_geometric.utils import dropout_edge, add_random_edge
from torch_geometric.loader import DataLoader
import torch

class Greedy():
    def __init__(self, undirected=True, pn=0.05):
        '''
        Params:
            underected: the graphs to be attacked are undirected
            pn: ratio of flipping edges
        '''
        self.pn = pn
        self.undirected = undirected

    def attack(self, eval_set, mask):
        '''
        Params:
            eval_set: the clean evaluation dataset, NOT dataloader
            mask: to indicate the graphs to attack
        Return:
            dataloader_eval_adv: the adversarial evaluation dataloader, which consists of adversarial samples only
        '''
        eval_set_adv = [data for data, mask_value in zip(eval_set, mask) if mask_value] # only pick out the data indicated by mask==True

        for one_graph in eval_set_adv:
            updated_edge_index, _ = dropout_edge(one_graph.edge_index, p=self.pn, force_undirected=self.undirected) # Randomly drop edges
            _, added_edges = add_random_edge(one_graph.edge_index, p=self.pn, force_undirected=self.undirected) # Randomly add edges
            updated_edge_index = torch.cat((updated_edge_index,added_edges), dim=1)
            one_graph.put_edge_index(updated_edge_index, layout='coo')

        dataloader_eval_adv = DataLoader(eval_set_adv)

        return dataloader_eval_adv


