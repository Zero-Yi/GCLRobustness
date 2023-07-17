from torch_geometric.contrib.nn import GRBCDAttack
import torch
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class MyGRBCDAttack(GRBCDAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int,
        device = 'cuda',
        mylog = False,
        **kwargs
    ):
        super().__init__(model=model, block_size=block_size, log=False, **kwargs)
        self.device = device
        self.mylog = mylog

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
        if self.mylog == True:
            for one_graph in tqdm(eval_set_adv):
                one_graph.to(self.device)
                if one_graph.x is None:
                    num_nodes = one_graph.num_nodes
                    one_graph.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)
                budget = int(one_graph.edge_index.shape[1] * attack_ratio)
                adv_edge_index, _ = super().attack(x=one_graph.x, edge_index=one_graph.edge_index, labels=one_graph.y, budget=budget)
                new_graph = Data(edge_index=adv_edge_index, x=one_graph.x, y=one_graph.y).to(self.device)
                adv_datalist_eval.append(new_graph)
        else:
            for one_graph in eval_set_adv:
                one_graph.to(self.device)
                if one_graph.x is None:
                    num_nodes = one_graph.num_nodes
                    one_graph.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)
                budget = int(one_graph.edge_index.shape[1] * attack_ratio)
                adv_edge_index, _ = super().attack(x=one_graph.x, edge_index=one_graph.edge_index, labels=one_graph.y, budget=budget)
                new_graph = Data(edge_index=adv_edge_index, x=one_graph.x, y=one_graph.y).to(self.device)
                adv_datalist_eval.append(new_graph)

        dataloader_eval_adv = DataLoader(adv_datalist_eval, batch_size=batch_size)

        return dataloader_eval_adv