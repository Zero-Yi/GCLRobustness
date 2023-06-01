from torch_geometric.utils import dropout_edge
from torch_geometric.loader import DataLoader

class Greedy():
    def __init__(self, encoder, classifier, pn=0.05):
        '''
        Params:
            pn: ratio of flipping edges
        '''
        self.encoder = encoder
        self.classifier = classifier
        self.pn = pn

    def attack(self, eval_set, mask):
        '''
        Params:
            eval_set: the clean evaluation dataset, NOT dataloader
            mask: to indicate the graphs to attack
        Return:
            dataloader_eval_adv: the adversarial evaluation dataloader, which consists of adversarial samples only
        '''
        eval_set_adv = [data for data, mask_value in zip(eval_set, mask) if mask_value]

        for one_graph in eval_set_adv:
            updated_edge_index, _ = dropout_edge(one_graph.edge_index, p = self.pn) # Randomly drop edges
            one_graph.put_edge_index(updated_edge_index, layout='coo') # Revise the graph in the dataset

        dataloader_eval_adv = DataLoader(eval_set_adv)

        return dataloader_eval_adv


