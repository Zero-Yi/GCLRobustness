from torch_geometric.utils import dropout_edge, add_random_edge, dense_to_sparse, to_dense_adj
from torch_geometric.loader import DataLoader
import torch

def flip_edges(graph, p, force_undirected=False):
    """
    Flip the connection status of edges in the graph with probability p.
    If an edge exists, it will be removed; if it does not exist, it will be added.
    
    Parameters:
        graph: A graph object containing edge_index
        p: Probability p to decide whether to flip each edge
        force_undirected: If True, ensure the graph is undirected
    
    Returns:
        new_edge_index: Updated edge_index
    """
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    
    # Create adjacency matrix using to_dense_adj
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].bool()
    
    # Generate a random matrix of the same shape, with elements being True with probability p
    random_matrix = torch.rand((num_nodes, num_nodes)) < p
    
    # Flip edges in the adjacency matrix using XOR operation
    new_adj_matrix = adj_matrix ^ random_matrix
    
    # Ensure diagonal elements are 0 (i.e., no self-loops)
    new_adj_matrix.fill_diagonal_(0)
    
    # If force_undirected is True, make the matrix symmetric
    if force_undirected:
        upper_triangle = torch.triu(new_adj_matrix, diagonal=1)
        new_adj_matrix = upper_triangle | upper_triangle.t()
    
    # Convert the new adjacency matrix back to edge_index format using dense_to_sparse
    new_edge_index, _ = dense_to_sparse(new_adj_matrix)
    
    return new_edge_index

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
            # `add_random_edge` from pytorch geometry seems to have introduced some new bugs
            # updated_edge_index, _ = dropout_edge(one_graph.edge_index, p=self.pn, force_undirected=self.undirected) # Randomly drop edges
            # _, added_edges = add_random_edge(one_graph.edge_index, p=self.pn, force_undirected=self.undirected) # Randomly add edges
            # updated_edge_index = torch.cat((updated_edge_index,added_edges), dim=1)
            updated_edge_index = flip_edges(one_graph, p=self.pn, force_undirected=self.undirected)
            one_graph.put_edge_index(updated_edge_index, layout='coo')

        dataloader_eval_adv = DataLoader(eval_set_adv)

        return dataloader_eval_adv


