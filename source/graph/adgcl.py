import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data.dataset import random_split

import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast

from torch_geometric.nn import GINConv, global_add_pool, summary
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import get_laplacian, remove_self_loops

from tqdm import tqdm
import itertools
import warnings
import sys
sys.path.append("..")
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os.path as osp
import argparse
import random

from utils.attacker.greedy import Greedy
from utils.attacker.PGD import PGDAttack
from utils.attacker.PRBCD import MyPRBCDAttack
from utils.attacker.GRBCD import MyGRBCDAttack
from utils.layer.wgin_conv import WGINConv

from graph.gin import GIN, LogReg, GCL_classifier, eval_encoder



from torch_scatter import scatter




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class Encoder(torch.nn.Module):
    def __init__(self, encoder, normalize=False):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.normalize = normalize

    def forward(self, x, edge_index, batch, edge_weight=None):
        if self.normalize:
            # use normalized adjacency
            edge_index, edge_weight = get_laplacian(edge_index, normalization = 'sym')
            edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
        else:
            edge_weight = None
 
        eg , en  = self.encoder(x, edge_index, edge_weight=edge_weight, batch=batch)
        return en, eg
    @staticmethod
    def calc_loss( x, x_aug, temperature=0.2, sym=True):
		# x and x_aug shape -> Batch x proj_hidden_dim

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1)/2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1
        return loss

class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, encoder_output_dim, mlp_edge_model_dim=64):
        super(ViewLearner, self).__init__()
        self.encoder = encoder
        # WHAT KIND OF DIM SHOULD BE HERE?
        self.mlp_edge_model = Sequential(
			Linear(encoder_output_dim* 2, mlp_edge_model_dim),
			ReLU(),
			Linear(mlp_edge_model_dim, 1)
		)

    def forward(self, x, edge_index, batch, edge_weight=None):
        _ , node_emb  = self.encoder(x, edge_index, edge_weight=edge_weight, batch=batch)

        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits

# Training in every epoch
def train(encoder_model, view_model, dataloader, optimizer_encoder, optimizer_view):
    model_loss_all = 0
    view_loss_all = 0
    reg_all = 0
    for data in dataloader:
        # set up
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        # train view to maximize contrastive loss
        view_model.train()
        view_model.zero_grad()
        encoder_model.eval()

        x, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        edge_logits = view_model(data.x, data.edge_index, batch=data.batch)

        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        x_aug, _ = encoder_model(data.x, data.edge_index, batch=data.batch, edge_weight=batch_aug_edge_weight)

        # regularization
        row, col = data.edge_index
        edge_batch = data.batch[row]
        edge_drop_out_prob = 1 - batch_aug_edge_weight

        uni, edge_batch_num = edge_batch.unique(return_counts=True)
        sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

        reg = []
        for b_id in range(128):
            if b_id in uni:
                num_edges = edge_batch_num[uni.tolist().index(b_id)]
                reg.append(sum_pe[b_id] / num_edges)
            else:
                    # means no edges in that graph. So don't include.
                pass
        num_graph_with_edges = len(reg)
        reg = torch.stack(reg)
        reg = reg.mean()


        view_loss = encoder_model.calc_loss(x, x_aug) - (5 * reg)
        view_loss_all += view_loss.item() * data.num_graphs
        reg_all += reg.item()
        # gradient ascent formulation
        (-view_loss).backward()
        optimizer_view.step()


        # train (model) to minimize contrastive loss
        encoder_model.train()
        view_model.eval()
        encoder_model.zero_grad()

        x, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        edge_logits = view_model(data.x, data.edge_index, batch=data.batch)

        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

        x_aug, _ = encoder_model(data.x, data.edge_index, batch=data.batch, edge_weight=batch_aug_edge_weight)

        model_loss = encoder_model.calc_loss(x, x_aug)
        model_loss_all += model_loss.item() * data.num_graphs
        # standard gradient descent formulation
        model_loss.backward()
        optimizer_encoder.step()
    return model_loss_all

def train_classifier(x, y, num_classse=2, epoches=100, lr=0.01, device='cuda', seed=42):
    '''
    Params:
    x: the embeddings
    y: the labels
    num_classse: the dimension for the output layer
    epoches: the epoches to train the classifier
    '''
    x.to(device)
    y.to(device)
    setup_seed(seed)
    classifier = LogReg(x.shape[1], num_classse).to(device)
    xent = CrossEntropyLoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=0.0)
    # classifier.cuda()

    classifier.train()
    for _ in range(epoches):
        opt.zero_grad()

        logits = classifier(x)
        loss = xent(logits, y)
        
        loss.backward()
        opt.step()

    classifier.eval()
    return classifier

def autoAug(dataset):
    if dataset=='PROTEINS':
        return A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=8),
                           A.NodeDropping(pn=0.2)], 1)
    elif dataset=='NCI1':
        return A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=6),
                           A.NodeDropping(pn=0.2)], 1)
    elif dataset=='DD':
        return A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=57),
                           A.NodeDropping(pn=0.2)], 1)
    elif dataset=='REDDIT-BINARY':
        return A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=86),
                           A.EdgeRemoving(pe=0.2),
                           A.NodeDropping(pn=0.2)], 1)
    elif dataset=='REDDIT-MULTI-5K':
        return A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=100),
                           A.EdgeRemoving(pe=0.2),
                           A.NodeDropping(pn=0.2)], 1)
    else:
        raise NotImplementedError

def arg_parse():
    parser = argparse.ArgumentParser(description='gin.py')
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='Dataset')
    parser.add_argument('--PGD', type=bool, default=False,
                        help='Whether apply PGD attack. Default: false')
    parser.add_argument('--PRBCD', type=bool, default=False,
                        help='Whether apply PRBCD attack. Default: false')
    parser.add_argument('--GRBCD', type=bool, default=False,
                        help='Whether apply GRBCD attack. Default: false')
    parser.add_argument('--seed_split', type=int, default=42,
                        help='Seed for the dataset split. Default: 42')
    parser.add_argument('--seeds_encoder', nargs='+', type=int, default=[1,2,3,4,5],
                        help='List of seeds for the encoder initialization. Default: [1,2,3,4,5]') 
    parser.add_argument('--seeds_lc', nargs='+', type=int, default=[1,2,3,4,5],
                        help='List of seeds for the classifier initialization. Default: [1,2,3,4,5]')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='Whether apply normalized Laplacian. Default: false')      
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    dataset_name = args.dataset
    do_PGD_attack = args.PGD
    do_PRBCD_attack = args.PRBCD
    do_GRBCD_attack = args.GRBCD
    seed_split = args.seed_split
    seeds_encoder = args.seeds_encoder
    seeds_lc = args.seeds_lc
    do_normalize = args.normalize

    # Hyperparams
    lr = 0.01
    num_layers = 3
    epochs = 20
    print(f'====== dataset:{dataset_name}, PGD :{do_PGD_attack}, PRBCD :{do_PRBCD_attack}, GRBCD :{do_GRBCD_attack}======')

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    num_features = max(dataset.num_features, 1)
    num_classes = dataset.num_classes
    if dataset.num_features==0 :
        print("No node feature, paddings of 1 will be used in GIN when forwarding.")

    # Split the dataset into two part for training classifier and final evaluation, train_set can be further divided into training and validation parts
    setup_seed(seed_split) # set seed for the reproducibility
    train_set, eval_set = random_split(dataset, [0.9, 0.1])
    dataloader_train = DataLoader(train_set, batch_size=128, shuffle=True)
    dataloader_eval = DataLoader(eval_set, batch_size=128, shuffle=False) # Do not shuffle the evaluation set to make it reproduceable

    # Define the augmentations
    # HERE I NEED TO START MY CHANGES

    list_encoders = []

    for seed_encoder in seeds_encoder: 

        setup_seed(seed_encoder) # set seed for the reproducibility
        gconv = GIN(num_features=num_features, dim=32, num_gc_layers=num_layers, device=device).to(device)
        encoder_model = Encoder(encoder=gconv).to(device)
        view_model = ViewLearner(encoder=gconv,encoder_output_dim=32*num_layers).to(device)
        encoder_optimizer = Adam(encoder_model.parameters(), lr=lr)
        view_optimizer = Adam(view_model.parameters(), lr=lr)

        # Train the encoder with full dataset without labels using contrastive learning
        with tqdm(total=20, desc='(T)') as pbar:
            for epoch in range(1, epochs + 1):
                loss = train(encoder_model, view_model, dataloader, encoder_optimizer,view_optimizer )
                pbar.set_postfix({'loss': loss})
                pbar.update()



        # HERE I NEED TO FINISH MY CHANGES
        # Get embeddings for the train_set
        encoder_model.eval()
        embedding_global, y = encoder_model.encoder.get_embeddings(dataloader_train)

        # Save trained model parameters for reproduce if needed
        # torch.save(encoder_model.state_dict(), 'Savings/model_params/model.pt')

        # Save the embeddings for reproduce if needed
        # torch.save(embedding_global, 'Savings/global_embeddings/embeddings.pt')
        # torch.save(y, 'Savings/global_embeddings/y.pt')    

        accs_clean = []
        accs_adv_PGD = []
        accs_adv_PRBCD = []
        accs_adv_GRBCD = []
        accs_adv_greedy = []

        for seed_lc in seeds_lc:
            # ====== Train classifiers and do attack =====================================
            classifier = train_classifier(embedding_global, y, num_classse=num_classes, seed=seed_lc)

            # Put encoder and classifier together, drop the augmentor
            encoder_classifier = GCL_classifier(encoder_model.encoder, classifier)
            
            encoder_classifier.eval() # Try to save memory
            encoder_classifier.requires_grad_(False) # Try to save memory

            # Accuracy on the clean evaluation data
            acc_clean, mask = eval_encoder(encoder_classifier, dataloader_eval)
            accs_clean.append(acc_clean)

            # Instantiate an attacker and attack the victim model
            # ++++++++++ Greedy ++++++++++++++++
            Greedyattacker = Greedy(pn=0.05)
            dataloader_eval_adv_greedy = Greedyattacker.attack(eval_set, mask)

            # Accuracy on the adversarial data only
            acc_adv_only_greedy, _ = eval_encoder(encoder_classifier, dataloader_eval_adv_greedy)

            # Overall adversarial accuracy
            acc_adv_greedy = acc_clean * acc_adv_only_greedy # T/all * Tadv/T = Tadv/all
            accs_adv_greedy.append(acc_adv_greedy)
            # ++++++++++ Greedy over ++++++++++++++++

            if do_PGD_attack == True:
                # ++++++++++ PGD ++++++++++++++++
                PGDattacker = PGDAttack(surrogate=encoder_classifier, device=device, epsilon=1e-4, log=False)
                dataloader_eval_adv_PGD = PGDattacker.attack(eval_set, mask, attack_ratio=0.05)

                # Accuracy on the adversarial data only
                acc_adv_only_PGD, _ = eval_encoder(encoder_classifier, dataloader_eval_adv_PGD)

                # Overall adversarial accuracy
                acc_adv_PGD = acc_clean * acc_adv_only_PGD # T/all * Tadv/T = Tadv/all
                accs_adv_PGD.append(acc_adv_PGD)
                # ++++++++++ PGD over ++++++++++++++++

            if do_PRBCD_attack == True:
                # ++++++++++ PRBCD ++++++++++++++++
                PRBCDattacker = MyPRBCDAttack(encoder_classifier, block_size=250_000, mylog=True)
                dataloader_eval_adv_PRBCD = PRBCDattacker.attack(eval_set, mask, attack_ratio=0.05)

                # Accuracy on the adversarial data only
                acc_adv_only_PRBCD, _ = eval_encoder(encoder_classifier, dataloader_eval_adv_PRBCD)

                # Overall adversarial accuracy
                acc_adv_PRBCD = acc_clean * acc_adv_only_PRBCD # T/all * Tadv/T = Tadv/all
                accs_adv_PRBCD.append(acc_adv_PRBCD)

                del dataloader_eval_adv_PRBCD
                # ++++++++++ PRBCD over ++++++++++++++++

            if do_GRBCD_attack == True:
                # ++++++++++ GRBCD ++++++++++++++++
                GRBCDattacker = MyGRBCDAttack(encoder_classifier, block_size=250_000, mylog=True)
                dataloader_eval_adv_GRBCD = GRBCDattacker.attack(eval_set, mask, attack_ratio=0.05)

                # Accuracy on the adversarial data only
                acc_adv_only_GRBCD, _ = eval_encoder(encoder_classifier, dataloader_eval_adv_GRBCD)

                # Overall adversarial accuracy
                acc_adv_GRBCD = acc_clean * acc_adv_only_GRBCD # T/all * Tadv/T = Tadv/all
                accs_adv_GRBCD.append(acc_adv_GRBCD)
                # ++++++++++ GRBCD over ++++++++++++++++

            print(f'(A): clean accuracy={acc_clean:.4f}, greedy adversarial accuracy={acc_adv_greedy:.4f}')
            del Greedyattacker # Try to save memory

            if do_PGD_attack == True:
                print(f'(A): PGD adversarial accuracy={acc_adv_PGD:.4f}')
                del PGDattacker # Try to save memory
            if do_PRBCD_attack == True:
                print(f'(A): PRBCD adversarial accuracy={acc_adv_PRBCD:.4f}')
                del PRBCDattacker # Try to save memory
            if do_GRBCD_attack == True:
                print(f'(A): GRBCD adversarial accuracy={acc_adv_GRBCD:.4f}')
                del GRBCDattacker # Try to save memory

            print(f'==== Trial with LC seed {seed_lc} finished. ====')
                
        accs_clean = torch.stack(accs_clean)
        accs_adv_greedy = torch.stack(accs_adv_greedy)
        print(f'(A): clean average accuracy={accs_clean.mean():.4f}, std={accs_clean.std():.4f}')
        print(f'(A): greedy adversarial average accuracy={accs_adv_greedy.mean():.4f}, std={accs_adv_greedy.std():.4f}, drop percentage:{(accs_clean.mean() - accs_adv_greedy.mean())/accs_clean.mean()}')
        if do_PGD_attack == True:
            accs_adv_PGD = torch.stack(accs_adv_PGD)
            print(f'(A): PGD adversarial average accuracy={accs_adv_PGD.mean():.4f}, std={accs_adv_PGD.std():.4f}, drop percentage:{(accs_clean.mean() - accs_adv_PGD.mean())/accs_clean.mean()}')
        if do_PRBCD_attack == True:
            accs_adv_PRBCD = torch.stack(accs_adv_PRBCD)
            print(f'(A): PRBCD adversarial average accuracy={accs_adv_PRBCD.mean():.4f}, std={accs_adv_PRBCD.std():.4f}, drop percentage:{(accs_clean.mean() - accs_adv_PRBCD.mean())/accs_clean.mean()}')
        if do_GRBCD_attack == True:
            print(f'(A): GRBCD adversarial average accuracy={accs_adv_GRBCD.mean():.4f}, std={accs_adv_GRBCD.std():.4f}, drop percentage:{(accs_clean.mean() - accs_adv_GRBCD.mean())/accs_clean.mean()}')
        
        list_encoders.append({'accs_clean': accs_clean.mean(), 
                                    'accs_adv_greedy': accs_adv_greedy.mean(),
                                    'accs_adv_PGD': accs_adv_PGD.mean() if do_PGD_attack else None,
                                    'accs_adv_PRBCD': accs_adv_PRBCD.mean() if do_PRBCD_attack else None,
                                    'accs_adv_GRBCD': accs_adv_GRBCD.mean() if do_GRBCD_attack else None})
        print(f'==== Trial with encoder seed {seed_encoder} finished. ====')
    
    for i in range(len(seeds_encoder)):
        print(f'Result for seed {seeds_encoder[i]}: {list_encoders[i]}')
