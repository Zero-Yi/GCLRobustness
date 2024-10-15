import torch
import torch.nn as nn
from torch_geometric.nn.models import DeepGraphInfomax, GCN
from torch_geometric.datasets import CitationFull
# from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian, dropout_edge, add_random_edge, remove_self_loops
from torch_geometric.contrib.nn import PRBCDAttack, GRBCDAttack

import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm

import GCL.augmentors as A
import copy
import sys
sys.path.append("..")

from utils.aug.my_feature_masking import MyFeatureMasking
from utils.layer.readout import AvgReadout
from utils.layer.corrupt import FeatureShuffle
from utils.attacker.greedy import Greedy
from utils.attacker.PGD import PGDAttack
from utils.layer.MyGCN import MyGCN
from graph.gin import LogReg
from graph.gcl import train_classifier

def disc(summary_aug, pos, neg, DGI):
    pos_logits = DGI.discriminate(z = pos, summary = summary_aug, sigmoid = False)
    neg_logits = DGI.discriminate(z = neg, summary = summary_aug, sigmoid = False)
    return torch.cat((pos_logits.unsqueeze(1), neg_logits.unsqueeze(1)),1)

class GCL_classifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        z = self.encoder(x, edge_index, edge_weight = edge_weight)
        logits = self.classifier(z)
        return logits

def eval_encoder(model, y, x, edge_index, edge_weight=None, idx_test=None, device='cuda'):
    '''
    Workflow:
        graph -> encoder => embeddings -> classifier => predictions
    Return:
        accuracy: accuracy on the evaluation set
        correct_mask: to indicate the correctly classified nodes in the test set
    '''
    model.eval()

    z = model.encoder(x = x, edge_index = edge_index, edge_weight = edge_weight)
    if idx_test == None:
        idx_test = torch.tensor(range(len(x))).to(device) # use all nodes as the test set
    test_z = z[idx_test]
    test_y = y[idx_test]

    logits = model.classifier(test_z)
    preds = torch.argmax(logits, dim=1)
    correct_mask = (preds == test_y)
    accuracy = torch.sum(correct_mask).float() / test_y.shape[0]

    return accuracy, correct_mask

def arg_parse():
    parser = argparse.ArgumentParser(description='gcl_node.py')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset')
    parser.add_argument('--ratio', type=float, default='0.1',
                        help='Perterbation ratio in augmentation. Default: 0.1')
    parser.add_argument('--aug', type=str, default='edge',
                        help='Type of augmentation. Edge, mask, node, or subgraph. Default: edge')
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
    aug_pe = args.ratio
    aug_type = args.aug
    do_PGD_attack = args.PGD
    do_PRBCD_attack = args.PRBCD
    do_GRBCD_attack = args.GRBCD
    seed_split = args.seed_split
    seeds_encoder = args.seeds_encoder
    seeds_lc = args.seeds_lc
    do_normalize = args.normalize
    hid_units = 512
    print(f'=== On dataset: {dataset_name}, using augmentation type: {aug_type}, ratio: {aug_pe}, normalize: {do_normalize}')

    # load dataset
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = CitationFull(path, name=dataset_name)
    # dataset = Planetoid(path, name=dataset_name)
    assert len(dataset) == 1, "Expecting node classification on one huge graph"
    
    data = dataset[0]
    x = data.x
    y = data.y
    edge_index = data.edge_index
    n_features = dataset.num_features
    num_classes = dataset.num_classes
    num_nodes = x.shape[0]

    lbl_1 = torch.ones(num_nodes).unsqueeze(1)
    lbl_2 = torch.zeros(num_nodes).unsqueeze(1)
    double_lbl = torch.cat((lbl_1, lbl_2), 1)

    train_set, val_set, test_set = torch.utils.data.random_split(range(num_nodes), [0.8,0,0.2])
    idx_train = torch.tensor(train_set.indices)
    idx_val = torch.tensor(val_set.indices)
    idx_test = torch.tensor(test_set.indices)

    # augmentations
    augmentor = None
    if aug_type == 'mask':
        augmentor = MyFeatureMasking(aug_pe)
    elif aug_type == 'edge':
        augmentor = A.EdgeRemoving(aug_pe)
    elif aug_type == 'subgraph':
        subgraph_size = int((1 - aug_pe) * num_nodes)
        augmentor = A.RWSampling(num_seeds=1000, walk_length=subgraph_size)
    elif aug_type == 'node':
        augmentor = A.NodeDropping(0.5)

    x1, edge_index1, _ = augmentor(x, edge_index)
    x2, edge_index2, _ = augmentor(x, edge_index)
    if do_normalize:
        edge_index, edge_weight = get_laplacian(edge_index, normalization = 'sym')
        edge_index1, edge_weight1 = get_laplacian(edge_index1, normalization = 'sym')
        edge_index2, edge_weight2 = get_laplacian(edge_index2, normalization = 'sym')
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
        edge_index1, edge_weight1 = remove_self_loops(edge_index1, -edge_weight1)
        edge_index2, edge_weight2 = remove_self_loops(edge_index2, -edge_weight2)

    # model
    torch.manual_seed(seeds_encoder[0])
    torch.cuda.manual_seed(seeds_encoder[0])
    encoder = MyGCN(in_channels=n_features, hidden_channels=hid_units, num_layers=1, act='prelu')
    summary = AvgReadout()
    corruption = FeatureShuffle()
    model = DeepGraphInfomax(hidden_channels=hid_units, 
                            encoder=encoder,
                            summary=summary,
                            corruption=corruption).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # to device
    y = y.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    x1 = x1.to(device)
    edge_index1 = edge_index1.to(device)
    x2 = x2.to(device)
    edge_index2 = edge_index2.to(device)
    if do_normalize:
        edge_weight = edge_weight.to(device)
        edge_weight1 = edge_weight1.to(device)
        edge_weight2 = edge_weight2.to(device)
    else:
        edge_weight = None
        edge_weight1 = None
        edge_weight2 = None

    double_lbl = double_lbl.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    # train
    with tqdm(total=2, desc='(T)') as pbar:
        for epoch in range(2):
            model.train()

            pz, nz, _ = model(x, edge_index, edge_weight = edge_weight)
            _, _, s1 = model(x1, edge_index1, edge_weight = edge_weight1)
            _, _, s2 = model(x2, edge_index2, edge_weight = edge_weight2)

            double_logits1 = disc(s1, pz, nz, model)
            double_logits2 = disc(s2, pz, nz, model)
            double_logits = double_logits1 + double_logits2

            loss = b_xent(double_logits, double_lbl)
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'try.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == 20:
                print('Early stopping!')
                break

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            pbar.set_postfix({'loss': loss.item()})
            pbar.update()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('try.pkl'))

    # test the model
    model.eval()
    model.requires_grad_(False)
    z = model.encoder(x, edge_index, edge_weight=edge_weight)

    train_z = z[idx_train]
    train_y = y[idx_train]

    xent = nn.CrossEntropyLoss()
    accs_clean = []
    accs_greedy = []
    accs_PGD = []
    accs_PRBCD = []
    accs_GRBCD = []
    for _ in tqdm(range(2)): # use 50 linear classifier
        log = LogReg(hid_units, num_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for _ in range(2):
            log.train()
            opt.zero_grad()

            logits = log(train_z)
            loss = xent(logits, train_y)
            
            loss.backward()
            opt.step()

        encoder_classifier = GCL_classifier(model.encoder, log)
        encoder_classifier.eval()
        # ++++++++++ clean ++++++++++++++++
        acc, _ = eval_encoder(encoder_classifier, y, x, edge_index, edge_weight=edge_weight, idx_test=idx_test, device=device)
        accs_clean.append(acc)

        # ++++++++++ Greedy ++++++++++++++++
        attack_ratio = 0.05
        updated_edge_index, _ = dropout_edge(edge_index, p=attack_ratio, force_undirected=False) # Randomly drop edges
        _, added_edges = add_random_edge(edge_index, p=attack_ratio, force_undirected=False) # Randomly add edges
        updated_edge_index = torch.cat((updated_edge_index,added_edges), dim=1)

        updated_edge_weight = None
        if do_normalize:
            updated_edge_index, updated_edge_weight = get_laplacian(updated_edge_index, normalization = 'sym')
            updated_edge_index, updated_edge_weight = remove_self_loops(updated_edge_index, -updated_edge_weight)

        acc_greedy, _ = eval_encoder(encoder_classifier, y, x, updated_edge_index, edge_weight=updated_edge_weight, idx_test=idx_test, device=device)
        accs_greedy.append(acc_greedy)
        # ++++++++++ Greedy over ++++++++++++++++

        if do_PGD_attack == True:
            # ++++++++++ PGD ++++++++++++++++
            PGDattacker = PGDAttack(surrogate=encoder_classifier, device=device, epsilon=1e-4, log=False)
            budget = int(edge_index.shape[1] * attack_ratio)
            _, _, adv_edge_index = PGDattacker.attack_one_graph(x=x, edge_index=edge_index, batch=None, y=y, n_perturbations=budget)

            adv_edge_weight = None
            if do_normalize:
                adv_edge_index, adv_edge_weight = get_laplacian(adv_edge_index, normalization = 'sym')
                adv_edge_index, adv_edge_weight = remove_self_loops(adv_edge_index, -adv_edge_weight)
            
            acc_PGD, _ = eval_encoder(encoder_classifier, y, x, adv_edge_index, edge_weight=adv_edge_weight, idx_test=idx_test, device=device)
            accs_PGD.append(acc_PGD)
            # ++++++++++ PGD over ++++++++++++++++

        if do_PRBCD_attack == True:
            # ++++++++++ PRBCD ++++++++++++++++
            PRBCDattacker = PRBCDAttack(model=encoder_classifier, block_size=250_000, log=False)
            budget = int(edge_index.shape[1] * attack_ratio)
            adv_edge_index, _ = PRBCDattacker.attack(x=x, edge_index=edge_index, labels=y, budget=budget)

            adv_edge_weight = None
            if do_normalize:
                adv_edge_index, adv_edge_weight = get_laplacian(adv_edge_index, normalization = 'sym')
                adv_edge_index, adv_edge_weight = remove_self_loops(adv_edge_index, -adv_edge_weight)

            acc_PRBCD, _ = eval_encoder(encoder_classifier, y, x, adv_edge_index, edge_weight=adv_edge_weight, idx_test=idx_test, device=device)
            accs_PRBCD.append(acc_PRBCD)
            # ++++++++++ PRBCD over ++++++++++++++++

        if do_GRBCD_attack == True:
            # ++++++++++ PRBCD ++++++++++++++++
            GRBCDattacker = GRBCDAttack(model=encoder_classifier, block_size=250_000, log=False)
            budget = int(edge_index.shape[1] * attack_ratio)
            adv_edge_index, _ = GRBCDattacker.attack(x=x, edge_index=edge_index, labels=y, budget=budget)

            adv_edge_weight = None
            if do_normalize:
                adv_edge_index, adv_edge_weight = get_laplacian(adv_edge_index, normalization = 'sym')
                adv_edge_index, adv_edge_weight = remove_self_loops(adv_edge_index, -adv_edge_weight)

            acc_GRBCD, _ = eval_encoder(encoder_classifier, y, x, adv_edge_index, edge_weight=adv_edge_weight, idx_test=idx_test, device=device)
            accs_GRBCD.append(acc_GRBCD)
            # ++++++++++ PRBCD over ++++++++++++++++
    
    accs_clean = torch.stack(accs_clean)
    accs_greedy = torch.stack(accs_greedy)
    print(f'(A): clean average accuracy={accs_clean.mean():.4f}, std={accs_clean.std():.4f}')
    print(f'(A): greedy adversarial average accuracy={accs_greedy.mean():.4f}, std={accs_greedy.std():.4f}, drop percentage:{(accs_clean.mean() - accs_greedy.mean())/accs_clean.mean()}')
    if do_PGD_attack == True:
        accs_PGD = torch.stack(accs_PGD)
        print(f'(A): PGD adversarial average accuracy={accs_PGD.mean():.4f}, std={accs_PGD.std():.4f}, drop percentage:{(accs_clean.mean() - accs_PGD.mean())/accs_clean.mean()}')
    if do_PRBCD_attack == True:
        accs_PRBCD = torch.stack(accs_PRBCD)
        print(f'(A): PRBCD adversarial average accuracy={accs_PRBCD.mean():.4f}, std={accs_PRBCD.std():.4f}, drop percentage:{(accs_clean.mean() - accs_PRBCD.mean())/accs_clean.mean()}')
    if do_GRBCD_attack == True:
        accs_GRBCD = torch.stack(accs_GRBCD)
        print(f'(A): GRBCD adversarial average accuracy={accs_GRBCD.mean():.4f}, std={accs_GRBCD.std():.4f}, drop percentage:{(accs_clean.mean() - accs_GRBCD.mean())/accs_clean.mean()}')