import torch
import torch.nn as nn
from torch_geometric.nn.models import DeepGraphInfomax, GCN
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import get_laplacian

import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm

import GCL.augmentors as A
import copy

from aug.my_feature_masking import MyFeatureMasking
from layer.readout import AvgReadout
from layer.corrupt import FeatureShuffle
from gin import LogReg

def disc(summary_aug, pos, neg, DGI):
    pos_logits = DGI.discriminate(z = pos, summary = summary_aug, sigmoid = False)
    neg_logits = DGI.discriminate(z = neg, summary = summary_aug, sigmoid = False)
    return torch.cat((pos_logits.unsqueeze(1), neg_logits.unsqueeze(1)),1)

def arg_parse():
    parser = argparse.ArgumentParser(description='dgi.py')
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
    parser.add_argument('--seed_split', type=int, default=42,
                        help='Seed for the dataset split. Default: 42')
    parser.add_argument('--seeds_encoder', nargs='+', type=int, default=[1,2,3,4,5],
                        help='List of seeds for the encoder initialization. Default: [1,2,3,4,5]') 
    parser.add_argument('--seeds_lc', nargs='+', type=int, default=[1,2,3,4,5],
                        help='List of seeds for the classifier initialization. Default: [1,2,3,4,5]') 
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    dataset_name = args.dataset
    aug_pe = args.ratio
    aug_type = args.aug
    do_PGD_attack = args.PGD
    do_PRBCD_attack = args.PRBCD
    seed_split = args.seed_split
    seeds_encoder = args.seeds_encoder
    seeds_lc = args.seeds_lc
    hid_units = 512

    # load dataset
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = CitationFull(path, name=dataset_name)
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
    # edge_index, edge_weight = get_laplacian(edge_index)
    # edge_index1, edge_weight1 = get_laplacian(edge_index1)
    # edge_index2, edge_weight2 = get_laplacian(edge_index2)

    # model
    torch.manual_seed(seeds_encoder[0])
    torch.cuda.manual_seed(seeds_encoder[0])
    encoder = GCN(in_channels=n_features, hidden_channels=hid_units, num_layers=1, act='prelu')
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
    # edge_weight = edge_weight.to(device)
    x1 = x1.to(device)
    edge_index1 = edge_index1.to(device)
    # edge_weight1 = edge_weight1.to(device)
    x2 = x2.to(device)
    edge_index2 = edge_index2.to(device)
    # edge_weight2 = edge_weight2.to(device)

    double_lbl = double_lbl.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    # train
    with tqdm(total=10000, desc='(T)') as pbar:
        for epoch in range(10000):
            model.train()

            pz, nz, _ = model(x, edge_index)
            _, _, s1 = model(x1, edge_index1)
            _, _, s2 = model(x2, edge_index2)

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
    z = model.encoder(x, edge_index)

    train_z = z[idx_train]
    train_y = y[idx_train]
    test_z = z[idx_test]
    test_y = y[idx_test]

    xent = nn.CrossEntropyLoss()
    accs_clean = []
    for _ in tqdm(range(50)): # use 50 linear classifier
        log = LogReg(hid_units, num_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_z)
            loss = xent(logits, train_y)
            
            loss.backward()
            opt.step()

        logits = log(test_z)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_y).float() / test_y.shape[0]
        accs_clean.append(acc)

    accs_clean = torch.stack(accs_clean)
    print(f'(A): clean average accuracy={accs_clean.mean():.4f}, std={accs_clean.std():.4f}')