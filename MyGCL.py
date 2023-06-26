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

from tqdm import tqdm
import itertools
import warnings
import sys
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os.path as osp
import argparse

from attacker.greedy import Greedy
from attacker.PGD import PGDAttack
from attacker.PRBCD import MyPRBCDAttack
from wgin_conv import WGINConv

from gin import GIN, LogReg, GCL_classifier, eval_encoder

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        eg , en  = self.encoder(x, edge_index, batch=batch)
        eg1, en1 = self.encoder(x1, edge_index1, batch=batch)
        eg2, en2 = self.encoder(x2, edge_index2, batch=batch)
        return en, eg, en1, en2, eg1, eg2

# Training in every epoch
def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, batch=data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        torch.cuda.empty_cache() # Added by Zinuo to try to save the memory
    return epoch_loss

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
    torch.manual_seed(seed)
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
    do_PGD_attack = args.PGD
    do_PRBCD_attack = args.PRBCD
    seed_split = args.seed_split
    seeds_encoder = args.seeds_encoder
    seeds_lc = args.seeds_lc

    # Hyperparams
    lr = 0.01
    num_layers = 3
    epochs = 20
    print(f'====== lr:{lr}, num_layers:{num_layers}, epochs:{epochs}, dataset:{dataset_name}, PGD attack:{do_PGD_attack}, PRBCD attack:{do_PRBCD_attack}======')

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    num_features = max(dataset.num_features, 1)
    num_classes = dataset.num_classes
    if dataset.num_features==0 :
        print("No node feature, paddings of 1 will be used in GIN when forwarding.")

    # Split the dataset into two part for training classifier and final evaluation, train_set can be further divided into training and validation parts
    torch.manual_seed(seed_split) # set seed for the reproducibility
    train_set, eval_set = random_split(dataset, [0.9, 0.1])
    dataloader_train = DataLoader(train_set, batch_size=128, shuffle=True)
    dataloader_eval = DataLoader(eval_set, batch_size=128, shuffle=False) # Do not shuffle the evaluation set to make it reproduceable

    # Define the augmentations
    aug1 = A.Identity()
    aug2 = autoAug(dataset_name)

    list_encoders = []

    for seed_encoder in seeds_encoder: 

        torch.manual_seed(seed_encoder) # set seed for the reproducibility
        gconv = GIN(num_features=num_features, dim=32, num_gc_layers=num_layers, device=device).to(device)
        torch.manual_seed(seed_encoder) # set seed for the reproducibility
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
        torch.manual_seed(seed_encoder) # set seed for the reproducibility
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
        optimizer = Adam(encoder_model.parameters(), lr=lr)

        # Train the encoder with full dataset without labels using contrastive learning
        with tqdm(total=20, desc='(T)') as pbar:
            for epoch in range(1, epochs + 1):
                loss = train(encoder_model, contrast_model, dataloader, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

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
                # ++++++++++ PRBCD over ++++++++++++++++

            print(f'(A): clean accuracy={acc_clean:.4f}, greedy adversarial accuracy={acc_adv_greedy:.4f}')
            del Greedyattacker # Try to save memory

            if do_PGD_attack == True:
                print(f'(A): PGD adversarial accuracy={acc_adv_PGD:.4f}')
                del PGDattacker # Try to save memory
            if do_PRBCD_attack == True:
                print(f'(A): PRBCD adversarial accuracy={acc_adv_PRBCD:.4f}')
                del PRBCDattacker # Try to save memory

            print(f'==== Trial with LC seed {seed_lc} finished. ====')
                
        accs_clean = torch.stack(accs_clean)
        accs_adv_greedy = torch.stack(accs_adv_greedy)
        print(f'(A): clean average accuracy={accs_clean.mean():.4f}, std={accs_clean.std():.4f}')
        print(f'(A): greedy adversarial average accuracy={accs_adv_greedy.mean():.4f}, std={accs_adv_greedy.std():.4f}')
        if do_PGD_attack == True:
            accs_adv_PGD = torch.stack(accs_adv_PGD)
            print(f'(A): PGD adversarial average accuracy={accs_adv_PGD.mean():.4f}, std={accs_adv_PGD.std():.4f}')
        if do_PRBCD_attack == True:
            accs_adv_PRBCD = torch.stack(accs_adv_PRBCD)
            print(f'(A): PRBCD adversarial average accuracy={accs_adv_PRBCD.mean():.4f}, std={accs_adv_PRBCD.std():.4f}')
        
        list_encoders.append({'accs_clean': accs_clean.mean(), 
                                    'accs_adv_greedy': accs_adv_greedy.mean(),
                                    'accs_adv_PGD': accs_adv_PGD.mean(),
                                    'accs_adv_PRBCD': accs_adv_PRBCD.mean()})
        print(f'==== Trial with encoder seed {seed_encoder} finished. ====')
    
    for i in range(len(seeds_encoder)):
        print(f'Result for seed {seeds_encoder[i]}: {list_encoders[i]}')
