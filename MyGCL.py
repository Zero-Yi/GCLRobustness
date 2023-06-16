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
from wgin_conv import WGINConv

from gin import GIN, LogReg, GCL_classifier, eval_encoder

# Firstly define the model, from GraphCL
# class GIN(torch.nn.Module):
#     def __init__(self, num_features, dim, num_gc_layers, dropout=0): # num_feature: node feature维度，dim: node embedding维度
#         super(GIN, self).__init__()

#         self.num_gc_layers = num_gc_layers

#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()

#         project_dim = dim * num_gc_layers
#         self.project = torch.nn.Sequential( # 这个projector也是会在training过程中被训练的
#             nn.Linear(project_dim, project_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(project_dim, project_dim)
#             )

#         self.dropout = dropout

#         for i in range(num_gc_layers):

#             if i:
#                 mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#             else: # 第一层在这里
#                 mlp = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
#             conv = WGINConv(mlp)
#             bn = torch.nn.BatchNorm1d(dim)

#             self.convs.append(conv)
#             self.bns.append(bn)

#     def forward(self, x, edge_index, batch, edge_weight=None):
#         if x is None:
#             x = torch.ones((batch.shape[0], 1)).to(device)

#         xs = []
#         for i in range(self.num_gc_layers):

#             x = F.relu(self.convs[i](x, edge_index, edge_weight))
#             x = self.bns[i](x)
#             F.dropout(x, p=self.dropout, training=self.training)
#             xs.append(x) # xs is the node representation
#             # if i == 2:
#                 # feature_map = x2

#         xpool = [global_add_pool(x, batch) for x in xs]
#         x_g = torch.cat(xpool, 1) # g is the global representation

#         return x_g, torch.cat(xs, 1) # global embeddings, nodes embeddings

#     def get_embeddings(self, loader):
#         '''
#         Returns:
#         ret: Tensor(num_graphs, global_embedding_features)
#         y: Tensor(num_graphs,)
#         '''
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         ret = []
#         y = []
#         with torch.no_grad():
#             for data in loader:

#                 # data = data[0]
#                 data.to(device)
#                 x, edge_index, batch = data.x, data.edge_index, data.batch
#                 edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

#                 if x is None:
#                     x = torch.ones((batch.shape[0],1)).to(device)
#                 x_g, _ = self.forward(x, edge_index, batch, edge_weight) # 只取用global embedding

#                 ret.append(x_g)
#                 y.append(data.y)
#         ret = torch.cat(ret, dim=0)
#         y = torch.cat(y, dim=0)
#         return ret, y

#     def get_embeddings_v(self, loader):
#         '''
#         Returns:
#         x_g: Tensor(batch_size, global_embedding_features)
#         ret: Tensor(num_nodes, node_embedding_features)
#         y: Tensor(batch_size,)
#         '''
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         ret = []
#         y = []
#         with torch.no_grad():
#             for n, data in enumerate(loader):
#                 data.to(device)
#                 x, edge_index, batch = data.x, data.edge_index, data.batch
#                 if x is None:
#                     x = torch.ones((batch.shape[0],1)).to(device)
#                 x_g, x = self.forward(x, edge_index, batch)
#                 x_g = x_g
#                 ret = x
#                 y = data.edge_index
#                 print(data.y)
#                 if n == 1: # 只取用一个batch做测试
#                    break

#         return x_g, ret, y

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        eg , en  = self.encoder(x, edge_index, batch)
        eg1, en1 = self.encoder(x1, edge_index1, batch)
        eg2, en2 = self.encoder(x2, edge_index2, batch)
        return en, eg, en1, en2, eg1, eg2

# Linear classifier to be used
# class LogReg(nn.Module):
#     def __init__(self, ft_in, nb_classes):
#         super(LogReg, self).__init__()
#         self.fc = nn.Linear(ft_in, nb_classes)

#         for m in self.modules():
#             self.weights_init(m)

#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)

#     def forward(self, seq):
#         ret = self.fc(seq)
#         return ret

# class GCL_classifier(torch.nn.Module):
#     """
#         To seal the encoder and classifier as a whole body and provide an end-to-end model.
#     """
#     def __init__(self, encoder, classifier):
#         """
#         Params:
#             encoder: Used encoder
#             classifier: Used classifier
#         """
#         super(GCL_classifier, self).__init__()

#         self.encoder = encoder
#         self.classifier = classifier

#     def forward(self, x, edge_index, batch, edge_weight=None):
#         x_g, _ = self.encoder(x, edge_index, batch, edge_weight)
#         logits = self.classifier(x_g)
#         return logits

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

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        torch.cuda.empty_cache() # Added by Zinuo to try to save the memory
    return epoch_loss

def train_classifier(x, y, num_classse=2, epoches=100, lr=0.01, device='cuda'):
    '''
    Params:
    x: the embeddings
    y: the labels
    num_classse: the dimension for the output layer
    epoches: the epoches to train the classifier
    '''
    x.to(device)
    y.to(device)
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

# def eval_encoder(model, dataloader_eval, device='cuda'):
#     '''
#     Workflow:
#         eval_set -> encoder => embeddings -> classifier => predictions
#     Return:
#         accuracy: accuracy on the evaluation set
#         correct_mask: to indicate the correctly classified samples
#     '''
#     model.eval()

#     ys = []
#     preds = []

#     for batch in dataloader_eval:
#         batch.to(device)
#         logits = model(batch.x, batch.edge_index, batch.batch)
#         preds.append(torch.argmax(logits, dim=1))
#         ys.append(batch.y)
        
#     preds = torch.cat(preds, dim=0)
#     ys = torch.cat(ys, dim=0)

#     correct_mask = (preds == ys)
#     accuracy = torch.sum(correct_mask).float() / len(ys)

#     return accuracy, correct_mask

def arg_parse():
    parser = argparse.ArgumentParser(description='gin.py')
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='Dataset')
    parser.add_argument('--multi_classifiers', type=bool, default=False,
                        help='Whether to train different classifiers when evaluating the encoder. Default: false')
    parser.add_argument('--PGD', type=bool, default=False,
                        help='Whether apply PGD attack. Default: false')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    dataset_name = args.dataset
    train_multiple_classifiers = args.multi_classifiers
    do_PGD_attack = args.PGD

    # Hyperparams
    lr = 0.01
    num_layers = 3
    epochs = 20
    print(f'======The hyperparams: lr={lr}, num_layers={num_layers}, epochs={epochs}. On dataset:{dataset_name}, whether activate PGD attack: {do_PGD_attack}======')

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    num_features = max(dataset.num_features, 1)
    num_classes = dataset.num_classes
    if dataset.num_features==0 :
        print("No node feature, paddings of 1 will be used in GIN when forwarding.")

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=86),
                           A.EdgeRemoving(pe=0.2),
                           A.NodeDropping(pn=0.2)], 1)

    # The graph neural network backbone model to use
    gconv = GIN(num_features=num_features, dim=32, num_gc_layers=num_layers, device=device).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr)

    # Train the encoder with full dataset without labels using contrastive learning
    with tqdm(total=20, desc='(T)') as pbar:
        for epoch in range(1, epochs + 1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    # Save trained model parameters for reproduce if needed
    # torch.save(encoder_model.state_dict(), 'Savings/model_params/model.pt')
    
    # Split the dataset into two part for training classifier and final evaluation, train_set can be further divided into training and validation parts
    generator = torch.Generator().manual_seed(42) # Fix the seed to do fair comparation
    train_set, eval_set = random_split(dataset, [0.9, 0.1], generator=generator)
    dataloader_train = DataLoader(train_set, batch_size=128, shuffle=True)
    dataloader_eval = DataLoader(eval_set, batch_size=128, shuffle=False) # Do not shuffle the evaluation set to make it reproduceable
    
    # Get embeddings for the train_set
    encoder_model.eval()
    embedding_global, y = encoder_model.encoder.get_embeddings(dataloader_train)

    # Save the embeddings for reproduce if needed
    # torch.save(embedding_global, 'Savings/global_embeddings/embeddings.pt')
    # torch.save(y, 'Savings/global_embeddings/y.pt')    

    if train_multiple_classifiers == True:
        # ====== Train multiple classifiers and take the average acc ======
        num_calssifier = 50
        accs=[]
        with tqdm(total=num_calssifier, desc='(E)') as pbar:
            for _ in range(num_calssifier): 
                # Train the downstream classifier
                classifier = train_classifier(embedding_global, y, num_classse=num_classes)

                # Put encoder and classifier together, drop the augmentor
                encoder_classifier = GCL_classifier(encoder_model.encoder, classifier)

                # Evaluation
                acc, _ = eval_encoder(encoder_classifier, dataloader_eval)
                accs.append(acc*100)

                pbar.set_postfix({'acc': acc})
                pbar.update()

        accs = torch.stack(accs)
        print(f'(E): average accuracy={accs.mean():.4f}, std={accs.std():.4f}')
    else:

        # ====== Train one classifier and do attack =====================================
        classifier = train_classifier(embedding_global, y, num_classse=num_classes)

        # Put encoder and classifier together, drop the augmentor
        encoder_classifier = GCL_classifier(encoder_model.encoder, classifier)
        
        encoder_classifier.eval() # Try to save memory
        encoder_classifier.requires_grad_(False) # Try to save memory

        runs = 5

        accs_clean = []
        accs_adv_PGD = []
        accs_adv_greedy = []
        for _ in range(runs):

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
                PGDattacker = PGDAttack(surrogate=encoder_classifier, device=device, epsilon=1e-4)
                dataloader_eval_adv_PGD = PGDattacker.attack(eval_set, mask, attack_ratio=0.05)

                # Accuracy on the adversarial data only
                acc_adv_only_PGD, _ = eval_encoder(encoder_classifier, dataloader_eval_adv_PGD)

                # Overall adversarial accuracy
                acc_adv_PGD = acc_clean * acc_adv_only_PGD # T/all * Tadv/T = Tadv/all
                accs_adv_PGD.append(acc_adv_PGD)
            # ++++++++++ PGD over ++++++++++++++++

            if do_PGD_attack == True:
                print(f'(A): clean accuracy={acc_clean:.4f}, greedy adversarial accuracy={acc_adv_greedy:.4f}, PGD adversarial accuracy={acc_adv_PGD:.4f}')
                del Greedyattacker, PGDattacker # Try to save memory
            else:
                print(f'(A): clean accuracy={acc_clean:.4f}, greedy adversarial accuracy={acc_adv_greedy:.4f}')
                del Greedyattacker # Try to save memory

            torch.cuda.empty_cache() # Try to save memory
            
        accs_clean = torch.stack(accs_clean)
        accs_adv_greedy = torch.stack(accs_adv_greedy)
        print(f'(A): clean average accuracy={accs_clean.mean():.4f}, std={accs_clean.std():.4f}')
        print(f'(A): greedy adversarial average accuracy={accs_adv_greedy.mean():.4f}, std={accs_adv_greedy.std():.4f}')
        if do_PGD_attack == True:
            accs_adv_PGD = torch.stack(accs_adv_PGD)
            print(f'(A): PGD adversarial average accuracy={accs_adv_PGD.mean():.4f}, std={accs_adv_PGD.std():.4f}')

