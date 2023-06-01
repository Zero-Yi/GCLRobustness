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

from attacker import Greedy

# Firstly define the model, from GraphCL
class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers): # num_feature: node feature维度，dim: node embedding维度
        super(GIN, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        project_dim = dim * num_gc_layers
        self.project = torch.nn.Sequential( # 这个projector也是会在training过程中被训练的
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim)
            )

        for i in range(num_gc_layers):

            if i:
                mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else: # 第一层在这里
                mlp = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(mlp)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x) # xs is the node representation
            # if i == 2:
                # feature_map = x2

        xpool = [global_add_pool(x, batch) for x in xs]
        x_g = torch.cat(xpool, 1) # g is the global representation

        return x_g, torch.cat(xs, 1) # global embeddings, nodes embeddings

    def get_embeddings(self, loader):
        '''
        Returns:
        ret: Tensor(num_graphs, global_embedding_features)
        y: Tensor(num_graphs,)
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                # data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, _ = self.forward(x, edge_index, batch) # 只取用global embedding

                ret.append(x_g)
                y.append(data.y)
        ret = torch.cat(ret, dim=0)
        y = torch.cat(y, dim=0)
        return ret, y

    def get_embeddings_v(self, loader):
        '''
        Returns:
        x_g: Tensor(batch_size, global_embedding_features)
        ret: Tensor(num_nodes, node_embedding_features)
        y: Tensor(batch_size,)
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g
                ret = x
                y = data.edge_index
                print(data.y)
                if n == 1: # 只取用一个batch做测试
                   break

        return x_g, ret, y

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
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

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

def train_classifier(x, y, epoches=100, lr=0.01):
    '''
    Params:
    x: the embeddings
    y: the labels
    epoches: the epoches to train the classifier
    '''
    classifier = LogReg(x.shape[1], y.shape[0])
    xent = CrossEntropyLoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=0.0)
    classifier.cuda()

    classifier.train()
    for _ in range(epoches):
        opt.zero_grad()

        logits = classifier(x)
        loss = xent(logits, y)
        
        loss.backward()
        opt.step()

    classifier.eval()
    return classifier

def eval_encoder(encoder, classifier, dataloader_eval):
    '''
    Workflow:
        eval_set -> encoder => embeddings -> classifier => predictions
    Return:
        accuracy: accuracy on the evaluation set
        correct_mask: to indicate the correctly classified samples
    '''
    encoder.eval()
    classifier.eval()

    emb, y = encoder.encoder.get_embeddings(dataloader_eval)

    logits = classifier(emb)
    preds = torch.argmax(logits, dim=1)
    correct_mask = (preds == y)
    accuracy = torch.sum(correct_mask).float() / y.shape[0]

    return accuracy, correct_mask

#====================================================================
#====================================================================

if __name__ == '__main__':
    dataset_name = 'PROTEINS'
    train_multiple_classifiers = False

    # Hyperparams
    lr = 0.01
    num_layers = 3
    epochs = 20
    print(f'======The hyperparams: lr={lr}, num_layers={num_layers}, epochs={epochs}. On dataset:{dataset_name}======')

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    num_features = max(dataset.num_features, 1)
    if dataset.num_features==0 :
        print("No node feature, paddings of 1 will be used in GIN when forwarding.")

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=86),
                           A.EdgeRemoving(pe=0.2),
                           A.NodeDropping(pn=0.2)], 1)

    # The graph neural network backbone model to use
    gconv = GIN(num_features=num_features, dim=32, num_gc_layers=num_layers).to(device)

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
    train_set, eval_set = random_split(dataset, [0.9, 0.1])
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
                classifier = train_classifier(embedding_global, y)

                # Evaluation
                acc, _ = eval_encoder(encoder_model, classifier, dataloader_eval)
                accs.append(acc*100)

                pbar.set_postfix({'acc': acc})
                pbar.update()

        accs = torch.stack(accs)
        print(f'(E): average accuracy={accs.mean():.4f}, std={accs.std():.4f}')
    else:
        # ====== Train one classifier and do attack =====================================

        classifier = train_classifier(embedding_global, y)
        # Accuracy on the clean evaluation data
        acc_clean, mask = eval_encoder(encoder_model, classifier, dataloader_eval)

        # Instantiate an attacker and attack the victim model
        attacker = Greedy(encoder_model, classifier, pn=0.5)
        dataloader_eval_adv = attacker.attack(eval_set, mask)

        # Accuracy on the adversarial data only
        acc_adv_only, _ = eval_encoder(encoder_model, classifier, dataloader_eval_adv)

        # Overall adversarial accuracy
        acc_adv = acc_clean * acc_adv_only # T/all * Tadv/T = Tadv/all

        print(f'(A): clean accuracy={acc_clean:.4f}, adversarial accuracy={acc_adv:.4f}')
        # ====== Train one classifier =====================================
