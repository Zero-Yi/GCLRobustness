import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split

import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast

from torch_geometric.nn import GINConv, global_add_pool, summary
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm
import itertools
import warnings
import sys
import numpy as np
import os.path as osp

from attacker.greedy import Greedy
from wgin_conv import WGINConv

# Firstly define the model, from GraphCL
class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, dropout=0): # num_feature: node feature维度，dim: node embedding维度
        super(GIN, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        project_dim = dim * num_gc_layers
        self.project = torch.nn.Sequential( # 这个projector也是会在training过程中被训练的
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim)
            )

        self.dropout = dropout

        for i in range(num_gc_layers):

            if i:
                mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else: # 第一层在这里
                mlp = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = WGINConv(mlp)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.bns[i](x)
            F.dropout(x, p=self.dropout, training=self.training)
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
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, _ = self.forward(x, edge_index, batch, edge_weight) # 只取用global embedding

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

class GCL_classifier(torch.nn.Module):
    """
        To seal the encoder and classifier as a whole body and provide an end-to-end model.
    """
    def __init__(self, encoder, classifier):
        """
        Params:
            encoder: Used encoder
            classifier: Used classifier
        """
        super(GCL_classifier, self).__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, edge_index, batch, edge_weight=None):
        x_g, _ = self.encoder(x, edge_index, batch, edge_weight)
        logits = self.classifier(x_g)
        return logits

# Training in every epoch
def train(encoder_model, dataloader, optimizer, scheduler=None):
    encoder_model.train()
    epoch_loss = 0

    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        logits = encoder_model(data.x, data.edge_index, data.batch)

        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        epoch_loss += loss.item()
    return epoch_loss

def eval_encoder(model, dataloader_eval, device='cuda'):
    '''
    Workflow:
        eval_set -> encoder => embeddings -> classifier => predictions
    Return:
        accuracy: accuracy on the evaluation set
        correct_mask: to indicate the correctly classified samples
    '''
    model.eval()

    ys = []
    preds = []

    for batch in dataloader_eval:
        batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds.append(torch.argmax(logits, dim=1))
        ys.append(batch.y)
        
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)

    correct_mask = (preds == ys)
    accuracy = torch.sum(correct_mask).float() / len(ys)

    return accuracy, correct_mask



if __name__ == '__main__':
    dataset_name = 'NCI1'
    find_hyperparams = False

    # Hyperparams
    lrs = [0.01]
    num_layers = [5]
    hidden_dims = [64] # {16, 32} for bio-graphs and 64 for social graphs
    dropouts = [0, 0.5]
    batch_sizes = [32, 128]
    epochs = 20

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=dataset_name)

    # Split the dataset into two part for training classifier and final evaluation, train_val_set can be further divided into training and validation parts
    train_val_set, eval_set = random_split(dataset, [0.9, 0.1])
    

    num_features = max(dataset.num_features, 1)
    num_classes = dataset.num_classes
    if dataset.num_features==0 :
        print("No node feature, paddings of 1 will be used in GIN when forwarding.")

    if find_hyperparams==True:
        # Do grid search on hyperparams with K-Fold validation
        best_hyperparams = {}
        best_acc_val = 0
        for lr, num_layer, hidden_dim, dropout, batch_size in itertools.product(lrs, num_layers, hidden_dims, dropouts, batch_sizes):
            
            print(f'======The hyperparams: lr={lr}, num_layers={num_layer}, hidden_dim={hidden_dim}, dropout={dropout}, batch_size={batch_size}. On dataset:{dataset_name}======')
            # Define model
            encoder_model = GIN(num_features=num_features, dim=hidden_dim, num_gc_layers=num_layer, dropout=dropout).to(device)
            classifier = LogReg(hidden_dim * num_layer, num_classes).to(device)
            model = GCL_classifier(encoder_model, classifier)
            optimizer = Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
            
            # K-Fold
            splitor = StratifiedKFold(n_splits=10, shuffle=True)

            dataset_train_val = dataset[train_val_set.indices] # This is necessary to construct a dataset object
            n_samples = len(dataset_train_val)

            accs_fold = []
            for i, (train_index, val_index) in enumerate(splitor.split(np.zeros(n_samples), dataset_train_val.y.cpu().numpy())):
                # Further split the dataset into train and val set
                dataloader_train = DataLoader(dataset_train_val[train_index], batch_size=batch_size, shuffle=True)
                dataloader_eval = DataLoader(dataset_train_val[val_index], batch_size=batch_size, shuffle=False)
                # Train the model
                for epoch in range(1, epochs + 1):
                    loss = train(model, dataloader_train, optimizer, scheduler)
                # Get the val accuracy
                acc_fold, _ = eval_encoder(model, dataloader_eval, device=device)
                accs_fold.append(acc_fold)
            
            acc_val = sum(accs_fold)/len(accs_fold)
            print(f'(CV): accuracy for this combination={acc_val:.4f}')

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_hyperparams = {'lr': lr, 
                                    'num_layer': num_layer, 
                                    'hidden_dim': hidden_dim, 
                                    'dropout': dropout, 
                                    'batch_size': batch_size}

        print("Best hyperparams: ", best_hyperparams)


    # Train and test. Run multilple times
    if find_hyperparams==True:
        lr, num_layer, hidden_dim, dropout, batch_size = best_hyperparams.values()
    else:
        best_hyperparams = {'lr': 0.01, 'num_layer': 5, 'hidden_dim': 32, 'dropout': 0, 'batch_size': 128}
        lr, num_layer, hidden_dim, dropout, batch_size = best_hyperparams.values()
    runs = 5

    dataloader_train = DataLoader(train_val_set, batch_size=batch_size, shuffle=True) # Use all train+val to train the final model
    dataloader_eval = DataLoader(eval_set, batch_size=128, shuffle=False) # Do not shuffle the evaluation set to make it reproduceable

    accs_clean = []
    accs_adv_PGD = []
    accs_adv_greedy = []
    for _ in range(runs):

        # The graph neural network backbone model to use
        encoder_model = GIN(num_features=num_features, dim=hidden_dim, num_gc_layers=num_layer, dropout=dropout).to(device)
        classifier = LogReg(hidden_dim * num_layer, num_classes).to(device)
        model = GCL_classifier(encoder_model, classifier)
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        # Train the encoder with full dataset without labels using contrastive learning
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1, epochs + 1):
                loss = train(model, dataloader_train, optimizer, scheduler)
                pbar.set_postfix({'loss': loss})
                pbar.update()
        
        # Accuracy on the clean evaluation data
        acc_clean, mask = eval_encoder(model, dataloader_eval)

        # Instantiate an attacker and attack the victim model
        # ++++++++++ Greedy ++++++++++++++++
        Greedyattacker = Greedy(pn=0.05)
        dataloader_eval_adv_greedy = Greedyattacker.attack(eval_set, mask)

        # Accuracy on the adversarial data only
        acc_adv_only_greedy, _ = eval_encoder(model, dataloader_eval_adv_greedy)

        # Overall adversarial accuracy
        acc_adv_greedy = acc_clean * acc_adv_only_greedy # T/all * Tadv/T = Tadv/all
        # ++++++++++ Greedy over ++++++++++++++++

        # ++++++++++ PGD ++++++++++++++++
        PGDattacker = PGDAttack(surrogate=model, device=device)
        dataloader_eval_adv_PGD = PGDattacker.attack(eval_set, mask, attack_ratio=0.05)

        # Accuracy on the adversarial data only
        acc_adv_only_PGD, _ = eval_encoder(model, dataloader_eval_adv_PGD)

        # Overall adversarial accuracy
        acc_adv_PGD = acc_clean * acc_adv_only_PGD # T/all * Tadv/T = Tadv/all
        # ++++++++++ PGD over ++++++++++++++++

        print(f'(A): clean accuracy={acc_clean:.4f}, greedy adversarial accuracy={acc_adv_greedy:.4f}, PGD adversarial accuracy={acc_adv_PGD:.4f}')
        # ====== Train one classifier =====================================
        accs_clean.append(acc_clean)
        accs_adv_greedy.append(acc_adv_greedy)
        accs_adv_PGD.append(acc_adv_PGD)

    accs_clean = torch.stack(accs_clean)
    accs_adv_greedy = torch.stack(accs_adv_greedy)
    accs_adv_PGD = torch.stack(accs_adv_PGD)
    print(f'(A): clean average accuracy={accs_clean.mean():.4f}, std={accs_clean.std():.4f}')
    print(f'(A): greedy adversarial average accuracy={accs_adv_greedy.mean():.4f}, std={accs_adv_greedy.std():.4f}')
    print(f'(A): PGD adversarial average accuracy={accs_adv_PGD.mean():.4f}, std={accs_adv_PGD.std():.4f}')

