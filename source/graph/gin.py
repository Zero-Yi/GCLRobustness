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

from tqdm import tqdm
import itertools
import warnings
import sys
sys.path.append("/workspaces/mygcl/source")
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
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# Firstly define the model, from GraphCL
class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, dropout=0, device='cuda'): # num_feature: node feature维度，dim: node embedding维度
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
        self.device = device

        for i in range(num_gc_layers):

            if i:
                mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else: # 第一层在这里
                mlp = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = WGINConv(mlp)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)

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
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                # data = data[0]
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(self.device)
                x_g, _ = self.forward(x, edge_index, edge_weight=edge_weight, batch=batch) # 只取用global embedding

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
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(self.device)
                x_g, x = self.forward(x, edge_index, batch=batch)
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

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x_g, _ = self.encoder(x, edge_index, edge_weight=edge_weight, batch=batch)
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

        logits = encoder_model(data.x, data.edge_index, batch=data.batch)

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
        logits = model(batch.x, batch.edge_index, batch=batch.batch)
        preds.append(torch.argmax(logits, dim=1))
        ys.append(batch.y)
        
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)

    correct_mask = (preds == ys)
    accuracy = torch.sum(correct_mask).float() / len(ys)

    return accuracy, correct_mask

def arg_parse():
    parser = argparse.ArgumentParser(description='gin.py')
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='Dataset')
    parser.add_argument('--find_HP', type=bool, default=False,
                        help='Whether use 10-Fold cross validation to find hyperparams first. Default: false')
    parser.add_argument('--PGD', type=bool, default=False,
                        help='Whether apply PGD attack. Default: false')
    parser.add_argument('--PRBCD', type=bool, default=False,
                        help='Whether apply PRBCD attack. Default: false')
    parser.add_argument('--GRBCD', type=bool, default=False,
                        help='Whether apply GRBCD attack. Default: false')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the dataset split and model initialization. Default: 42')                
    return parser.parse_args()

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    args = arg_parse()

    dataset_name = args.dataset
    find_hyperparams = args.find_HP
    do_PGD_attack = args.PGD
    do_PRBCD_attack = args.PRBCD
    do_GRBCD_attack = args.GRBCD
    seed = args.seed

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
    setup_seed(seed) # set seed for the reproducibility
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
            encoder_model = GIN(num_features=num_features, dim=hidden_dim, num_gc_layers=num_layer, dropout=dropout, device=device).to(device)
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
        best_hyperparams = {'lr': 0.01, 'num_layer': 5, 'hidden_dim': 64, 'dropout': 0, 'batch_size': 128}
        lr, num_layer, hidden_dim, dropout, batch_size = best_hyperparams.values()

    print(f'=== On dataset: {dataset_name}, used hyperparams: {best_hyperparams}, whether activate PGD attack: {do_PGD_attack} ===')
    

    dataloader_train = DataLoader(train_val_set, batch_size=batch_size, shuffle=True) # Use all train+val to train the final model
    dataloader_eval = DataLoader(eval_set, batch_size=128, shuffle=False) # Do not shuffle the evaluation set to make it reproduceable
    
    # The graph neural network backbone model to use
    setup_seed(seed) # set seed for the reproducibility
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
    
    model.eval() # Try to save memory
    model.requires_grad_(False) # Try to save memory

    runs = 5

    accs_clean = []
    accs_adv_PGD = []
    accs_adv_PRBCD = []
    accs_adv_GRBCD = []
    accs_adv_greedy = []
    for _ in range(runs):

        # Accuracy on the clean evaluation data
        acc_clean, mask = eval_encoder(model, dataloader_eval)
        accs_clean.append(acc_clean)

        # Instantiate an attacker and attack the victim model
        # ++++++++++ Greedy ++++++++++++++++
        Greedyattacker = Greedy(pn=0.05)
        dataloader_eval_adv_greedy = Greedyattacker.attack(eval_set, mask)

        # Accuracy on the adversarial data only
        acc_adv_only_greedy, _ = eval_encoder(model, dataloader_eval_adv_greedy)

        # Overall adversarial accuracy
        acc_adv_greedy = acc_clean * acc_adv_only_greedy # T/all * Tadv/T = Tadv/all
        accs_adv_greedy.append(acc_adv_greedy)
        # ++++++++++ Greedy over ++++++++++++++++

        if do_PGD_attack == True:
        # ++++++++++ PGD ++++++++++++++++
            PGDattacker = PGDAttack(surrogate=model, device=device, epsilon=1e-4)
            dataloader_eval_adv_PGD = PGDattacker.attack(eval_set, mask, attack_ratio=0.05)

            # Accuracy on the adversarial data only
            acc_adv_only_PGD, _ = eval_encoder(model, dataloader_eval_adv_PGD)

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

        torch.cuda.empty_cache() # Try to save memory
        
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
    if do_GRBCD_attack == True:
        accs_adv_GRBCD = torch.stack(accs_adv_GRBCD)
        print(f'(A): GRBCD adversarial average accuracy={accs_adv_GRBCD.mean():.4f}, std={accs_adv_GRBCD.std():.4f}')

