import torch
import os.path as osp
import GCL.losses as L

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
import torch_geometric
from torch_geometric.datasets import TUDataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from SVM_classify import *
import pandas as pd
import numpy as np

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    x_attack, y_attack = attack_test(encoder_model, dataloader)
    result = SVMEvaluator(linear=True)(x, y, split)
    attack_result = SVMEvaluator(linear=True)(x_attack, y_attack, split)
    print(result)
    return result, attack_result
def attack_test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        updated_edge_index, _ = torch_geometric.utils.dropout_edge(data.edge_index, p = 0.1)
        data.put_edge_index(updated_edge_index, layout='coo')
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    return x,y


def main(dataset_name):
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=8)
    input_dim = max(dataset.num_features, 1)

    gconv = GConv(input_dim=input_dim, hidden_dim=256, activation=torch.nn.ReLU, num_layers=2).to(device)
    fc1 = FC(hidden_dim=256 * 2)
    fc2 = FC(hidden_dim=256 * 2)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
    #test_result = test(encoder_model, dataloader)
    clf, correctly_classified_indices, x_test, y_test, accuracy = svm_build(dataloader, encoder_model)
    att_acc = random_attack(encoder_model, clf, correctly_classified_indices, x_test, y_test, dataloader)
    return accuracy, att_acc

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='Dataset')  
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    dataset_name = args.dataset
    accs = []
    att_accs=[]
    for _ in range(7):
        acc, att_acc = main(dataset_name)
        accs.append(acc)
        att_accs.append(att_acc)
    df = pd.DataFrame(np.array([accs, att_accs]).T, columns = ['accuracy', 'attack_accuracy'])
        df.to_csv(dataset)
        print(df.mean(axis=0))
        print(df.std(axis=0))