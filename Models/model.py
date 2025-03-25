import torch
import torch.nn as nn
from Models.gcn import GCN

class CoraModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__()

        self.gcn1 = GCN(in_features=in_features, out_features=hidden_features)
        self.gcn2 = GCN(in_features=hidden_features, out_features=out_features)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x_ = self.relu(x)
        x = self.dropout(x_)
        x = self.gcn2(x, adj)

        return x, x_
    

class CoraTGS(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__()



