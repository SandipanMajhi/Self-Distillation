import torch
import torch.nn as nn

from abc import ABC


class TGSBackbone(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim, params):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_dim))

        for _ in range(params["n_layers"]):
            self.layers.append(nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(params["dropout"]),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        self.inf = nn.Sequential(
                                    nn.ReLU(),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.Dropout(params["dropout"]),
                                    nn.Linear(hidden_dim, out_dim)
                                )
        

        self.out = nn.Sequential(
                                    nn.ReLU(),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.Dropout(params["dropout"]),
                                    nn.Linear(hidden_dim, out_dim)
                                )
        

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.out(x), self.inf(x)
    


class EdgeAdapter(nn.Module):
    def __init__(self, in_features, hidden_dim, param):
        super().__init__()

        self.fc_layer = nn.Linear(in_features, hidden_dim)
        self.attn = nn.Linear(2*hidden_dim, 1, bias = False)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.param = param

    
    def forward(self, src, dest):
        
        x_src = self.norm(self.fc_layer(src))
        x_dest = self.norm(self.fc_layer(dest))

        h = self.attn(torch.cat([x_src, x_dest], dim = 1))
        ratio = self.sigmoid(h)

        return ratio * (1 - self.params["ratio"]) + self.params["ratio"]





