import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

    def forward(self):
        pass

