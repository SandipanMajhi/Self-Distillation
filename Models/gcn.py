import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_layer = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.fc_layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        """
            x : shape = (B, Nodes, num_features)
            adj : shape = (B, Nodes, Nodes)
        """
        num_batches = x.shape[0]
        
        self_loops = torch.eye(adj.shape[1]).cuda()
        adj = adj + self_loops
        
        neighbours = torch.sum(adj, dim = -1)
        neighbour_inv = torch.pow(neighbours, -0.5)
        neighbour_inv = torch.diag(neighbour_inv)
        adj = torch.mm(torch.mm(neighbour_inv, adj), neighbour_inv)
        # adj = torch.bmm(torch.bmm(neighbour_inv, adj), neighbour_inv)
        
        #### Feature Projection ####
        x_ = torch.mm(adj, x)
        x_ = self.fc_layer(x_)
        x_ = torch.nn.functional.normalize(x_, dim = 1)
        
        return x_

