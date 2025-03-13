import dgl
import dgl.data
import torch
import numpy as np
import os

def dataloader(param, default_dir = "./Data", device = "cpu"):
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)

    if param["dataset"] == "cora":
        graph = dgl.data.CoraGraphDataset(raw_dir = default_dir)[0]
    if param["dataset"] == "citeseer":
        graph = dgl.data.CiteseerGraphDataset(raw_sir = default_dir)[0]


    features = graph.ndata["feat"].to(device)
    labels = graph.ndata["label"].to(device)

    train_mask = graph.ndata["train_mask"].to(device)
    val_mask = graph.ndata["val_mask"].to(device)
    test_mask = graph.ndata["test_mask"].to(device)

    train_idx = torch.where(train_mask == True)[0]
    val_idx = torch.where(val_mask == True)[0]
    test_idx = torch.where(test_mask == True)[0]

    train_graph = dgl.node_subgraph(graph, train_idx)
    val_graph = dgl.node_subgraph(graph, val_idx)
    test_graph = dgl.node_subgraph(graph, test_idx)

    train_adj = train_graph.adjacency_matrix()
    val_adj = val_graph.adjacency_matrix()
    test_adj = test_graph.adjacency_matrix()

    src, dest = graph.edges()

    # print(train_mask.shape)
    # print(val_mask.shape)
    # print(test_mask.shape)
    # print(labels.shape)
    # print(features.shape)

    # print(src)
    # print(dest)


    # print(test_adj.shape)


    return graph, features, labels, train_mask, val_mask, test_mask


