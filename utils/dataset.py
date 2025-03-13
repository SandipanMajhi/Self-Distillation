import dgl
import dgl.data
import torch
import numpy as np
import os

def dataloader(param, mode, default_dir = "./Data", device = "cpu"):
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)

    if param["dataset"] == "cora":
        graph = dgl.data.CoraGraphDataset(raw_dir = default_dir)[0]
    if param["dataset"] == "citeseer":
        graph = dgl.data.CiteseerGraphDataset(raw_sir = default_dir)[0]


    features = graph.ndata["feat"].to(device)
    labels = graph.ndata["labels"].to(device)

    train_mask = graph.ndata["train_mask"].to(device)
    val_mask = graph.ndata["val_mask"].to(device)
    test_mask = graph.ndata["test_mask"].to(device)


    return graph, features, labels, train_mask, val_mask, test_mask


