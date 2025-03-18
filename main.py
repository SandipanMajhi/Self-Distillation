import torch
import json
import argparse
from utils.dataset import *
from utils.trainer import Trainer

from Models.model import CoraModel
from torch.optim import Adam

if __name__ == "__main__":
    with open("utils/params.json") as fp:
        data = json.load(fp)

    cora_params = data["cora"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    graph, edges, adj, features, labels, train_mask, val_mask, test_mask = dataloader(param=cora_params)
    model = CoraModel(in_features=features.shape[1], hidden_features=cora_params["hid_dim"], out_features=7, dropout=cora_params["dropout"])
    optimizer = Adam(model.parameters(), lr = cora_params["lr"], weight_decay=cora_params["weight_decay"])

    trainer = Trainer(model = model, loss_fn = torch.nn.CrossEntropyLoss(), epochs=200, optimizer=optimizer, device = device)
    trainer.train(adj = adj, features = features, train_mask = train_mask, val_mask = val_mask, labels = labels)
    trainer.evaluate(adj = adj, features = features, labels = labels, test_mask = test_mask)



