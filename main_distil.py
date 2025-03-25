from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json

from utils.dataset import *
from Models.model import CoraModel
from torch.optim import Adam
from Models.distillers import TGSBackbone, EdgeAdapter
from utils.trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    with open("utils/params.json") as fp:
        data = json.load(fp)

    cora_params = data["cora"]
    print(f"Device used = {device}")

    #### Loading the Features Part on which distillation is done ####
    
    graph, edges, adj, features, labels, train_mask, val_mask, test_mask = dataloader(param=cora_params)

    # gnn = CoraModel(in_features=features.shape[1], hidden_features=cora_params["hid_dim"], out_features=7, dropout=cora_params["dropout"])
    # trainer = Trainer(model = gnn, loss_fn = torch.nn.CrossEntropyLoss(), epochs=None, optimizer=None, device = device)

    # features = trainer.evaluate(adj = adj, features = features, labels = labels, test_mask = test_mask)

    #### Graph Attributes ####
    src, dest = graph.edges()
    n_nodes = features.shape[0]
    n_edges = src.shape[0]
    n_class = int(labels.max().item() + 1)

    label_ndist = labels[torch.arange(n_nodes)[train_mask]].float().histc(n_class)
    label_edist = (labels[src[train_mask[src]]].float().histc(n_class) + labels[dest[train_mask[dest]]].float().histc(n_class))
    weight = n_class * torch.nn.functional.normalize(label_ndist/label_edist, p=1, dim = 0)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    val_mask = val_mask.to(device)
    weight = weight.to(device)

    #### Now the distillation phase ###
    tgs_model = TGSBackbone(in_features=features.shape[1], hidden_dim = cora_params["hid_dim"], out_dim = n_class, params=cora_params)
    factor_model = EdgeAdapter(in_features=features.shape[1], hidden_dim=cora_params["hid_dim"], param = cora_params)

    tgs_model = tgs_model.to(device)
    factor_model = factor_model.to(device)

    optimize_model = torch.optim.Adam(list(tgs_model.parameters()), lr=float(cora_params['lr']), weight_decay=float(cora_params['weight_decay']))
    optimize_factor = torch.optim.Adam(list(factor_model.parameters()), lr=1e-4, weight_decay=float(cora_params['weight_decay']))

    early_stop = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in tqdm(range(cora_params["epochs"])):
        negative_sample = torch.randint(0, n_nodes, (n_edges,))
        edge_id_loader = DataLoader(range(n_edges), batch_size = cora_params["batch_size"], shuffle = True)

        epoch_classification_loss = 0
        epoch_neighbourhood_loss = 0
        epoch_total_loss = 0

        for idx in edge_id_loader:
            tgs_model.train()

            neg_idx = negative_sample[idx].to(device)
            src_idx = src[idx].to(device)
            dest_idx = dest[idx].to(device)

            #### Here we get the interpolation coefficients ####
            beta_src = factor_model(features[src_idx].to(device), features[dest_idx].to(device))
            beta_dest = factor_model(features[dest_idx].to(device), features[src_idx].to(device))

            #### Create virtual Nodes ####
            y,z = tgs_model(features[neg_idx].to(device))
            y1, z1 = tgs_model(features[src_idx].to(device))
            y2, z2 = tgs_model(features[dest_idx].to(device))

            y3, z3 = tgs_model(beta_dest * features[dest_idx].to(device) + (1-beta_dest) * features[src_idx].to(device))
            y4, z4 = tgs_model(beta_src * features[src_idx].to(device) + (1-beta_src) * features[dest_idx].to(device))

            #### Neighbourhood distillation ###
            neighbourhood_loss = F.mse_loss(y1, z3) + F.mse_loss(y2, z4) - F.mse_loss(F.softmax(y1, dim = -1), F.softmax(z, dim = -1)) - F.mse_loss(F.softmax(y2, dim = -1), F.softmax(z, dim = -1))
            classification_loss = torch.zeros((1)).to(device)

            m = train_mask[src_idx]
            if m.any().item():
                target = labels[src_idx][m]
                classification_loss += F.cross_entropy(y1[m], target.to(device), weight = weight) + F.cross_entropy(z3[m], target.to(device), weight = weight)

            m = train_mask[dest_idx]
            if m.any().item():
                target = labels[dest_idx][m]
                classification_loss += F.cross_entropy(y2[m], target.to(device), weight = weight) + F.cross_entropy(z4[m], target.to(device), weight = weight)


            loss_total = neighbourhood_loss + cora_params["alpha"] * classification_loss

            optimize_model.zero_grad()
            optimize_factor.zero_grad()
            loss_total.backward()
            optimize_model.step()
            optimize_factor.step()

            epoch_classification_loss += classification_loss.item()
            epoch_neighbourhood_loss += neighbourhood_loss.item()
            epoch_total_loss += loss_total.item()


        tgs_model.eval()
        logits, _ = tgs_model(features)

        train_acc = ((logits[train_mask].max(dim=1).indices == labels[train_mask]).sum() / train_mask.sum().float()).item()
        val_acc = ((logits[val_mask].max(dim=1).indices == labels[val_mask]).sum() / val_mask.sum().float()).item()
        test_acc = ((logits[test_mask].max(dim=1).indices == labels[test_mask]).sum() / test_mask.sum().float()).item()

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 50:
                print("Early Stopping")
                break


        print(f"Epoch = {epoch+1}/{cora_params["epochs"]} --- classification loss = {epoch_classification_loss/len(edge_id_loader)} --- neighbourhood loss = {epoch_neighbourhood_loss/len(edge_id_loader)} --- Total Loss = {epoch_total_loss/len(edge_id_loader)}")
        print(f"Train Acc = {train_acc} --- Val Acc = {val_acc} --- Val Best Acc = {val_best} --- Test Acc = {test_acc} --- Test Best Acc = {test_best} --- Test Val Acc = {test_val}")


        









