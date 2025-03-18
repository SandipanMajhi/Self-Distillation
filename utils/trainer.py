import torch
from torch.optim import Adam

class Trainer:
    def __init__(self, model, loss_fn, optimizer, epochs,  device = "cpu", checkpoint_path = "./Checkpoints/"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_path = f"{checkpoint_path}best_model.pt"

        self.history = {
            "train" : {
                "loss" : [],
                "acc" : []
            },
            "val" : {
                "loss" : [],
                "acc" : []
            },
            "test" : {
                "loss" : None,
                "acc" : None
            }
        }

    def train(self, **kwargs):
        adj = kwargs.get('adj')
        features = kwargs.get('features')
        train_mask = kwargs.get('train_mask')
        val_mask = kwargs.get('val_mask')
        labels = kwargs.get('labels')

        train_idx = torch.where(train_mask == True)[0]
        val_idx = torch.where(val_mask == True)[0]

        trainloss = 0
        valloss = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(features, adj)
            loss = self.loss_fn(output[train_idx], labels[train_idx])

            loss.backward()
            self.optimizer.step()

            train_pred = torch.argmax(output[train_idx], dim = -1)
            train_label = labels[train_idx]



            #         optimizer.zero_grad()
            # output = model(features, adj)
            # loss = F.nll_loss(output[idx_train], labels[idx_train])
            # acc = accuracy(output[idx_train], labels[idx_train])
            # loss.backward()
            # optimizer.step()
            
            # return loss.item(), acc

    def evaluate(self, **kwargs):
        pass 