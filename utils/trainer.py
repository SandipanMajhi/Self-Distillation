import torch
from tqdm import tqdm


from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(self, model, loss_fn, optimizer, epochs,  device = "cpu", checkpoint_path = "./Checkpoints/"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_path = f"{checkpoint_path}best_model.pt"
        self.device = device
        self.model = self.model.to(self.device)

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

        adj = adj.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)


        train_idx = torch.where(train_mask == True)[0]
        val_idx = torch.where(val_mask == True)[0]

        ### Train and validation Loop ###

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            output,_ = self.model(features, adj)
            loss = self.loss_fn(output[train_idx].to(self.device), labels[train_idx].to(self.device))

            
            loss.backward()
            self.optimizer.step()

            train_pred = torch.argmax(output[train_idx], dim = -1)
            train_label = labels[train_idx]

            trainacc = accuracy_score(train_label.cpu(), train_pred.cpu())
            self.history["train"]["loss"].append(loss.item())
            self.history["train"]["acc"].append(trainacc)

            self.model.eval()
            with torch.no_grad():
                output,_ = self.model(features, adj)
                loss = self.loss_fn(output[val_idx].to(self.device), labels[val_idx].to(self.device))

                val_pred = torch.argmax(output[val_idx],  dim = 1)
                val_label = labels[val_idx]

                valacc = accuracy_score(val_label.cpu(), val_pred.cpu())
                if len(self.history["val"]["loss"]) > 0:
                    if loss.item() < min(self.history["val"]["loss"]):
                        torch.save(self.model.state_dict(), self.model_path)

                self.history["val"]["loss"].append(loss.item())
                self.history["val"]["acc"].append(valacc)


                print(f"{epoch+1}/{self.epochs} -- trainloss = {self.history["train"]["loss"][-1]} -- trainacc = {trainacc}")
                print(f"valloss -- {self.history["val"]["loss"][-1]} -- valacc = {valacc}")


    def evaluate(self, **kwargs):
        adj = kwargs.get('adj')
        features = kwargs.get('features')
        test_mask = kwargs.get('test_mask')
        labels = kwargs.get('labels')
        test_idx = torch.where(test_mask == True)[0]

        adj = adj.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)

        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model = self.model.to(self.device)
    
        self.model.eval()
        with torch.no_grad():
            output, features = self.model(features, adj)
            loss = self.loss_fn(output[test_idx].to(self.device), labels[test_idx].to(self.device))

            test_pred = torch.argmax(output[test_idx],  dim = 1)
            test_label = labels[test_idx]

            testacc = accuracy_score(test_label.cpu(), test_pred.cpu())

            print(f"Test Set Accuracy of GCN alone = {testacc}")

        return features

