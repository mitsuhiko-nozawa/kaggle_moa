import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


import numpy as np
import time
import sys, os

sys.path.append("../../Models/")
#from tabnet import TabNetRegressor

class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # if cycle
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)


        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds

class Model(nn.Module):
    def __init__(self, num_features, num_targets):
        super(Model, self).__init__()
        self.hidden_size = [1500, 1250, 1000, 750]
        self.dropout_value = [0.5, 0.35, 0.3, 0.25]

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, self.hidden_size[0])
        
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size[0])
        self.dropout2 = nn.Dropout(self.dropout_value[0])
        self.dense2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

        self.batch_norm3 = nn.BatchNorm1d(self.hidden_size[1])
        self.dropout3 = nn.Dropout(self.dropout_value[1])
        self.dense3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])

        self.batch_norm4 = nn.BatchNorm1d(self.hidden_size[2])
        self.dropout4 = nn.Dropout(self.dropout_value[2])
        self.dense4 = nn.Linear(self.hidden_size[2], self.hidden_size[3])

        self.batch_norm5 = nn.BatchNorm1d(self.hidden_size[3])
        self.dropout5 = nn.Dropout(self.dropout_value[3])
        self.dense5 = nn.utils.weight_norm(nn.Linear(self.hidden_size[3], num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x


def run_training(model, trainloader, validloader, epochs, optimizer, scheduler, fine_tune_scheduler, loss_fn, loss_tr, early_stopping_steps, verbose, device, fold, seed, path):
    
    early_step = 0
    best_loss = np.inf
    best_epoch = 0
    
    start = time.time()
    t = time.time() - start
    for epoch in range(epochs):
        if fine_tune_scheduler is not None:
            fine_tune_scheduler.step(epoch, model)
        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, device)
        valid_loss = valid_fn(model, loss_fn, validloader, device)
        if epoch % verbose==0 or epoch==epochs-1:
            t = time.time() - start
            print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}, time: {t}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), path)
            early_step = 0
            best_epoch = epoch
        
        elif early_stopping_steps != 0:
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                t = time.time() - start
                print(f"early stopping in iteration {epoch},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
                return model
    t = time.time() - start       
    print(f"training until max epoch {epochs},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
    return model
            
    
def predict(model, testloader, device):
    model.to(device)
    predictions = inference_fn(model, testloader, device)
    
    return predictions

class FineTuneScheduler:
    def __init__(self, epochs, layer_num):
        self.epochs = epochs
        self.epochs_per_step = 0
        self.frozen_layers = []
        self.layer_num = layer_num

    def copy_without_top(self, model, num_features, num_targets, num_targets_new, device):
        self.frozen_layers = []

        model_new = Model(num_features, num_targets)
        model_new.load_state_dict(model.state_dict())

        # Freeze all weights
        for name, param in model_new.named_parameters():
            layer_index = int(name.split('.')[0][-1])

            if layer_index == self.layer_num:  #最後の層以外は凍結する
                continue

            param.requires_grad = False
            # Save frozen layer names
            if layer_index not in self.frozen_layers:
                self.frozen_layers.append(layer_index)

        self.epochs_per_step = 4
        
        # Replace the top layers with another ones, 最後に追加されてく
        model_new.batch_norm5 = nn.BatchNorm1d(model_new.hidden_size[3])
        model_new.dropout5 = nn.Dropout(model_new.dropout_value[3])
        model_new.dense5 = nn.utils.weight_norm(nn.Linear(model_new.hidden_size[-1], num_targets_new))
        model_new.to(device)
        return model_new

    def step(self, epoch, model):
        if len(self.frozen_layers) == 0 or epoch == 0:
            return

        if epoch % self.epochs_per_step == 0:
            last_frozen_index = self.frozen_layers[-1]
            
            # Unfreeze parameters of the last frozen layer
            for name, param in model.named_parameters():
                layer_index = int(name.split('.')[0][-1])

                if layer_index == last_frozen_index:
                    print(epoch, "   ",name)
                    param.requires_grad = True

            del self.frozen_layers[-1]  # Remove the last layer as unfrozen

from pytorch_tabnet.metrics import Metric
class LogitsLogLoss(Metric):
    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, n_cls=2):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing + smoothing / n_cls
        self.smoothing = smoothing / n_cls

    def forward(self, x, target):
        probs = torch.nn.functional.sigmoid(x,)
        target1 = self.confidence * target + (1-target) * self.smoothing
        loss = -(torch.log(probs+1e-15) * target1 + (1-target1) * torch.log(1-probs+1e-15))

        return loss.mean()

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


def run_model1(train_X, train_y, train_y_all, valid_X, valid_y, valid_y_all, test_X, param, seed, fold, mode):
    BATCH_SIZE = param["BATCH_SIZE"]
    DEVICE = param["DEVICE"]
    LEARNING_RATE = param["LEARNING_RATE"]
    WEIGHT_DECAY = param["WEIGHT_DECAY"]
    TR_WEIGHT_DECAY = param["TR_WEIGHT_DECAY"]
    EPOCHS = param["EPOCHS"]
    EARLY_STOPPING_STEPS = param["EARLY_STOPPING_STEPS"]
    DIR = param["WEIGHT_DIR"]

    MAX_LR = param["MAX_LR"]
    TR_MAX_LR = param["TR_MAX_LR"]

    DIV_FACTOR = param["DIV_FACTOR"]
    TR_DIV_FACTOR = param["TR_DIV_FACTOR"]
    PCT_START = param["PCT_START"]


    train_dataset = MoADataset(train_X, train_y_all)
    valid_dataset = MoADataset(valid_X, valid_y_all)
    test_dataset = TestDataset(test_X)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    train_dataset2 = MoADataset(train_X, train_y)
    valid_dataset2 = MoADataset(valid_X, valid_y)
    trainloader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    validloader2 = torch.utils.data.DataLoader(valid_dataset2, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model = Model(
        num_features=train_X.shape[1],
        num_targets=train_y_all.shape[1],
    )

    model.to(DEVICE)
    
    
    optimizer = torch.optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=PCT_START, div_factor=DIV_FACTOR, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(trainloader) )
    fine_tune_scheduler = FineTuneScheduler(EPOCHS, 5)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing=1e-3)
    
    path = os.path.join(DIR, '{}_{}.pt'.format(seed, fold))
    # train
    if mode == "train":
        model = run_training(
            model=model,
            trainloader=trainloader,
            validloader=validloader,
            epochs=EPOCHS,
            optimizer=optimizer,
            scheduler=scheduler,
            fine_tune_scheduler=None,
            loss_fn=loss_fn,
            loss_tr=loss_tr,
            early_stopping_steps=EARLY_STOPPING_STEPS,
            device=DEVICE,
            verbose=5,
            fold=fold,
            seed=seed,
            path=path)
    
        model.load_state_dict(torch.load(path, DEVICE))
        model = fine_tune_scheduler.copy_without_top(model, train_X.shape[1], train_y_all.shape[1], train_y.shape[1], DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=TR_WEIGHT_DECAY, eps=1e-6)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,pct_start=PCT_START, div_factor=TR_DIV_FACTOR, max_lr=TR_MAX_LR, epochs=EPOCHS, steps_per_epoch=len(trainloader2) )
        model = run_training(
            model=model,
            trainloader=trainloader2,
            validloader=validloader2,
            epochs=EPOCHS,
            optimizer=optimizer,
            scheduler=scheduler,
            fine_tune_scheduler=fine_tune_scheduler,
            loss_fn=loss_fn,
            loss_tr=loss_tr,
            early_stopping_steps=EARLY_STOPPING_STEPS,
            device=DEVICE,
            verbose=5,
            fold=fold,
            seed=seed,
            path=path,)

    if mode == "infer":
        model = Model(
            num_features=train_X.shape[1],
            num_targets=train_y.shape[1],
        )

    model.load_state_dict(torch.load(path, DEVICE))
    #valid predict
    val_preds = predict(
        model=model,
        testloader=validloader2,
        device=DEVICE,)
    
    #test predict
    test_preds = predict(
        model=model,
        testloader=testloader,
        device=DEVICE)
    return val_preds, test_preds

def run_model2(train_X, train_y, train_y_all, valid_X, valid_y, valid_y_all, test_X, param, seed, fold, mode):

    DEVICE = param["DEVICE"]
    EARLY_STOPPING_STEPS = param["EARLY_STOPPING_STEPS"]
    BATCH_SIZE = param["BATCH_SIZE"]
    V_BATCH_SIZE = param["V_BATCH_SIZE"]
    EPOCHS = param["EPOCHS"]
    DIR = param["WEIGHT_DIR"]

    param["tabnet"]["seed"] = seed
    model = TabNetRegressor(**param["tabnet"])
    model.device = DEVICE
    name = os.path.join(param["WEIGHT_DIR"], '{}_{}.pt'.format(seed, fold))

    if mode == "train":
        model.fit(
            X_train=train_X,
            y_train=train_y_all,
            eval_set=[(valid_X, valid_y_all)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs=EPOCHS,
            patience=EARLY_STOPPING_STEPS, 
            batch_size=BATCH_SIZE, 
            virtual_batch_size=V_BATCH_SIZE,
            num_workers=1, 
            drop_last=False,
            loss_fn=LabelSmoothing(1e-5)
        )
        model.network.tabnet.final_mapping = nn.Linear(param["tabnet"]["n_d"], 206, bias=False)

        model.optimizer_fn = param["tabnet"]["optimizer_fn"]
        model.optimizer_params = param["tabnet"]["optimizer_params"]
        model.scheduler_fn = param["tabnet"]["scheduler_fn"]
        model.scheduler_params = param["tabnet"]["scheduler_params"]

        model.fit(
            X_train=train_X,
            y_train=train_y,
            eval_set=[(valid_X, valid_y)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs=EPOCHS,
            patience=EARLY_STOPPING_STEPS, 
            batch_size=BATCH_SIZE, 
            virtual_batch_size=V_BATCH_SIZE,
            num_workers=1, 
            drop_last=False,
            loss_fn=LabelSmoothing(1e-5)
        )
        model.save_model(name)
        name += ".zip"

    model.load_model(name)
    val_preds = 1 / (1 + np.exp(-model.predict(valid_X)))
    test_preds = 1 / (1 + np.exp(-model.predict(test_X)))
    return val_preds, test_preds

def run_model3():
    pass