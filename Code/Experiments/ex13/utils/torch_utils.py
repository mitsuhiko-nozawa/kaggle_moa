import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import time
import os

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
        self.hidden1 = 2048
        self.hidden2 =  1024
        
        self.nn_model = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.utils.weight_norm(nn.Linear(num_features, self.hidden1)),
            nn.LeakyReLU(),
            
            nn.BatchNorm1d(self.hidden1),
            nn.Dropout(0.25),
            nn.utils.weight_norm(nn.Linear(self.hidden1, self.hidden2)),
            nn.LeakyReLU(),
            
            nn.BatchNorm1d(self.hidden2),
            nn.Dropout(0.25),
            nn.utils.weight_norm(nn.Linear(self.hidden2, num_targets)),
        )

    
    def forward(self, x):
        return self.nn_model(x)

def run_training(model, trainloader, validloader, epoch_, optimizer, scheduler, loss_fn, loss_tr, early_stopping_steps, verbose, device, fold, seed, path):
    
    early_step = 0
    best_loss = np.inf
    best_epoch = 0
    
    start = time.time()
    t = time.time() - start
    for epoch in range(epoch_):
        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, device)
        valid_loss = valid_fn(model, loss_fn, validloader, device)
        if epoch % verbose==0 or epoch==epoch_-1:
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
    print(f"training until max epoch {epoch_},  : best itaration is {best_epoch}, valid loss is {best_loss}, time: {t}")
    return model
            
    
def predict(model, testloader, device):
    model.to(device)
    predictions = inference_fn(model, testloader, device)
    
    return predictions

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


def run_dnn1(train_X, train_y, valid_X, valid_y, test_X, param, seed, fold, mode):
    BATCH_SIZE = param["BATCH_SIZE"]
    DEVICE = param["DEVICE"]
    LEARNING_RATE = param["LEARNING_RATE"]
    WEIGHT_DECAY = param["WEIGHT_DECAY"]
    EPOCHS = param["EPOCHS"]
    EARLY_STOPPING_STEPS = param["EARLY_STOPPING_STEPS"]
    DIR = param["WEIGHT_DIR"]


    train_dataset = MoADataset(train_X, train_y)
    valid_dataset = MoADataset(valid_X, valid_y)
    test_dataset = TestDataset(test_X)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = Model(
        num_features=train_X.shape[1],
        num_targets=train_y.shape[1],
    )

    model.to(DEVICE)
    
    
    optimizer = torch.optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader) )
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing=1e-3)
    
    path = os.path.join(DIR, '{}_{}.pt'.format(seed, fold))
    # train
    if model == "train"
        model = run_training(
            model=model,
            trainloader=trainloader,
            validloader=validloader,
            epoch_=EPOCHS,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            loss_tr=loss_tr,
            early_stopping_steps=EARLY_STOPPING_STEPS,
            device=DEVICE,
            verbose=5,
            fold=fold,
            seed=seed,
            path=path)
    
    model.load_state_dict(torch.load(path, DEVICE))
    
    #valid predict
    val_preds = predict(
        model=model,
        testloader=validloader,
        device=DEVICE,)
    
    #test predict
    test_preds = predict(
        model=model,
        testloader=testloader,
        device=DEVICE)
    return val_preds, test_preds