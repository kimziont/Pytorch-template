import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from omegaconf import OmegaConf
from dataloaders.dataset import BasicDataset, MaskDataset
from importlib import import_module


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model = {'state_dict': None, 'optimizer': None}
    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model['state_dict'] = model.state_dict()
        self.best_model['optimizer'] = optimizer.state_dict()
        torch.save(self.best_model, self.path)
        self.val_loss_min = val_loss


# 현재 k-fold는 fold가 iteration될 때마다 model과 loss같은 것들이 초기화가 안됨 -> 수정 필요
def stratified_k_fold(config, X, y):
    stratified_k_fold = StratifiedKFold(n_splits=config['kfold'], shuffle=True, random_state=config["seed"])
    for train_index, valid_index in stratified_k_fold.split(X, y):
        X_stratified_train = X[train_index]
        X_stratified_valid = X[valid_index]
        y_stratified_train = y[train_index]
        y_stratified_valid = y[valid_index]
        dataset_module = getattr(import_module("dataloaders.dataset"), config["dataset"])
        train_dataset = dataset_module(kfold_annotations=X_stratified_train, kfold_label=y_stratified_train)
        valid_dataset = dataset_module(kfold_annotations=X_stratified_valid, kfold_label=y_stratified_valid)
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
        valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
        
        yield train_dataset, valid_dataset, train_dataloader, valid_dataloader