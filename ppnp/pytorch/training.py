from typing import Type
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from ..data.sparsegraph import SparseGraph
from ..preprocessing import gen_seeds, gen_splits, normalize_attributes
from .earlystopping import EarlyStopping, stopping_args
from .utils import matrix_to_torch


def get_dataloaders(idx, labels_np, batch_size=None):
    labels = torch.LongTensor(labels_np)
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    
    datasets = {phase: TensorDataset(ind, labels[ind]) for phase, ind in idx.items()}
    
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                   for phase, dataset in datasets.items()}
    
    # dataloaders = {k:[(x.cuda(), y.cuda()) for x, y in v] for k,v in dataloaders.items()}
    
    return dataloaders


def train_model(
        name: str, 
        model_class: Type[nn.Module], 
        graph: SparseGraph, 
        model_args: dict,
        learning_rate: float, 
        reg_lambda: float,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: dict = stopping_args,
        test: bool = False, 
        device: str = 'cuda',
        torch_seed: int = None, 
        print_interval: int = 10
    ) -> nn.Module:
    
    labels_all = graph.labels
    
    idx_np = {}
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits(labels_all, idx_split_args, test=test)
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}
    
    if torch_seed is None:
        torch_seed = gen_seeds()
    
    torch.manual_seed(seed=torch_seed)
    
    nfeatures = graph.attr_matrix.shape[1]
    nclasses  = max(labels_all) + 1
    model     = model_class(nfeatures, nclasses, **model_args).to(device)
    
    reg_lambda = torch.tensor(reg_lambda, device=device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloaders      = get_dataloaders(idx_all, labels_all)
    early_stopping   = EarlyStopping(model, **stopping_args)
    attr_mat_norm_np = normalize_attributes(graph.attr_matrix)
    attr_mat_norm    = matrix_to_torch(attr_mat_norm_np).to(device).to_dense()
    
    epoch_stats = {'train': {}, 'stopping': {}}
    
    for epoch in range(early_stopping.max_epochs):
        for phase in epoch_stats.keys():
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            n_total          = 0
            running_loss     = 0
            running_corrects = 0
            
            for idx, labels in dataloaders[phase]:
                idx    = idx.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    log_preds = model(attr_mat_norm, idx)
                    preds     = torch.argmax(log_preds, dim=1)
                    
                    cross_entropy_mean = F.nll_loss(log_preds, labels)
                    l2_reg             = sum((torch.sum(param ** 2) for param in model.reg_params))
                    loss               = cross_entropy_mean + reg_lambda / 2 * l2_reg
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    n_total          += idx.shape[0]
                    running_loss     += float(loss) * idx.shape[0]
                    running_corrects += int(torch.sum(preds == labels))
            
            epoch_stats[phase]['loss'] = float(running_loss) / n_total
            epoch_stats[phase]['acc']  = float(running_corrects) / n_total
            
        if epoch % print_interval == 0:
            print({
                "epoch" : epoch,
                "train_loss" : epoch_stats['train']['loss'],
                "train_acc"  : epoch_stats['train']['acc'],
                "stop_loss"  : epoch_stats['stopping']['loss'],
                "stop_acc"   : epoch_stats['stopping']['acc'],
            })
        
        if len(early_stopping.stop_vars) > 0:
            stop_vars = [epoch_stats['stopping'][key] for key in early_stopping.stop_vars]
            if early_stopping.check(stop_vars, epoch):
                break
    
    best_state = early_stopping.best_state
    if 'propagation.A_hat' in best_state:
        del best_state['propagation.A_hat']
    
    model.load_state_dict(best_state, strict=False)
    
    stopping_preds = get_predictions(model, attr_mat_norm, idx_all['stopping'])
    valtest_preds  = get_predictions(model, attr_mat_norm, idx_all['valtest'])
    
    stopping_acc = (stopping_preds == labels_all[idx_all['stopping']]).mean()
    valtest_acc  = (valtest_preds == labels_all[idx_all['valtest']]).mean()
    
    return model, {"stopping_acc" : stopping_acc, "valtest_acc" : valtest_acc, "test" : test}


def get_predictions(model, attr_matrix, idx, batch_size=None):
    if batch_size is None:
        batch_size = idx.numel()
    
    dataset    = TensorDataset(idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    preds = []
    for idx, in dataloader:
        idx = idx.to(attr_matrix.device)
        with torch.set_grad_enabled(False):
            log_preds = model(attr_matrix, idx)
            preds.append(torch.argmax(log_preds, dim=1))
    
    return torch.cat(preds, dim=0).cpu().numpy()
