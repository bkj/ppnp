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

def get_predictions(model, attr_matrix, idx):
    idx = idx.cuda()
    with torch.set_grad_enabled(False):
        logits = model(attr_matrix, idx)
        preds  = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

def train_model(
        model_class: Type[nn.Module],
        graph: SparseGraph,
        model_args: dict,
        learning_rate: float,
        reg_lambda: float,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: dict = stopping_args,
        test: bool = False,
        torch_seed: int = None) -> nn.Module:
    
    STATE = 0
    
    labels_np = graph.labels
    labels    = torch.LongTensor(labels_np)
    
    idx_all = {}
    idx_all['train'], idx_all['stopping'], idx_all['valtest'] =\
        gen_splits(labels_np, idx_split_args, test=test)
    
    idx_all = {key: torch.LongTensor(val) for key, val in idx_all.items()}
    
    torch.manual_seed(seed=gen_seeds())
    
    nfeatures = graph.attr_matrix.shape[1]
    nclasses  = max(labels_np) + 1
    model     = model_class(nfeatures, nclasses, **model_args).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    dataloaders = {phase : (ind.cuda(), labels[ind].cuda()) for phase, ind in idx_all.items()}
    
    early_stopping = EarlyStopping(model, **stopping_args)
    
    attr_mat_norm = normalize_attributes(graph.attr_matrix)
    attr_mat_norm = matrix_to_torch(attr_mat_norm)
    attr_mat_norm = attr_mat_norm.cuda().to_dense()
    
    epoch_stats = {'train': {}, 'stopping': {}}
    
    for epoch in range(early_stopping.max_epochs):
        for phase in epoch_stats.keys():
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            idx, labels = dataloaders[phase]
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                logits = model(attr_mat_norm, idx)
                preds  = torch.argmax(logits, dim=1)
                
                loss   = F.nll_loss(logits, labels)
                l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                loss   = loss + reg_lambda / 2 * l2_reg
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # STATE += float(loss)
                # print(float(STATE))
            
            # Collect statistics
            epoch_stats[phase]['loss'] = float(loss)
            epoch_stats[phase]['acc']  = float((preds == labels).float().mean())
        
        # >>
        # if epoch == 20:
        #     raise Exception()
        # <<
        
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
    
    stopping_acc = (stopping_preds == labels_np[idx_all['stopping']]).mean()
    valtest_acc  = (valtest_preds == labels_np[idx_all['valtest']]).mean()
    
    return {
        "best_epoch"   : int(best_state['best_epoch']),
        "stopping_acc" : float(stopping_acc),
        "valtest_acc"  : float(valtest_acc),
        "test"         : test
    }
