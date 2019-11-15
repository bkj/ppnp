#!/usr/bin/env python

"""
    helpers.py
"""

import torch
import random
import numpy as np
import scipy.sparse as sp

def set_seeds(seed):
    _ = random.seed(seed + 1)
    _ = np.random.seed(seed + 2)
    _ = torch.manual_seed(seed + 3)
    _ = torch.cuda.manual_seed(seed + 4)

class SimpleEarlyStopping:
    def __init__(self, model, patience=100):
        
        self.model        = model
        self.patience     = patience
        self.max_patience = patience
        
        self.best_acc   = -np.inf
        self.best_nloss = -np.inf
        
        self.best_epoch       = -1
        self.best_epoch_score = (-np.inf, -np.inf)
    
    def should_stop(self, acc, loss, epoch):
        nloss = -1 * loss
        
        if (acc < self.best_acc) and (nloss < self.best_nloss):
            self.patience -= 1
            return self.patience == 0
        
        self.patience = self.max_patience
        
        self.best_acc   = max(acc, self.best_acc)
        self.best_nloss = max(nloss, self.best_nloss)
        
        if (acc, nloss) > self.best_epoch_score:
            self.best_epoch       = epoch
            self.best_epoch_score = (acc, nloss)
            self.best_state       = {k:v.cpu() for k,v in self.model.state_dict().items()}
        
        return False


def calc_A_hat(adj, mode):
    A = adj + sp.eye(adj.shape[0])
    D = np.sum(A, axis=1).A1
    if mode == 'sym':
        D_inv = sp.diags(1 / np.sqrt(D))
        return D_inv @ A @ D_inv
    elif mode == 'rw':
        D_inv = sp.diags(1 / D)
        return D_inv @ A

def compute_ppr(adj, alpha, mode='sym'):
    A_hat   = calc_A_hat(adj, mode=mode)
    A_inner = sp.eye(adj.shape[0]) - (1 - alpha) * A_hat
    return alpha * np.linalg.inv(A_inner.toarray())