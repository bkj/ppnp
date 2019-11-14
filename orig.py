#!/usr/bin/env python

"""
    orig.py
    
    Straight from notebook in repo
"""

import os
import math
import random
import numpy as np
import scipy.sparse as sp

import torch
from torch import nn
from torch.nn import functional as F

torch.backends.cudnn.deterministic = True
_ = random.seed(123 + 1)
_ = np.random.seed(123 + 2)
_ = torch.manual_seed(123 + 3)
_ = torch.cuda.manual_seed(123 + 4)

from ppnp.data.io import load_dataset
from ppnp.preprocessing import gen_seeds, gen_splits, normalize_attributes

# --
# Helpers

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

def calc_ppr_exact(adj, alpha, mode='sym'):
    A_hat   = calc_A_hat(adj, mode=mode)
    A_inner = sp.eye(adj.shape[0]) - (1 - alpha) * A_hat
    return alpha * np.linalg.inv(A_inner.toarray())


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        if self.bias is None:
            return input @ self.weight
        else:
            return torch.addmm(self.bias, input, self.weight)


class PPNP(nn.Module):
    def __init__(self, n_features, n_classes, ppr, hidden_dim=64, drop_prob=0.5, bias=False):
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Dropout(drop_prob),
            CustomLinear(n_features, hidden_dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, n_classes, bias=bias)
        )
        
        self.register_buffer('ppr', ppr)
        
        self._reg_params = list(self.encoder[1].parameters())
    
    def get_norm(self):
        return sum((torch.sum(param ** 2) for param in self._reg_params))
    
    def forward(self, X, idx):
        return self.ppr[idx] @ self.encoder(X)

# --
# Helpers

num_runs = 5

for _ in range(num_runs):
    
    graph_name = 'cora_ml'
    graph      = load_dataset(graph_name)
    graph.standardize(select_lcc=True)
    
    idx_split_args = {
        'ntrain_per_class' : 20,
        'nstopping'        : 500,
        'nknown'           : 1500,
        'seed'             : 2413340114,
    }
    
    max_epochs     = 10_000
    reg_lambda     = 5e-3
    learning_rate  = 0.01
    
    test = True
    
    # --
    #  Define data
    
    X = normalize_attributes(graph.attr_matrix)
    X = np.asarray(X.todense())
    X = torch.FloatTensor(X).cuda()
    
    y = torch.LongTensor(graph.labels)
    
    idx_train, idx_stop, idx_valid = gen_splits(graph.labels, idx_split_args, test=test)
    idx_train, idx_stop, idx_valid = map(torch.LongTensor, (idx_train, idx_stop, idx_valid))
    
    y_train, y_stop, y_valid = y[idx_train], y[idx_stop], y[idx_valid]
    
    idx_train, idx_stop, idx_valid = map(lambda x: x.cuda(), (idx_train, idx_stop, idx_valid))
    y_train, y_stop, y_valid       = map(lambda x: x.cuda(), (y_train, y_stop, y_valid))
    
    torch.manual_seed(seed=gen_seeds())
    
    ppr   = torch.FloatTensor(calc_ppr_exact(graph.adj_matrix, alpha=0.1))
    model = PPNP(n_features=X.shape[1], n_classes=y.max() + 1, ppr=ppr).cuda()
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    early_stopping = SimpleEarlyStopping(model)
    
    for epoch in range(max_epochs):
        
        # --
        # Train
        
        _ = model.train()
        
        logits     = model(X, idx_train)
        train_loss = F.cross_entropy(logits, y_train)
        train_loss = train_loss + reg_lambda / 2 * model.get_norm()
        
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        
        preds     = logits.argmax(dim=-1)
        train_acc = (preds == y_train).float().mean()
        
        # --
        # Stop
        
        _ = model.eval()
        
        with torch.no_grad():
            logits    = model(X, idx_stop)
            stop_loss = F.cross_entropy(logits, y_stop)
            stop_loss = stop_loss + reg_lambda / 2 * model.get_norm()
            
            preds    = logits.argmax(dim=-1)
            stop_acc = (preds == y_stop).float().mean()
        
        if early_stopping.should_stop(acc=float(stop_acc), loss=float(stop_loss), epoch=epoch):
            break
    
    _ = model.load_state_dict(early_stopping.best_state)
    _ = model.eval()
    
    train_acc = (model(X, idx_train).argmax(dim=-1) == y_train).float().mean()
    stop_acc  = (model(X, idx_stop).argmax(dim=-1) == y_stop).float().mean()
    valid_acc = (model(X, idx_valid).argmax(dim=-1) == y_valid).float().mean()
    
    print({
        "epochs"     : int(epoch),
        "best_epoch" : int(early_stopping.best_epoch),
        "train_acc"  : float(train_acc),
        "stop_acc"   : float(stop_acc),
        "valid_acc"  : float(valid_acc),
    })
