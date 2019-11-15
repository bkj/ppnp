#!/usr/bin/env python

"""
    simple.py
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

from ppnp.data.sparsegraph import SparseGraph
from ppnp.preprocessing import gen_seeds, gen_splits, normalize_attributes

from model import PPNP
from helpers import set_seeds, compute_ppr, SimpleEarlyStopping

# --
# Train

set_seeds(123)

num_runs = 5

for _ in range(num_runs):
    
    inpath = 'ppnp/data/cora_ml.npz'
    with np.load(inpath, allow_pickle=True) as loader:
        graph = SparseGraph.from_flat_dict(dict(loader))
    
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
    
    ppr   = torch.FloatTensor(compute_ppr(graph.adj_matrix, alpha=0.1))
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
