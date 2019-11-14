#!/usr/bin/env python

"""
    orig.py
    
    Straight from notebook in repo
"""

import os
import random
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

# torch.set_default_tensor_type('torch.DoubleTensor')
torch.backends.cudnn.deterministic = True
_ = random.seed(123 + 1)
_ = np.random.seed(123 + 2)
_ = torch.manual_seed(123 + 3)
_ = torch.cuda.manual_seed(123 + 4)

from ppnp.pytorch.training import train_model
from ppnp.pytorch.propagation import PPRExact, PPRPowerIteration
from ppnp.data.io import load_dataset

from ppnp.preprocessing import gen_seeds, gen_splits, normalize_attributes
from ppnp.pytorch.earlystopping import EarlyStopping, stopping_args

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# --
import math
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
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
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
    def __init__(self, n_features, n_classes, propagation, hidden_dim=64, drop_prob=0.5, bias=False):
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Dropout(drop_prob),
            CustomLinear(n_features, hidden_dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, n_classes, bias=bias)
        )
        
        self._reg_params = list(self.encoder[1].parameters())
        self.propagation = propagation
    
    def get_norm(self):
        return sum((torch.sum(param ** 2) for param in self._reg_params))
    
    def forward(self, X, idx):
        E = self.encoder(X)
        E = self.propagation(E, idx)
        return E

# --
# Helpers

num_runs = 5

all_results = []
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
    
    reg_lambda     = 5e-3
    learning_rate  = 0.01
    
    test = True
    
    # --
    # Orig
    
    # result = train_model(
    #     model_class=PPNP,
    #     graph=graph,
    #     model_args=model_args,
    #     learning_rate=learning_rate,
    #     reg_lambda=reg_lambda,
    #     test=True,
    # )
    # all_results.append(result)
    # for r in all_results:
    #     print(r)
    
    # os._exit(0)
    
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
    
    ppr   = PPRExact(graph.adj_matrix, alpha=0.1)
    model = PPNP(n_features=X.shape[1], n_classes=y.max() + 1, propagation=ppr).cuda()
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    early_stopping = EarlyStopping(model, **stopping_args)
    
    for epoch in range(early_stopping.max_epochs):
        
        # --
        # Train
        
        _ = model.train()
        opt.zero_grad()
        logits = model(X, idx_train)
        preds  = logits.argmax(dim=-1)
        
        train_loss = F.cross_entropy(logits, y_train)
        train_loss = train_loss + reg_lambda / 2 * model.get_norm()
        
        train_loss.backward()
        opt.step()
        
        train_acc = (preds == y_train).float().mean()
        
        # --
        # Stop
        
        _ = model.eval()
        with torch.no_grad():
            logits = model(X, idx_stop)
            preds  = logits.argmax(dim=-1)
            
            stop_loss = F.cross_entropy(logits, y_stop)
            stop_loss = stop_loss + reg_lambda / 2 * model.get_norm()
            
            stop_acc = (preds == y_stop).float().mean()
        
        stop_vars = [float(stop_acc), float(stop_loss)]
        if early_stopping.check(stop_vars, epoch):
            break
    
    _ = model.load_state_dict(early_stopping.best_state)
    _ = model.eval()
    
    train_acc = (model(X, idx_train).argmax(dim=-1) == y_train).float().mean()
    stop_acc  = (model(X, idx_stop).argmax(dim=-1) == y_stop).float().mean()
    valid_acc = (model(X, idx_valid).argmax(dim=-1) == y_valid).float().mean()
    
    print({
        "epochs"     : int(epoch),
        "train_acc"  : float(train_acc),
        "stop_acc"   : float(stop_acc),
        "valid_acc"  : float(valid_acc),
    })
