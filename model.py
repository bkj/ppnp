#!/usr/bin/env python

"""
    model.py
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

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
    
    def forward(self, X, idx=None, ppr=None):
        if idx is not None:
            return self.ppr[idx] @ self.encoder(X)
        elif ppr is not None:
            return ppr @ self.encoder(X)
        else:
            raise Exception()


class UnsupervisedPPNP(nn.Module):
    def __init__(self, ppr):
        super().__init__()
        
        self.emb = nn.Embedding(ppr.shape[0], 128)
        self.register_buffer('ppr', ppr)
    
    def get_norm(self):
        return 0
    
    def forward(self, X, idx):
        enc = self.emb.weight
        enc = F.normalize(enc, dim=-1)       # Prevent from going to zero
        return enc[idx], self.ppr[idx] @ enc
