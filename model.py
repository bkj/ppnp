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

# Supervised
class PPNP(nn.Module):
    def __init__(self, n_features, n_classes, ppr, hidden_dim=64):
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Dropout(0.5),
            CustomLinear(n_features, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes, bias=False)
        )
        
        self.ppr = ppr
        
        self._reg_params = list(self.encoder[1].parameters())
    
    def get_norm(self):
        return sum((torch.sum(param ** 2) for param in self._reg_params))
    
    def forward(self, X, idx):
        return self.ppr(X, idx, self.encoder)


# Node Embedding Only
class NormalizedEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._emb = nn.Embedding(*args, **kwargs)
    
    def forward(self, x):
        w = F.normalize(self._emb.weight, dim=-1)
        return w[x]


class EmbeddingPPNP(nn.Module):
    def __init__(self, n_nodes, ppr, hidden_dim=128):
        
        super().__init__()
        
        self.encoder = NormalizedEmbedding(n_nodes, hidden_dim)
        self.ppr     = ppr
    
    def get_norm(self):
        return 0
    
    def forward(self, X, idx):
        node_enc = self.encoder(X[idx])
        hood_enc = self.ppr(X, idx, self.encoder)
        return node_enc, hood_enc

