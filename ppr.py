#!/usr/bin/env python

"""
    ppr.py
"""

import numpy as np
from scipy import sparse as sp

import torch
from torch import nn
from torch.nn import functional as F

def calc_A_hat(adj, mode):
    A = adj + sp.eye(adj.shape[0])
    D = np.sum(A, axis=1).A1
    if mode == 'sym':
        D_inv = sp.diags(1 / np.sqrt(D))
        return D_inv @ A @ D_inv
    elif mode == 'rw':
        D_inv = sp.diags(1 / D)
        return D_inv @ A


def exact_ppr(adj, alpha, mode='sym'):
    A_hat   = calc_A_hat(adj, mode=mode)
    A_inner = sp.eye(adj.shape[0]) - (1 - alpha) * A_hat
    return alpha * np.linalg.inv(A_inner.toarray())


class ExactPPR(nn.Module):
    def __init__(self, adj, alpha, mode='sym', topk=None, sparse=False, batch=False):
        super().__init__()
        
        ppr = exact_ppr(adj, alpha, mode=mode)
        
        # Full dense PPR
        if topk is None:
            ppr = torch.FloatTensor(ppr)
            
            self.register_buffer('ppr', ppr)
        
        # Truncated but still dense
        if topk and not sparse:
            ppr    = torch.FloatTensor(ppr)
            topk   = ppr.topk(topk, dim=-1)
            thresh = topk.values[:,-1].view(-1, 1)
            ppr[ppr < thresh] = 0
            
            self.register_buffer('ppr', ppr)
        
        # Truncated and sparse
        if topk and sparse:
            ppr  = torch.FloatTensor(ppr)
            topk = ppr.topk(topk, dim=-1)
            
            self.register_buffer('indices', topk.indices)
            self.register_buffer('values', topk.values)
        
        self.sparse = sparse
        self.batch  = batch
        
    def forward(self, X, idx, encoder):
        
        # Straightforward
        if not self.sparse and not self.batch:
            return self.ppr[idx] @ encoder(X.cuda())
        
        # Don't pass unecessary points through encoder
        if not self.sparse and self.batch:
            tmp = self.ppr[idx]
            sel = (tmp > 0).any(dim=0)
            tmp = tmp[:,sel]
            
            return tmp @ encoder(X[sel].cuda())
        
        if self.sparse:
            indices = self.indices[idx]
            values  = self.values[idx]
            return (encoder(indices) * values.unsqueeze(-1)).sum(axis=1)
        
        raise Exception()
