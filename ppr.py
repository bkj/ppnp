#!/usr/bin/env python

"""
    ppr.py
"""

import sys
import numpy as np
from numba import jit, prange
from scipy import sparse as sp

import torch
from torch import nn
from torch.nn import functional as F

# --
# Exact PPR

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

# --
# PR-Nibble

@jit(nopython=True)
def _ppr_inner_loop(seed, degrees, adj_indices, adj_indptr, alpha, epsilon):
    num_nodes = degrees.shape[0]
    
    p = np.zeros(num_nodes)
    r = np.zeros(num_nodes)
    r[seed] = 1
    
    frontier = np.array([seed])
    it = 0
    while True:
        if len(frontier) == 0:
            break
        
        r_prime = r.copy()
        
        p[frontier] += (2 * alpha) / (1 + alpha) * r[frontier]
        r_prime[frontier] = 0
        
        for src_idx in frontier:
            neighbors = adj_indices[adj_indptr[src_idx]:adj_indptr[src_idx + 1]]
            update    = ((1 - alpha) / (1 + alpha)) * r[src_idx] / degrees[src_idx]
            r_prime[neighbors] += update
        
        r = r_prime
        
        frontier = np.where((r >= degrees * epsilon) & (degrees > 0))[0]
        
        it += 1
    
    return p

@jit(nopython=True, parallel=True)
def _parallel_pr_nibble(seeds, degrees, adj_indices, adj_indptr, alpha, epsilon):
    out = np.zeros((len(seeds), degrees.shape[0]))
    for i in prange(len(seeds)):
        out[i] = _ppr_inner_loop(
            seeds[i], degrees, adj_indices, adj_indptr, alpha, epsilon)
    
    return out

def parallel_pr_nibble(seeds, adj, alpha, epsilon):
    degrees = adj @ np.ones(adj.shape[0])
    return _parallel_pr_nibble(seeds, degrees, adj.indices, adj.indptr, alpha, epsilon)

# --
# Sparse PR-Nibble
# !! This is a stupid amount of additional code

@jit(nopython=True)
def _sparse_ppr_inner_loop(seed, degrees, adj_indices, adj_indptr, alpha, epsilon, topk):
    num_nodes = degrees.shape[0]
    
    p = np.zeros(num_nodes)
    r = np.zeros(num_nodes)
    r[seed] = 1
    
    frontier = np.array([seed])
    it = 0
    while True:
        if len(frontier) == 0:
            break
        
        r_prime = r.copy()
        
        p[frontier] += (2 * alpha) / (1 + alpha) * r[frontier]
        r_prime[frontier] = 0
        
        for src_idx in frontier:
            neighbors = adj_indices[adj_indptr[src_idx]:adj_indptr[src_idx + 1]]
            update    = ((1 - alpha) / (1 + alpha)) * r[src_idx] / degrees[src_idx]
            r_prime[neighbors] += update
        
        r = r_prime
        
        frontier = np.where((r >= degrees * epsilon) & (degrees > 0))[0]
        
        it += 1
    
    indices = np.argsort(-p)[:topk]
    values  = p[indices]
    
    return indices, values

@jit(nopython=True, parallel=True)
def _sparse_parallel_pr_nibble(seeds, degrees, adj_indices, adj_indptr, alpha, epsilon, topk):
    indices = np.zeros((len(seeds), topk), dtype=np.int32)
    values  = np.zeros((len(seeds), topk), dtype=np.float32)
    
    for i in prange(len(seeds)):
        indices[i], values[i] = _sparse_ppr_inner_loop(
            seeds[i], degrees, adj_indices, adj_indptr, alpha, epsilon, topk)
        
    return indices, values

def sparse_parallel_pr_nibble(seeds, adj, alpha, epsilon, topk):
    degrees = adj @ np.ones(adj.shape[0])
    return _sparse_parallel_pr_nibble(seeds, degrees, adj.indices, adj.indptr, alpha, epsilon, topk)

# --

class _PPR(nn.Module):
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
            indices  = self.indices[idx]
            values   = self.values[idx]
            
            
            # !! What's the mode efficient way to do this?
            # <<
            u_fwd, u_bwd = indices.unique(return_inverse=True)
            return (encoder(X[u_fwd])[u_bwd] * values.unsqueeze(-1)).sum(axis=1)
            # --
            # Alternatives
            # return (encoder(X[indices]) * values.unsqueeze(-1)).sum(axis=1)
            # return (encoder(X)[indices] * values.unsqueeze(-1)).sum(axis=1)
            # <<
        
        raise Exception()

class ExactPPR(_PPR):
    def __init__(self, adj, alpha, mode='sym', topk=None, sparse=False, batch=False):
        super().__init__()
        
        ppr = exact_ppr(adj, alpha, mode=mode)
        
        # Full dense PPR
        if topk is None:
            print('full', file=sys.stderr)
            ppr = torch.FloatTensor(ppr)
            
            self.register_buffer('ppr', ppr)
        
        # Truncated but still dense
        if topk and not sparse:
            print('topk and not sparse', file=sys.stderr)
            ppr    = torch.FloatTensor(ppr)
            topk   = ppr.topk(topk, dim=-1)
            thresh = topk.values[:,-1].view(-1, 1)
            ppr[ppr < thresh] = 0
            
            self.register_buffer('ppr', ppr)
        
        # Truncated and sparse
        if sparse:
            print('sparse', file=sys.stderr)
            assert topk is not None
            ppr  = torch.FloatTensor(ppr)
            topk = ppr.topk(topk, dim=-1)
            
            self.register_buffer('indices', topk.indices)
            self.register_buffer('values', topk.values)
        
        self.sparse = sparse
        self.batch  = batch


class DenseNibblePPR(_PPR):
    def __init__(self, adj, alpha, topk=None, batch=False, epsilon=1e-5):
        super().__init__()
        print('DenseNibblePPR', file=sys.stderr)
        
        seeds = np.arange(adj.shape[0])
        ppr   = parallel_pr_nibble(seeds, adj, alpha, epsilon=epsilon)
        
        # Full dense PPR
        if topk is None:
            print('full', file=sys.stderr)
            ppr = torch.FloatTensor(ppr)
            
            self.register_buffer('ppr', ppr)
        
        # Truncated but still dense
        if topk and not sparse:
            print('topk and not sparse', file=sys.stderr)
            ppr    = torch.FloatTensor(ppr)
            topk   = ppr.topk(topk, dim=-1)
            thresh = topk.values[:,-1].view(-1, 1)
            ppr[ppr < thresh] = 0
            
            self.register_buffer('ppr', ppr)
        
        self.sparse = False
        self.batch  = batch


class SparseNibblePPR(_PPR):
    def __init__(self, adj, alpha, topk, batch=False, epsilon=1e-5):
        super().__init__()
        print('SparseNibblePPR', file=sys.stderr)
        
        assert topk is not None
        
        seeds = np.arange(adj.shape[0])
        indices, values = sparse_parallel_pr_nibble(seeds, adj, alpha, epsilon=epsilon, topk=topk)
        indices, values = torch.LongTensor(indices), torch.FloatTensor(values)
        
        self.register_buffer('indices', indices)
        self.register_buffer('values', values)
        
        self.sparse = True
        self.batch  = batch



# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np

# e = torch.rand(1000, 4)
# x = torch.LongTensor(np.random.choice(1000, (128, 2)))
# model = nn.Linear(4, 4)

# e[x].shape

# u, m = x.unique(return_inverse=True)
# assert (e[u][m] == e[x]).all()
# assert (model(e)[x] == model(e[u])[m]).all()

