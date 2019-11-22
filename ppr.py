
import numpy as np
import scipy.sparse as sp
from numba import jit, prange

# --
# Exact (small graphs)

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
# Approximate (large graphs)

# @jit(nopython=True)
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

# @jit(nopython=True, parallel=True)
def _parallel_pr_nibble(seeds, degrees, adj_indices, adj_indptr, alpha, epsilon):
    out = np.zeros((len(seeds), degrees.shape[0]))
    # for i in prange(len(seeds)):
    for i in range(len(seeds)):
        out[i] = _ppr_inner_loop(seeds[i], degrees, adj_indices, adj_indptr, alpha, epsilon)
        if not (i + 1) % 5000:
            print('.')
    
    return out

def parallel_pr_nibble(seeds, adj, alpha=0.1, epsilon=1e-5):
    degrees = adj @ np.ones(adj.shape[0])
    return _parallel_pr_nibble(seeds, degrees, adj.indices, adj.indptr, alpha, epsilon)
