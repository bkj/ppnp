#!/usr/bin/env python

"""
    main.py
    
    Use IO + preprocessing + random seeds from https://github.com/klicperajo/ppnp
    to guarantee reproducibility
"""

import os
import sys
import math
import json
import random
import argparse
import numpy as np
import pandas as pd
from time import time
import scipy.sparse as sp

import torch
from torch import nn
from torch.nn import functional as F

torch.backends.cudnn.deterministic = True

from ppnp.data.sparsegraph import SparseGraph
from ppnp.preprocessing import gen_splits, normalize_attributes

from model import PPNP
from helpers import set_seeds, SimpleEarlyStopping
from ppr import ExactPPR, DenseNibblePPR, SparseNibblePPR

# --
# Helpers

def load_csr(path):
    row, col, val = np.load(path).T
    row, col = row.astype(np.int), col.astype(np.int)
    return sp.csr_matrix((val, (row, col)))

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--inpath', type=str, default='ppnp/data/cora_ml.npz')
    # parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--seed',   type=int, default=123)
    
    parser.add_argument('--ntrain-per-class', type=int,   default=20)
    parser.add_argument('--nstopping',        type=int,   default=500)
    parser.add_argument('--nknown',           type=int,   default=1500)
    parser.add_argument('--max-epochs',       type=int,   default=10_000)
    parser.add_argument('--reg-lambda',       type=float, default=5e-3)
    parser.add_argument('--lr',               type=float, default=0.01)
    parser.add_argument('--alpha',            type=float, default=0.1)
    parser.add_argument('--test',             action="store_true")
    
    parser.add_argument('--sparse',           action="store_true")
    parser.add_argument('--ppr-topk',         type=int)
    parser.add_argument('--ppr-mode',         type=str, default='exact')
    
    parser.add_argument('--verbose', action="store_true")
    
    return parser.parse_args()

args = parse_args()
args.verbose = True
set_seeds(args.seed)

# --
# Run

graph_inpath = '/home/bjohnson/projects/spectral-experiments/data/DS72784/subj1-scan1.A_ptr.npy'
label_inpath = '/home/bjohnson/projects/spectral-experiments/data/DS72784/subj1-scan1.y.npy'
adj = load_csr(graph_inpath)
y   = np.load(label_inpath)

assert not (adj.diagonal() != 0).any()
assert adj.shape[0] == adj.shape[1]
n_nodes   = adj.shape[0]
n_classes = len(set(y))

# single connected component?

# if args.ppr_mode == 'nibble':
#     # I _think_ this is necessary?
#     adj = sp.eye(adj.shape[0]) + adj
#     adj = (adj > 0).astype(np.float32)

# --
# Define data

X = torch.arange(n_nodes).cuda()
y = torch.LongTensor(y)

np.random.seed(123)
folds = np.random.choice(['train', 'stop', 'valid'], n_nodes, p=[0.1, 0.1, 0.8])

idx_train = np.where(folds == 'train')[0]
idx_stop  = np.where(folds == 'stop')[0]
idx_valid = np.where(folds == 'valid')[0]

y_train, y_stop, y_valid = y[idx_train], y[idx_stop], y[idx_valid]

# to cuda
idx_train, idx_stop, idx_valid = map(lambda x: torch.LongTensor(x).cuda(), (idx_train, idx_stop, idx_valid))
y_train, y_stop, y_valid       = map(lambda x: x.cuda(), (y_train, y_stop, y_valid))

# --
# Precompute PPR

from joblib import Parallel, delayed
from scipy.sparse.linalg import cg
from ppr import calc_A_hat

_cg = None
def exact_ppr_joblib(adj, alpha, mode='sym', n_jobs=60):
    global _cg
    assert mode == 'sym'
    
    A_hat   = calc_A_hat(adj, mode=mode)
    A_inner = sp.eye(adj.shape[0]) - (1 - alpha) * A_hat
    
    def _cg(signals):
        tmp = [cg(A_inner, signal, maxiter=10000, tol=1e-8)[0] for signal in signals]
        return np.row_stack(tmp)
    
    signals = np.eye(adj.shape[0])
    jobs    = [delayed(_cg)(chunk) for chunk in np.array_split(signals, 4 * n_jobs)]
    res     = Parallel(backend='loky', n_jobs=n_jobs, verbose=10)(jobs)
    return np.row_stack(res)

# ppr_array = exact_ppr_joblib(adj, alpha=0.9)
# np.save('ppr_array', ppr_array)
ppr_array = np.load('ppr_array.npy')

# --
# Define model

import torch
from torch import nn
from torch.nn import functional as F

from ppr import _PPR
from model import NormalizedEmbedding

class PrecomputedPPR(_PPR):
    def __init__(self, ppr):
        super().__init__()
        self.register_buffer('ppr', torch.FloatTensor(ppr))
        
        self.sparse = False
        self.batch  = False


class EmbeddingPPNP2(nn.Module):
    def __init__(self, n_nodes, n_classes, ppr, hidden_dim):
        
        super().__init__()
        
        self.encoder    = NormalizedEmbedding(n_nodes, hidden_dim)
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self.ppr        = ppr
    
    def get_norm(self):
        return 0
    
    def forward(self, X, idx):
        hood_enc = self.ppr(X, idx, self.encoder)
        return self.classifier(hood_enc)


model = EmbeddingPPNP2(
    n_nodes=n_nodes,
    n_classes=n_classes,
    ppr=PrecomputedPPR(ppr=ppr_array),
    hidden_dim=16
)

model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

early_stopping = SimpleEarlyStopping(model)

from sklearn import metrics
def acc_score(act, pred):
    return (act == pred).float().mean()

def f1_score(act, pred, average='macro'):
    act  = act.cpu().numpy()
    pred = pred.cpu().numpy()
    return metrics.f1_score(act, pred, average=average)

metric_fn = f1_score

t = time()
for epoch in range(args.max_epochs):
    
    # --
    # Train
    
    _ = model.train()
    
    logits     = model(X, idx_train)
    train_loss = F.cross_entropy(logits, y_train)
    train_loss = train_loss + args.reg_lambda / 2 * model.get_norm()
    
    opt.zero_grad()
    train_loss.backward()
    opt.step()
    
    train_score = metric_fn(y_train, logits.argmax(dim=-1))
    
    # --
    # Stop
    
    _ = model.eval()
    
    with torch.no_grad():
        logits    = model(X, idx_stop)
        stop_loss = F.cross_entropy(logits, y_stop)
        stop_loss = stop_loss + args.reg_lambda / 2 * model.get_norm()
        
        stop_score  = metric_fn(y_stop, logits.argmax(dim=-1))
        valid_score = metric_fn(y_valid, model(X, idx_valid).argmax(dim=-1))
    
    record = {
        "epoch"       : int(epoch),
        "elapsed"     : float(time() - t),
        "train_score" : float(train_score),
        "stop_score"  : float(stop_score),
        "valid_score" : float(valid_score),
    }
    
    if args.verbose:
        print(json.dumps(record), file=sys.stderr)
        sys.stderr.flush()
    
    if early_stopping.should_stop(acc=float(stop_score), loss=float(stop_loss), epoch=epoch, record=record):
        break

record = early_stopping.record
print(record)


with torch.no_grad():
    pred_test = model(X, idx_valid).argmax(dim=-1)

from sklearn import metrics
metrics.f1_score(y_valid.cpu().numpy(), pred_test.cpu().numpy(), average='macro')
metrics.f1_score(y_valid.cpu().numpy(), pred_test.cpu().numpy(), average='micro')

