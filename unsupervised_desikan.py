#!/usr/bin/env python

"""
    main.py
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
import scipy.sparse as sp

import torch
from torch import nn
from torch.nn import functional as F

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

torch.backends.cudnn.deterministic = True

from ppr import PrecomputedPPR
from model import EmbeddingPPNP
from helpers import set_seeds


# --
# Helpers (same as spectral-experiments/helpers.py)

def load_csr(path):
    row, col, val = np.load(path).T
    row, col = row.astype(np.int), col.astype(np.int)
    return sp.csr_matrix((val, (row, col)))

def train_stop_valid_split(n, p, random_state=None):
    assert len(p) == 3
    
    if random_state is not None:
        rng = np.random.RandomState(seed=random_state)
    else:
        rng = np.random
    
    folds = rng.choice(['train', 'stop', 'valid'], n_nodes, p=p)
    
    idx_train = np.where(folds == 'train')[0]
    idx_stop  = np.where(folds == 'stop')[0]
    idx_valid = np.where(folds == 'valid')[0]
    
    return idx_train, idx_stop, idx_valid

# --
# More 

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    
    
    parser.add_argument('--seed',   type=int, default=123)
    parser.add_argument('--verbose', action="store_true")
    
    return parser.parse_args()

args = parse_args()
set_seeds(args.seed)

# --
# IO

graph_inpath = '/home/bjohnson/projects/spectral-experiments/data/DS72784/subj1-scan1.A_ptr.npy'
label_inpath = '/home/bjohnson/projects/spectral-experiments/data/DS72784/subj1-scan1.y.npy'
adj = load_csr(graph_inpath)
y   = np.load(label_inpath)

ppr_array = np.load('ppr_array.npy')
np.fill_diagonal(ppr_array, 0)

assert not (adj.diagonal() != 0).any()
assert adj.shape[0] == adj.shape[1]

n_nodes = adj.shape[0]

# --
# Train embeddings

max_epochs        = 1000
samples_per_batch = 2 ** 11

X_dummy = torch.arange(n_nodes).cuda()

model = EmbeddingPPNP(
    n_nodes    = n_nodes,
    ppr        = PrecomputedPPR(ppr=ppr_array),
    hidden_dim = 8,
).cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

t = time()
for epoch in range(max_epochs):
    
    _ = model.train()
    
    # sample a batch -- unsupervised, so can look at all nodes
    idx = np.random.choice(n_nodes, samples_per_batch, replace=False)
    node_enc, hood_enc = model(X=X_dummy, idx=idx)
    
    train_loss = ((node_enc - hood_enc) ** 2).mean()
    
    opt.zero_grad()
    train_loss.backward()
    opt.step()
    
    print(json.dumps({
        "epoch"      : int(epoch),
        "elapsed"    : float(time() - t),
        "train_loss" : float(train_loss),
    }))


X_hat = []
with torch.no_grad():
    for idx in np.array_split(range(n_nodes), n_nodes // 2048):
        _, tmp = model(X=X_dummy, idx=idx)
        tmp = tmp.detach().cpu().numpy()
        X_hat.append(tmp)

X_hat = np.row_stack(X_hat)
nX_hat = normalize(X_hat, axis=1, norm='l2')

# --
# Train/test split
# `train_stop_valid_split` is used to match `ppnp` methods, which
#  may use a `stop` split for early stopping

idx_train, idx_stop, idx_valid = \
    train_stop_valid_split(n_nodes, p=[0.05, 0.05, 0.9], random_state=111)

idx_train = np.concatenate([idx_train, idx_stop])
del idx_stop

nX_train, nX_valid = nX_hat[idx_train], nX_hat[idx_valid]
y_train, y_valid   = y[idx_train], y[idx_valid]

# --
# Train model
# !! TODO -- Should tune model

clf = RandomForestClassifier(n_estimators=512, n_jobs=10)
clf = clf.fit(nX_train, y_train)

prob_valid = clf.predict_proba(nX_valid)
pred_valid = prob_valid.argmax(axis=-1)

print({
    "acc"      : metrics.accuracy_score(y_valid, pred_valid),
    "f1_macro" : metrics.f1_score(y_valid, pred_valid, average='macro'),
    "f1_micro" : metrics.f1_score(y_valid, pred_valid, average='micro'),
})



