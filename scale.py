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
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.deterministic = True

from model import SparseUnsupervisedPPNP
from helpers import set_seeds, SimpleEarlyStopping
from ppr import exact_ppr, parallel_pr_nibble

#

def gen_seeds():
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(max_uint32 + 1, size=1, dtype=np.uint32)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--inpath', type=str, default='ppnp/data/cora_ml.npz')
    parser.add_argument('--n-runs', type=int, default=5)
    
    parser.add_argument('--ntrain-per-class', type=int,   default=20)
    parser.add_argument('--nstopping',        type=int,   default=500)
    parser.add_argument('--nknown',           type=int,   default=1500)
    parser.add_argument('--max-epochs',       type=int,   default=1000)
    parser.add_argument('--reg-lambda',       type=float, default=5e-3)
    parser.add_argument('--lr',               type=float, default=0.01)
    parser.add_argument('--alpha',            type=float, default=0.1)
    parser.add_argument('--test',             action="store_true")
    
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--ppr-topk',   type=int, default=128)
    
    parser.add_argument('--seed',   type=int, default=123)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--sparse', action="store_true")
    
    args = parser.parse_args()
    
    # is_ms_academic = 'ms_academic' in args.inpath
    # if is_ms_academic:
    #     args.alpha  = 0.2
    #     args.nknown = 5000
    
    return args

args = parse_args()
set_seeds(args.seed)

from scipy import sparse
def load():
    data    = np.load('adj_data.npy')
    indices = np.load('adj_indices.npy')
    indptr  = np.load('adj_indptr.npy')
    
    n_nodes = indices.max() + 1
    return sparse.csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))

adj = load()

# --
# Run

# all_records = []
# for _ in range(args.n_runs):

idx_split_args = {
    'ntrain_per_class' : args.ntrain_per_class,
    'nstopping'        : args.nstopping,
    'nknown'           : args.nknown,
    # >>
    # 'seed'             : 2413340114,
    'seed'             : gen_seeds(),  # Variance is too small if we don't do this
    # <<
}

# --
#  Define data

n_nodes      = adj.shape[0]
idx_train    = torch.arange(n_nodes).long()
train_loader = DataLoader(TensorDataset(idx_train), batch_size=args.batch_size, shuffle=True, num_workers=0)

ppr_values, ppr_indices = compute_sparse_ppr(graph.adj_matrix, alpha=args.alpha)
model = SparseUnsupervisedPPNP(n_nodes=n_nodes, ppr_values=ppr_values, ppr_indices=ppr_indices).cuda()











opt = torch.optim.Adam(model.parameters(), lr=args.lr)

t = time()
for epoch in range(args.max_epochs):
    
    # --
    # Train
    
    _ = model.train()
    
    train_loss = 0
    for idx_batch, y_batch in train_loader:
        a, b = model(idx=idx_batch)
        loss = ((a - b) ** 2).mean()
        loss = loss + args.reg_lambda / 2 * model.get_norm()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss
    
    # --
    # Stop
    
    _ = model.eval()
    
    with torch.no_grad():
        a, b   = model(idx=idx_stop, batched=True, sparse=args.sparse)
        stop_loss = ((a - b) ** 2).mean()
        stop_loss = stop_loss + args.reg_lambda / 2 * model.get_norm()
    
    record = {
        "epoch"      : int(epoch),
        "elapsed"    : float(time() - t),
        "train_loss" : float(train_loss),
        "stop_loss"  : float(stop_loss),
    }
    
    if args.verbose:
        print(json.dumps(record), file=sys.stderr)
        sys.stderr.flush()
    
    if early_stopping.should_stop(acc=float(-stop_loss), loss=float(stop_loss), epoch=epoch, record=record):
        break

record = early_stopping.record

# >>
from sklearn.svm import LinearSVC

enc_train, denc_train = model(idx=idx_train)
enc_valid, denc_valid = model(idx=idx_valid)

enc_train, denc_train = enc_train.detach().cpu().numpy(), denc_train.detach().cpu().numpy()
enc_valid, denc_valid = enc_valid.detach().cpu().numpy(), denc_valid.detach().cpu().numpy()

model = LinearSVC().fit(denc_train, y_train.detach().cpu().numpy())
pred  = model.predict(denc_valid)
record['acc'] = (pred == y_valid.detach().cpu().numpy()).mean()
# <<

print(record)
sys.stdout.flush()
all_records.append(record)

# print('epoch per second', epoch / (time() - t), file=sys.stderr)

# --
# Print summary

df = pd.DataFrame(all_records)

print('-' * 50, file=sys.stderr)
print(df.mean(), file=sys.stderr)
print(df.std(), file=sys.stderr)


