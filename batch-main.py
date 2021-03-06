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

from ppnp.data.sparsegraph import SparseGraph
from ppnp.preprocessing import gen_splits, normalize_attributes

from model import PPNP
from helpers import set_seeds, compute_ppr, SimpleEarlyStopping

def gen_seeds():
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(max_uint32 + 1, size=1, dtype=np.uint32)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='ppnp/data/cora_ml.npz')
    parser.add_argument('--n-runs', type=int, default=5)
    
    parser.add_argument('--ntrain-per-class', type=int,   default=20)
    parser.add_argument('--nstopping',        type=int,   default=500)
    parser.add_argument('--nknown',           type=int,   default=1500)
    parser.add_argument('--max-epochs',       type=int,   default=10_000)
    parser.add_argument('--reg-lambda',       type=float, default=5e-3)
    parser.add_argument('--lr',               type=float, default=0.01)
    parser.add_argument('--alpha',            type=float, default=0.1)
    parser.add_argument('--test',             action="store_true")

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--ppr-topk',   type=int, default=128)

    parser.add_argument('--seed',   type=int, default=123)
    parser.add_argument('--verbose', action="store_true")
    
    args = parser.parse_args()
    
    is_ms_academic = 'ms_academic' in args.inpath
    if is_ms_academic:
        args.alpha  = 0.2
        args.nknown = 5000
    
    return args

args = parse_args()
set_seeds(args.seed)

# --
# Run

all_records = []
for _ in range(args.n_runs):
    
    graph = np.load(args.inpath, allow_pickle=True)
    graph = SparseGraph.from_flat_dict(dict(graph))
    graph.standardize(select_lcc=True)
    
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
    
    X = normalize_attributes(graph.attr_matrix)
    X = np.asarray(X.todense())
    X = torch.FloatTensor(X).cuda()
    
    y = torch.LongTensor(graph.labels)
    
    idx_train, idx_stop, idx_valid = gen_splits(graph.labels, idx_split_args, test=args.test)
    idx_train, idx_stop, idx_valid = map(torch.LongTensor, (idx_train, idx_stop, idx_valid))
    
    y_train, y_stop, y_valid = y[idx_train], y[idx_stop], y[idx_valid]
    
    train_loader = DataLoader(TensorDataset(idx_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    idx_train, idx_stop, idx_valid = map(lambda x: x.cuda(), (idx_train, idx_stop, idx_valid))
    y_train, y_stop, y_valid       = map(lambda x: x.cuda(), (y_train, y_stop, y_valid))
    
    torch.manual_seed(seed=gen_seeds())
    
    ppr = torch.FloatTensor(compute_ppr(graph.adj_matrix, alpha=args.alpha))
    
    # >>
    # Sparsify PPR matrix
    thresh, _ = ppr.topk(args.ppr_topk, axis=-1)
    ppr[ppr < thresh[:,-1]] = 0
    # <<
    
    model = PPNP(n_features=X.shape[1], n_classes=y.max() + 1, ppr=ppr).cuda()
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    early_stopping = SimpleEarlyStopping(model)
    
    t = time()
    for epoch in range(args.max_epochs):
        
        # --
        # Train
        
        _ = model.train()
        
        train_loss = 0
        for idx_batch, y_batch in train_loader:
            
            # !! This assums PPR is dense for now, but could be changed
            #    If we expand support to sparse matrices, write a little wrapper that
            #    does the same thing.  OR embed it in the model class
            # !! Depending on dataset size, this loader may look a little different
            ppr_sub = model.ppr[idx_batch]
            sel     = (ppr_sub > 0).any(dim=0)
            ppr_sub = ppr_sub[:,sel]
            
            X_batch, y_batch = X[sel].cuda(), y_batch.cuda()
            
            logits = model(X_batch, idx=None, ppr=ppr_sub)
            loss   = F.cross_entropy(logits, y_batch)
            loss   = loss + args.reg_lambda / 2 * model.get_norm()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss
        
        # --
        # Stop
        
        _ = model.eval()
        
        with torch.no_grad():
            logits    = model(X, idx_stop)
            stop_loss = F.cross_entropy(logits, y_stop)
            stop_loss = stop_loss + args.reg_lambda / 2 * model.get_norm()
            
            preds    = logits.argmax(dim=-1)
            stop_acc = (preds == y_stop).float().mean()
            
            valid_acc = (model(X, idx_valid).argmax(dim=-1) == y_valid).float().mean() # !! Naughty
        
        record = {
            "epoch"     : int(epoch),
            "elapsed"   : float(time() - t),
            "stop_acc"  : float(stop_acc),
            "valid_acc" : float(valid_acc),
        }
        
        if args.verbose:
            print(json.dumps(record), file=sys.stderr)
            sys.stderr.flush()
        
        if early_stopping.should_stop(acc=float(stop_acc), loss=float(stop_loss), epoch=epoch, record=record):
            break
    
    record = early_stopping.record
    
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


