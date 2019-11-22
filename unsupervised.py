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

from model import EmbeddingPPNP
from helpers import set_seeds, SimpleEarlyStopping
from ppr import ExactPPR

def gen_seeds():
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(max_uint32 + 1, size=1, dtype=np.uint32)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='ppnp/data/cora_ml.npz')
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--seed',   type=int, default=123)
    
    parser.add_argument('--ntrain-per-class', type=int,   default=20)
    parser.add_argument('--nstopping',        type=int,   default=500)
    parser.add_argument('--nknown',           type=int,   default=1500)
    parser.add_argument('--max-epochs',       type=int,   default=1000)
    parser.add_argument('--reg-lambda',       type=float, default=5e-3)
    parser.add_argument('--lr',               type=float, default=0.01)
    parser.add_argument('--alpha',            type=float, default=0.1)
    parser.add_argument('--test',             action="store_true")
    
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
        'ntrain_per_class' : args.ntrain_per_class, # What is the score on the official split?
        'nstopping'        : args.nstopping,
        'nknown'           : args.nknown,            # What does this mean when test is true?
        # >>
        # 'seed'             : 2413340114,
        'seed'             : gen_seeds(),            # Variance is too small if we don't do this
        # <<
    }
    
    # --
    #  Define data
    
    y = torch.LongTensor(graph.labels)
    
    idx_train, idx_stop, idx_valid = gen_splits(graph.labels, idx_split_args, test=args.test)
    idx_train, idx_stop, idx_valid = map(torch.LongTensor, (idx_train, idx_stop, idx_valid))
    idx_all = torch.arange(graph.adj_matrix.shape[0])
    
    y_train, y_stop, y_valid = y[idx_train], y[idx_stop], y[idx_valid]
    
    idx_train, idx_stop, idx_valid = map(lambda x: x.cuda(), (idx_train, idx_stop, idx_valid))
    y_train, y_stop, y_valid       = map(lambda x: x.cuda(), (y_train, y_stop, y_valid))
    
    torch.manual_seed(seed=gen_seeds())
    
    model = EmbeddingPPNP(
        n_nodes = graph.adj_matrix.shape[0],
        ppr     = ExactPPR(graph.adj_matrix, alpha=args.alpha),
    ).cuda()
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    early_stopping = SimpleEarlyStopping(model)
    
    t = time()
    for epoch in range(args.max_epochs):
        
        # --
        # Train
        
        _ = model.train()
        
        node_enc, hood_enc = model(X=idx_all, idx=idx_train)
        
        train_loss = ((node_enc - hood_enc) ** 2).mean()
        train_loss = train_loss + args.reg_lambda / 2 * model.get_norm()
        
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        
        # --
        # Stop
        
        _ = model.eval()
        
        with torch.no_grad():
            node_enc, hood_enc = model(X=idx_all, idx=idx_stop)
            stop_loss = ((node_enc - hood_enc) ** 2).mean()
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

    _, hood_enc_train = model(X=idx_all, idx=idx_train)
    _, hood_enc_valid = model(X=idx_all, idx=idx_valid)
    
    hood_enc_train = hood_enc_train.detach().cpu().numpy()
    hood_enc_valid = hood_enc_valid.detach().cpu().numpy()

    model = LinearSVC().fit(hood_enc_train, y_train.detach().cpu().numpy())
    pred  = model.predict(hood_enc_valid)
    record['acc'] = (pred == y_valid.detach().cpu().numpy()).mean()
    # <<
    
    print(record)
    sys.stdout.flush()
    all_records.append(record)

# --
# Print summary

df = pd.DataFrame(all_records)

print('-' * 50, file=sys.stderr)
print(df.mean(), file=sys.stderr)
print(df.std(), file=sys.stderr)


