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
from tqdm import tqdm
from time import time
import scipy.sparse as sp

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.deterministic = True

from ppnp.data.sparsegraph import SparseGraph
from ppnp.preprocessing import gen_splits, normalize_attributes

from model import EmbeddingPPNP
from helpers import set_seeds, SimpleEarlyStopping
from ppr import ExactPPR, DenseNibblePPR, SparseNibblePPR

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
    parser.add_argument('--max-epochs',       type=int,   default=40)
    parser.add_argument('--reg-lambda',       type=float, default=5e-3)
    parser.add_argument('--lr',               type=float, default=0.01)
    parser.add_argument('--alpha',            type=float, default=0.1)
    parser.add_argument('--test',             action="store_true")
    
    parser.add_argument('--sparse',           action="store_true")
    parser.add_argument('--ppr-topk',         type=int)
    parser.add_argument('--ppr-mode',         type=str, default='exact')
    
    parser.add_argument('--batch-size',       type=int, default=256)
    
    parser.add_argument('--verbose', action="store_true")
    
    args = parser.parse_args()
    
    is_ms_academic = 'ms_academic' in args.inpath
    if is_ms_academic:
        args.alpha  = 0.2
        args.nknown = 5000
    
    return args

args = parse_args()
set_seeds(args.seed)

args.ppr_topk = 128
args.verbose  = True

def load():
    data    = np.load('data/youtube/adj_data.npy')
    indices = np.load('data/youtube/adj_indices.npy')
    indptr  = np.load('data/youtube/adj_indptr.npy')
    
    n_nodes = indices.max() + 1
    return sp.csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))

adj = load()

n       = 100_000
n_train = int(n * 0.9)

adj     = adj[:n, :n]

adj = sp.eye(adj.shape[0]) + adj
adj = (adj > 0).astype(np.float32)

# --
#  Define data

X = torch.arange(adj.shape[0]) # Just doing embeddings

t = time()
ppr = SparseNibblePPR(adj=adj, alpha=args.alpha, topk=args.ppr_topk)
ppr_elapsed = time() - t

print('ppr_elapsed', ppr_elapsed, file=sys.stderr)

model = EmbeddingPPNP(
    n_nodes = adj.shape[0],
    ppr     = ppr
).cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

idx    = torch.randperm(X.shape[0])
idx_train, idx_valid = idx[:n_train], idx[n_train:]

train_loader = DataLoader(TensorDataset(idx_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(TensorDataset(idx_valid), batch_size=args.batch_size, shuffle=False, num_workers=0)

t = time()
for epoch in range(args.max_epochs):
    
    # --
    # Train
    
    _ = model.train()
    
    train_loss    = 0
    train_batches = len(train_loader)
    for idx_batch, in tqdm(train_loader, total=train_batches):
        node_enc, hood_enc = model(X=X, idx=idx_batch)
        
        loss = ((node_enc - hood_enc) ** 2).mean()
        loss = loss + args.reg_lambda / 2 * model.get_norm()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss
    
    train_loss = 1000 * train_loss / train_batches
    
    # --
    # Valid
    
    _ = model.eval()
    
    valid_loss    = 0
    valid_batches = len(valid_loader)
    for idx_batch, in tqdm(valid_loader, total=valid_batches):
        node_enc, hood_enc = model(X=X, idx=idx_batch)
        
        loss = ((node_enc - hood_enc) ** 2).mean()
        loss = loss + args.reg_lambda / 2 * model.get_norm()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        valid_loss += loss
    
    valid_loss = 1000 * valid_loss / valid_batches
    
    record = {
        "epoch"      : int(epoch),
        "elapsed"    : float(time() - t),
        "train_loss" : float(train_loss),
        "valid_loss" : float(valid_loss),
    }
    
    if args.verbose:
        print(json.dumps(record), file=sys.stderr)
        sys.stderr.flush()


# --
# Train classifier

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split

all_encs = []
idx_all      = torch.arange(adj.shape[0])
valid_loader = DataLoader(TensorDataset(idx_all), batch_size=args.batch_size, shuffle=False, num_workers=0)
for idx_batch, in tqdm(valid_loader, total=len(valid_loader)):
    _, hood_enc = model(X=X, idx=idx_batch)
    hood_enc = hood_enc.detach().cpu().numpy()
    all_encs.append(hood_enc)


all_encs = np.vstack(all_encs)
all_encs /= np.sqrt((all_encs ** 2).sum(axis=-1, keepdims=True))

labels         = pd.read_csv('/home/bjohnson/.graphvite/dataset/youtube/youtube_label.txt', header=None, sep='\t')
labels.columns = ('id', 'label')
labels         = labels[labels.id.isin(idx_all)]

ulabs = np.array(labels.label.value_counts().index)
Y     = np.column_stack([labels.id.isin(labels.id[labels.label == l]).values for l in ulabs])

XX = all_encs[labels.id.values]

XX_train, XX_valid, Y_train, Y_valid = train_test_split(XX, Y, train_size=0.1)
classifier = RandomForestClassifier(n_estimators=100, n_jobs=60, verbose=1).fit(XX_train, Y_train)
preds = classifier.predict(XX_valid)
metrics.f1_score(Y_valid, preds, average='micro')
metrics.f1_score(Y_valid, preds, average='macro')

