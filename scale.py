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
from ppr import ExactPPR, DenseNibblePPR, SparseNibblePPR, PrecomputedSparseNibblePPR

def gen_seeds():
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(max_uint32 + 1, size=1, dtype=np.uint32)

def parse_args():
    parser = argparse.ArgumentParser()
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
    
    parser.add_argument('--batch-size',       type=int, default=1024)
    
    parser.add_argument('--verbose', action="store_true")
    
    args = parser.parse_args()
    
    return args

args = parse_args()
set_seeds(args.seed)

args.ppr_topk = 128
args.verbose  = True

def load():
    data    = np.load('data/youtube/adj_data.npy')
    indices = np.load('data/youtube/adj_indices.npy')
    indptr  = np.load('data/youtube/adj_indptr.npy')
    mids    = np.load('data/youtube/adj_ids.npy')
    
    n_nodes = indices.max() + 1
    adj = sp.csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))
    
    assert len(mids) == n_nodes
    
    return adj, mids

adj, mids = load()

# >>
# n       = adj.shape[0]
# n_train = int(n * 0.9)
# --

n       = 500_000
adj     = adj[:n,:n]

keep = np.where(adj @ np.ones(adj.shape[0]))[0]
mids = mids[keep]
adj  = adj[keep][:,keep]

n_train = int(adj.shape[0] * 0.9)

# <<

adj = sp.eye(adj.shape[0]) + adj
adj = (adj > 0).astype(np.float32)
adj = adj.copy()

assert (adj != adj.T).sum() == 0

# --
#  Define data

X = torch.arange(adj.shape[0]) # Just doing embeddings

t = time()
# >>
# indices = torch.load('ppr.indices')
# values  = torch.load('ppr.values')
# ppr = PrecomputedSparseNibblePPR(indices=indices, values=values)
# --
ppr = SparseNibblePPR(adj=adj, alpha=args.alpha, topk=args.ppr_topk)
# torch.save((ppr.indices, ppr.values), 'ppr.1138499')
# <<

ppr_elapsed = time() - t

print('ppr_elapsed', ppr_elapsed, file=sys.stderr)
# 3.25 hours for whole graph
# 500s for 224K nodes

model = EmbeddingPPNP(
    n_nodes = adj.shape[0],
    ppr     = ppr
).cuda()

opt = torch.optim.Adam(model.parameters(), lr=0.05) # !! Should anneal

idx = torch.randperm(X.shape[0])
idx_train, idx_valid = idx[:n_train], idx[n_train:]

args.batch_size = 1024
train_loader = DataLoader(TensorDataset(idx_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = list(DataLoader(TensorDataset(idx_valid), batch_size=args.batch_size, shuffle=False, num_workers=0))

for param_group in opt.param_groups:
        param_group['lr'] *= 0.5

t = time()
for epoch in range(args.max_epochs):
    
    # --
    # Train
    
    _ = model.train()
    
    train_loss    = 0
    train_batches = len(train_loader)
    gen = tqdm(train_loader, total=train_batches)
    for idx_batch, in gen:
        node_enc, hood_enc = model(X=X, idx=idx_batch)
        
        loss = ((node_enc - hood_enc) ** 2).mean()
        loss = loss + args.reg_lambda / 2 * model.get_norm()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss
        
        gen.set_postfix(**{'loss' : 1000 * float(loss)})
    
    train_loss = 1000 * train_loss / train_batches
    
    # --
    # Valid
    
    _ = model.eval()
    
    valid_loss    = 0
    valid_batches = len(valid_loader)
    gen = tqdm(valid_loader, total=valid_batches)
    for idx_batch, in gen:
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split

_ = model.eval()

all_encs = []
idx_all  = torch.arange(adj.shape[0])
valid_loader = DataLoader(TensorDataset(idx_all), batch_size=args.batch_size, shuffle=False, num_workers=0)
for idx_batch, in tqdm(valid_loader, total=len(valid_loader)):
    _, hood_enc = model(X=X, idx=idx_batch)
    hood_enc = hood_enc.detach().cpu().numpy()
    all_encs.append(hood_enc)


all_encs = np.vstack(all_encs)
all_encs_orig = all_encs.copy()
all_encs /= np.sqrt((all_encs ** 2).sum(axis=-1, keepdims=True))

labels         = pd.read_csv('/home/bjohnson/.graphvite/dataset/youtube/youtube_label.txt', header=None, sep='\t')
labels.columns = ('id', 'label')
labels         = labels[labels.id.isin(mids)]

# !! Doing the mapping correctly?
# mids[i] is the id corresponding to the i'th row of all_encs
id2pos        = dict(zip(mids, range(len(mids))))
labels['pos'] = labels.id.apply(id2pos.get)

ulabs = np.unique(labels.label.values)
upos  = labels.pos.drop_duplicates().sort_values().reset_index(drop=True)
Y     = np.column_stack([upos.isin(labels.pos[labels.label == l]).values for l in ulabs])

XX = all_encs[upos.values]

XX_train, XX_valid, Y_train, Y_valid = train_test_split(XX, Y, train_size=0.05)
classifier = RandomForestClassifier(n_estimators=1024, n_jobs=60, verbose=1).fit(XX_train, Y_train)

# >>
# preds = classifier.predict(XX_valid)
# --
# !! How graphvite does prediction -- IMO this is sketchy
scores = np.column_stack([p[:,1] for p in classifier.predict_proba(XX_valid)])
thresh = np.sort(scores, axis=-1)[(np.arange(scores.shape[0]), -Y_valid.sum(axis=-1))]
preds  = (scores >= thresh.reshape(-1, 1))
# <<

metrics.f1_score(Y_valid, preds, average='micro')
metrics.f1_score(Y_valid, preds, average='macro')

# 40 epochs (5%)
# 0.3260745874928878
# 0.20069794610611497

# .. more epochs ..
# 0.41138600581887214
# 0.3038943278914481

# >>

from tqdm import trange
from sklearn.svm import LinearSVC

scores = [LinearSVC().fit(XX_train, Y_train[:,i]).decision_function(XX_valid) for i in trange(Y_train.shape[1])]
scores = np.column_stack(scores)
thresh = np.sort(scores, axis=-1)[(np.arange(scores.shape[0]), -Y_valid.sum(axis=-1))]
preds  = (scores >= thresh.reshape(-1, 1))
f1_micro = metrics.f1_score(Y_valid, preds, average='micro')
f1_macro = metrics.f1_score(Y_valid, preds, average='macro')

f1_micro
f1_macro
# <<
