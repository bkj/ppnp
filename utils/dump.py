#!/usr/bin/env python

"""
    prep.py
"""

import numpy as np
import pandas as pd
from scipy import sparse

inpath = '/home/bjohnson/.graphvite/dataset/youtube/youtube_graph.txt'

edges = pd.read_csv(inpath, header=None, sep='\t').values

edges -= edges.min()
n_nodes = edges.max() + 1

vals = np.ones(edges.shape[0])
adj  = sparse.csr_matrix((vals, (edges[:,0], edges[:,1])), shape=(n_nodes, n_nodes))

# Do we want to treat this as directory or undirected?
adj  = ((adj + adj.T) > 0).astype(np.float32)

# keep = np.where(adj @ np.ones(adj.shape[0]) != 0)[0]
# adj  = adj[keep]
# adj  = adj[:,keep]

adj.sort_indices()
adj.eliminate_zeros()

np.save('data/youtube/adj_data.npy', adj.data)
np.save('data/youtube/adj_indices.npy', adj.indices)
np.save('data/youtube/adj_indptr.npy', adj.indptr)
# np.save('adj_ids.npy', keep)


