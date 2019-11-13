#!/usr/bin/env python

"""
    orig.py
    
    Straight from notebook in repo
"""

import logging
import numpy as np

from ppnp.pytorch import PPNP
from ppnp.pytorch.training import train_model
from ppnp.pytorch.propagation import PPRExact, PPRPowerIteration
from ppnp.data.io import load_dataset

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

num_runs = 5

all_accs = []
for _ in range(num_runs):
    graph_name = 'cora_ml'
    graph      = load_dataset(graph_name)
    graph.standardize(select_lcc=True)
    
    model_args = {
        'hiddenunits' : [64],
        'drop_prob'   : 0.5,
        
        # 'propagation' : PPRPowerIteration(graph.adj_matrix, alpha=0.1, niter=10)
        'propagation' : PPRExact(graph.adj_matrix, alpha=0.1)
    }
    
    reg_lambda     = 5e-3
    learning_rate  = 0.01
    
    model, accs = train_model(
        name=graph_name,
        model_class=PPNP,
        graph=graph,
        model_args=model_args,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        test=True,
        torch_seed=None,
        print_interval=20
    )
    
    all_accs.append(accs)
    for a in all_accs:
        print(a)


# {'stopping_acc': 0.848, 'valtest_acc': 0.8465116279069768, 'test': False}
# {'stopping_acc': 0.846, 'valtest_acc': 0.8430232558139535, 'test': False}
# {'stopping_acc': 0.844, 'valtest_acc': 0.8406976744186047, 'test': False}
# {'stopping_acc': 0.848, 'valtest_acc': 0.8465116279069768, 'test': False}
# {'stopping_acc': 0.844, 'valtest_acc': 0.8406976744186047, 'test': False}

# {'stopping_acc': 0.844, 'valtest_acc': 0.8465648854961833, 'test': True}
# {'stopping_acc': 0.844, 'valtest_acc': 0.8564885496183207, 'test': True}
# {'stopping_acc': 0.842, 'valtest_acc': 0.850381679389313, 'test': True}
# {'stopping_acc': 0.85, 'valtest_acc': 0.849618320610687, 'test': True}
# {'stopping_acc': 0.85, 'valtest_acc': 0.8526717557251908, 'test': True}
