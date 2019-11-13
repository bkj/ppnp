#!/usr/bin/env python

"""
    simple.py
"""

import sys
from time import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from ppnp.pytorch.ppnp import PPNP
from ppnp.pytorch.training import train_model
from ppnp.pytorch.earlystopping import stopping_args, EarlyStopping
from ppnp.pytorch.propagation import PPRExact, PPRPowerIteration
from ppnp.data.io import load_dataset
from ppnp.data.sparsegraph import SparseGraph
from ppnp.pytorch.utils import matrix_to_torch
from ppnp.preprocessing import gen_seeds, gen_splits, normalize_attributes

# --
# IO

graph_name = 'cora_ml'
graph = load_dataset(graph_name)
graph.standardize(select_lcc=True)

# --
# Params

model_args = {
    'hiddenunits' : [64],
    'drop_prob'   : 0.5,
    
    'propagation' : PPRPowerIteration(graph.adj_matrix, alpha=0.1, niter=10),
    # 'propagation' : PPRExact(graph.adj_matrix, alpha=0.1)
}

idx_split_args = {
    'ntrain_per_class' : 20, 
    'nstopping'        : 500, 
    'nknown'           : 1500, 
    'seed'             : 2413340114
}

reg_lambda    = 5e-3
learning_rate = 0.01

test           = False
device         = 'cuda'
print_interval = 20
torch_seed     = 789

model_class = PPNP

# --
# Helpers

def get_predictions(model, attr_matrix, idx, batch_size=None):
    if batch_size is None:
        batch_size = idx.numel()
    
    dataset    = TensorDataset(idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    preds = []
    for idx, in dataloader:
        idx = idx.to(attr_matrix.device)
        with torch.set_grad_enabled(False):
            log_preds = model(attr_matrix, idx)
            preds.append(torch.argmax(log_preds, dim=1))
    
    return torch.cat(preds, dim=0).cpu().numpy()

def get_dataloaders(idx, labels_np, batch_size=None):
    labels = torch.LongTensor(labels_np)
    
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    
    datasets    = {phase: TensorDataset(ind, labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) for phase, dataset in datasets.items()}
    return dataloaders

# --
# Run train

labels_all = graph.labels

idx_np = {}
idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits(
        labels_all, idx_split_args, test=test)

# training: ntrain_per_class * num_classes
# stopping: 500
# valtest: nknown - (n_training + n_stopping)

idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}

if torch_seed is None:
    torch_seed = gen_seeds()

_ = torch.manual_seed(seed=torch_seed)
_ = torch.cuda.manual_seed(seed=torch_seed + 1)

nfeatures = graph.attr_matrix.shape[1]
nclasses  = max(labels_all) + 1
model     = model_class(nfeatures, nclasses, **model_args).to(device)
print(model, file=sys.stderr)

reg_lambda = torch.tensor(reg_lambda, device=device)
optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataloaders = get_dataloaders(idx_all, labels_all)

early_stopping = EarlyStopping(model, **stopping_args)

attr_mat_norm_np = normalize_attributes(graph.attr_matrix)
attr_mat_norm    = matrix_to_torch(attr_mat_norm_np).to(device)

epoch_stats = {'train': {}, 'stopping': {}}

start_time = time()
for epoch in range(early_stopping.max_epochs):
    for phase in epoch_stats.keys():
        
        if phase == 'train':
            _ = model.train()
        else:
            _ = model.eval()
        
        running_loss     = 0
        running_corrects = 0
        
        for idx, labels in dataloaders[phase]:
            idx    = idx.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                log_preds = model(attr_mat_norm, idx)
                preds     = torch.argmax(log_preds, dim=1)
                
                cross_entropy_mean = F.nll_loss(log_preds, labels)
                l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                loss   = cross_entropy_mean + reg_lambda / 2 * l2_reg
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss     += loss.item() * idx.size(0)
                running_corrects += torch.sum(preds == labels)
        
        nobs = len(dataloaders[phase].dataset)
        epoch_stats[phase]['loss'] = running_loss / nobs
        epoch_stats[phase]['acc']  = running_corrects.item() / nobs
        
    if epoch % print_interval == 0:
        print({
            "epoch"               : epoch,
            "train_loss"          : epoch_stats['train']['loss'],
            "train_acc"           : epoch_stats['train']['acc'],
            "early_stopping_loss" : epoch_stats['stopping']['loss'],
            "early_stopping_acc"  : epoch_stats['stopping']['loss'],
            "elapsed"             : time() - start_time
        })
        
    if len(early_stopping.stop_vars) > 0:
        stop_vars = [epoch_stats['stopping'][key] for key in early_stopping.stop_vars]
        if early_stopping.check(stop_vars, epoch):
            break

best_state = early_stopping.best_state
if 'propagation.A_hat' in best_state:
    del best_state['propagation.A_hat']

model.load_state_dict(best_state, strict=False)

stopping_preds = get_predictions(model, attr_mat_norm, idx_all['stopping'])
stopping_acc   = (stopping_preds == labels_all[idx_all['stopping']]).mean()

valtest_preds = get_predictions(model, attr_mat_norm, idx_all['valtest'])
valtest_acc   = (valtest_preds == labels_all[idx_all['valtest']]).mean()

print({
    "stopping_acc" : stopping_acc,
    "valtest_acc"  : valtest_acc,
})
