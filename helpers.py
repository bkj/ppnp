#!/usr/bin/env python

"""
    helpers.py
"""

import sys
import torch
import random
import numpy as np
import scipy.sparse as sp

def set_seeds(seed):
    _ = random.seed(seed + 1)
    _ = np.random.seed(seed + 2)
    _ = torch.manual_seed(seed + 3)
    _ = torch.cuda.manual_seed(seed + 4)

class SimpleEarlyStopping:
    def __init__(self, model, patience=100, store_weights=False):
        
        self.model         = model
        self.patience      = patience
        self.max_patience  = patience
        self.store_weights = store_weights
        self.record        = None,
        
        self.best_acc   = -np.inf
        self.best_nloss = -np.inf
        
        self.best_epoch       = -1
        self.best_epoch_score = (-np.inf, -np.inf)
    
    def should_stop(self, acc, loss, epoch, record=None):
        nloss = -1 * loss
        
        if (acc < self.best_acc) and (nloss < self.best_nloss):
            self.patience -= 1
            return self.patience == 0
        
        self.patience = self.max_patience
        
        self.best_acc   = max(acc, self.best_acc)
        self.best_nloss = max(nloss, self.best_nloss)
        
        if (acc, nloss) > self.best_epoch_score:
            self.best_epoch       = epoch
            self.best_epoch_score = (acc, nloss)
            if self.store_weights:
                self.best_state = {k:v.cpu() for k,v in self.model.state_dict().items()}
            
            if record:
                self.record = record
        
        return False
