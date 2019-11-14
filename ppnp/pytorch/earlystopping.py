from typing import List
import copy
import operator
from enum import Enum, auto
import numpy as np

from torch.nn import Module


class StopVariable(Enum):
    LOSS = auto()
    ACCURACY = auto()
    NONE = auto()


class Best(Enum):
    RANKED = auto()
    ALL = auto()


stopping_args = dict(
        stop_varnames=[StopVariable.ACCURACY, StopVariable.LOSS],
        patience=100, max_epochs=10000, remember=Best.RANKED)


class SimpleEarlyStopping:
    def __init__(self, model, patience=100):
        
        self.model        = model
        self.patience     = patience
        self.max_patience = patience
        
        self.best_acc   = -np.inf
        self.best_nloss = -np.inf
        
        self.best_epoch       = -1
        self.best_epoch_score = (-np.inf, -np.inf)
    
    def should_stop(self, acc, loss, epoch):
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
            self.best_state       = {k:v.cpu() for k,v in self.model.state_dict().items()}
        
        return False

class EarlyStopping:
    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 10, max_epochs: int = 200, remember: Best = Best.ALL):
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None
        self.best_state = None

    def check(self, values: List[np.floating], epoch: int) -> bool:
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values)]
        if any(checks):
            self.best_vals = np.choose(checks, [self.best_vals, values])
            self.patience = self.max_patience

            comp_remembered = [
                    self.comp_ops[i](val, self.remembered_vals[i])
                    for i, val in enumerate(values)]
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    self.best_state = {
                            key: value.cpu() for key, value
                            in self.model.state_dict().items()}
            elif self.remember is Best.RANKED:
                # print(epoch, values, self.remembered_vals, comp_remembered)
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not(self.remembered_vals[i] == values[i]):
                            # print('save')
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            self.best_state = {
                                    key: value.cpu() for key, value
                                    in self.model.state_dict().items()}
                            break
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0
