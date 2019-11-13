from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MixedLinear, MixedDropout


class PPNP(nn.Module):
    def __init__(self, nfeatures: int, nclasses: int, hiddenunits: List[int], drop_prob: float,
                 propagation: nn.Module, bias: bool = False):
        
        super().__init__()
        
        self.fcs = nn.ModuleList([
            MixedLinear(nfeatures, hiddenunits[0], bias=bias)
        ] + [
            nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias)
                for i in range(1, len(hiddenunits))
        ] + [
            nn.Linear(hiddenunits[-1], nclasses, bias=bias)
        ])
        
        self.reg_params = list(self.fcs[0].parameters())
        
        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        
        self.act_fn = nn.ReLU()
        
        self.propagation = propagation

    def _transform_features(self, x):
        x = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            print('hidden!')
            x = self.act_fn(fc(x))
        
        x = self.fcs[-1](self.dropout(x))
        return x

    def forward(self, x, idx):
        x = self._transform_features(x)
        x = self.propagation(x, idx)
        return F.log_softmax(x, dim=-1)

# dropout
# dense
# act
# 