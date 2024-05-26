from copy import deepcopy
import math

import torch
import torch.nn as nn

from .get_ntk import get_ntk

def get_dense_net(inp_dim, outp_dim, cfg, **kwargs):
    return DenseNet(
        inp_dim=inp_dim,
        hdims=cfg.hdims,
        outp_dim=outp_dim,
        bias=cfg.bias,
        scaling=math.pow(10, cfg.log_scaling),
        linear_readout=cfg.linear_readout,
        seed=cfg.seed
    )

class DenseNet(nn.Module):
    def __init__(self, inp_dim, hdims=None, outp_dim=1, nonlinearity=nn.ReLU(), bias=False, scaling=1., linear_readout=False, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        hdims = hdims or []
        L = []
        for i, (_in, _out) in enumerate(zip([inp_dim]+hdims, hdims+[outp_dim])):
            if i>0:
                L.append(nonlinearity)
            linear = nn.Linear(_in, _out, bias=bias)
            nn.init.normal_(linear.weight, std=scaling*math.sqrt(2/_in))
            if bias:
                nn.init.uniform_(linear.bias, -scaling*math.sqrt(1/_in), scaling*math.sqrt(1/_in))
            L.append(linear)
        self.features = nn.Sequential(*L[:(-1)])
        self.readout = L[-1]
        self.linear_readout = linear_readout
        if linear_readout:
            nn.init.zeros_(linear.weight)
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        else:
            return super().parameters()
    
    def forward(self, x):
        if self.linear_readout:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = self.readout(x)
        return x

    @torch.no_grad()
    def get_features(self, x):
        lst = [x]
        for feat in self.features:
            x = feat(x)
            lst.append(x)
        x = self.readout(x)
        lst.append(x)
        return lst
    
    @torch.enable_grad()
    def get_ntk(self, x):
        return get_ntk(self, x)
