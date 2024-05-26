import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from omegaconf import DictConfig, OmegaConf

from ..structure.errors import check_column

def expand_grid(*comps):
    return np.array(list(itertools.product(*[range(n) for n in comps])))

def _leave_out_conjunctions(x, leftout_conjunctions):
    for conj in leftout_conjunctions:
        is_leftout = all([x[i] in conj_i for i, conj_i in conj.items()])
        if is_leftout:
            return False
    return True

def leave_out_conjunctions(leftout_conjunctions):
    def fun(x):
        return _leave_out_conjunctions(x, leftout_conjunctions)
    return fun

class Task:
    def __init__(self, x: np.ndarray, y: np.ndarray, splits: pd.DataFrame | np.ndarray[bool], cfg: DictConfig=None):
        super().__init__()
        self.cfg = cfg
        self.x = x
        self.y = y
        if isinstance(splits, np.ndarray):
            self.df = pd.DataFrame({
                'idx': np.arange(len(self.x)),
                'split': np.where(splits, 'train', 'test')
            })
        elif callable(splits):
            self.df = pd.DataFrame({
                'idx': np.arange(len(self.x)),
                'split': ['train' if splits(_x) else 'test' for _x in x]
            })
        else:
            self.df = splits
        self.df['spec'] = list(x)
        check_column(self.df, 'idx', "Argument 'splits'")
        check_column(self.df, 'split', "Argument 'splits'")
