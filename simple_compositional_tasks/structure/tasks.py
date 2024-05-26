import itertools
import abc
import argparse

import hydra
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from omegaconf import DictConfig, OmegaConf

from .errors import check_column
from .generics import arg_wrapper, resolve
from .generator import Generator, OnehotGenerator, AllConjunctionsGenerator, generate

def expand_grid(*comps):
    return np.array(list(itertools.product(*[range(n) for n in comps])))

class Task:
    _input_default_settings = {
        'generator': OnehotGenerator()
    }
    _output_default_settings = {
        'generator': Generator()
    }

    def __init__(self, x: np.ndarray, y: np.ndarray, splits: pd.DataFrame | np.ndarray[bool], metrics=None, cfg: DictConfig=None):
        super().__init__()
        self.cfg = cfg
        self.x = x
        self.y = y
        if isinstance(splits, np.ndarray):
            self.df = pd.DataFrame({
                'idx': np.arange(len(self.x)),
                'train': splits
            })
        elif callable(splits):
            self.df = pd.DataFrame({
                'idx': np.arange(len(self.x)),
                'train': [splits(_x) for _x in x]
            })
        else:
            self.df = splits
        check_column(self.df, 'idx', "Argument 'splits'")
        check_column(self.df, 'train', "Argument 'splits'")

    def generate_input(self, **kwargs):
        cfg = OmegaConf.merge(self.cfg['input_embedding'], kwargs)
        return generate(self.x, cfg)
    
    def generate_output(self, **kwargs):
        cfg = OmegaConf.merge(self.cfg['output_embedding'], kwargs)
        return generate(self.x, cfg)

    def compositional_indices(self, model, **kwargs):
        x = self.generate_input(**kwargs)
        yhat = model.eval(x)
        generator = AllConjunctionsGenerator()
        pred_conj = self.generate_input(generator = generator)
        labels = generator.get_labels(pred_conj)
        conj_selector = (pred_conj[self.df.train]!=0).any(axis=0)
        pred_conj = pred_conj[:, conj_selector]
        labels = labels.iloc[conj_selector]
        predictor = Ridge(alpha=1e-10)
        predictor.fit(pred_conj, yhat)
        additivity = predictor.score(pred_conj, yhat)
        additivity = pd.DataFrame({
            ['additivity']: [additivity]
        })
        labels['value'] = predictor.coef_
        return additivity, labels
    
    def analysis(self, model, **kwargs):
        return self.compositional_indices(model, **kwargs)
    
    def metrics(self, yhat, y):
        return [metric(yhat, y) for metric in self._metrics]
