import itertools
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from tqdm import tqdm

from ..analysis.analysis_formats import CSVDataFrame

def complete_sims(sims, comps):
    if any([isinstance(key, (str,)) for key in sims.keys()]):
        new_sims = {}
        for i in range(len(comps), -1, -1):
            if f'rel_{i}' in sims.keys():
                new_sims[i] = new_sims[i+1]*sims[f'rel_{i}']
            else:
                new_sims[i] = sims[i]
        sims = new_sims
    sims = {
        combination: sims[combination] if combination in sims.keys() else sims[sum(combination)]\
            for combination in itertools.product(*[(1,0) for n_comp in comps])
    }
    return sims

def is_pos_def(x, tol=1e-8):
    return np.all(np.linalg.eigvals(x) >= -tol)

def sims_pos_def(sims, comps):
    sims = complete_sims(sims, comps)
    all_inputs = np.array(list(itertools.product(*[range(n_comp) for n_comp in comps])))
    return is_pos_def(kernel(all_inputs, all_inputs, sims))

def kernel(x1, x2, sims):
    overlap = (np.expand_dims(x1, 1)==np.expand_dims(x2, 0)).astype(int)
    mat = (np.array(list(sims.keys())).T[None,None]==overlap[:,:,:,None]).all(axis=-2)*np.array(list(sims.values()))
    mat = mat.sum(axis=-1)
    return mat

def kernel_index(x1, x2, sims):
    mat = np.array([[
        sims[*_x1, *_x2] for _x2 in x2
    ] for _x1 in x1])
    return mat

def binary_accuracy(preds, targets):
    return (np.sign(preds)==np.sign(targets)).astype(float).mean()

def mse(preds, targets):
    return np.mean((preds-targets)**2)

def margin(preds, targets):
    return np.mean(preds*targets)

metrics = {
    'binary_accuracy': binary_accuracy,
    'mse': mse,
    'margin': margin
}

def get_kernel_machine(cfg, analysis, comps, metrics):
    C = eval(cfg.C) if isinstance(cfg.C, str) else cfg.C
    if isinstance(C, (float,)):
        C = [cfg.C]
    if cfg.equivariant:
        sims = {key if 'rel' in key else eval(key): eval(value) if isinstance(value, str) else [value] for key, value in cfg.sims.items()}
        combinations = list(itertools.product(*(list(sims.values())+[C])))
        sims = [
            {key: value for key, value in zip(sims.keys(), _comb[:(-1)])}\
                for _comb in combinations
        ]
        C = [_comb[-1] for _comb in combinations]
        return KernelMachines(sims, comps, cfg, analysis, C=C, metrics=metrics)
    else:
        kernel_file = np.load(cfg.kernel_file)
        combinations = list(itertools.product(list(kernel_file.keys()), C))
        keys = [_comb[0] for _comb in combinations]
        C = [_comb[1] for _comb in combinations]
        return KernelMachines(keys, comps, cfg, analysis, C=C, metrics=metrics, equivariant=False, sim_kernels = kernel_file)

class KernelMachines:
    def __init__(self, sims: list[dict[int|tuple[int], float]], comps, cfg, analysis, C:list[int]|int=1., metrics=None, equivariant: bool=True, sim_kernels: str|None=None):
        super().__init__()
        if not isinstance(C, list):
            C = [C]*len(sims)
        self.equivariant = equivariant
        if equivariant:
            self.models = []
            for _sims, _C in zip(sims, C):
                model = KernelMachine(_sims, comps, analysis, objective=cfg.objective, C=_C, metrics=metrics)
                if sims_pos_def(model.sims, comps):
                    self.models.append(model)
        else:
            self.models = {}
            for _sims, _C in zip(sims, C):
                model = NonEquivariantKernelMachine(sim_kernels[_sims], comps, analysis, objective=cfg.objective, C=_C, metrics=metrics)
                self.models[_sims] = model
        self.cfg = cfg
    
    def fit(self, train_data, test_data):
        wd = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        header, mode = True, 'w'
        if self.equivariant:
            for model in tqdm(self.models):
                dfs = model.fit(train_data, test_data)
                for df in dfs:
                    df['C'] = model.C
                    for sim_k, sim_v in model.sims.items():
                        df[f'sim_{sim_k}'] = sim_v
                    df.save(wd)
                header, mode = False, 'a'
        else:
            for key, model in tqdm(self.models.items()):
                dfs = model.fit(train_data, test_data)
                for df in dfs:
                    df['C'] = model.C
                    df['key'] = key
                    df.save(wd)
                header, mode = False, 'a'

class KernelMachine:
    def __init__(self, sims: dict[int|tuple[int], float], comps, analysis, objective='regression', C=1., metrics=None):
        super().__init__()
        self.sims = complete_sims(sims, comps)
        self.C = C
        self.model = {
            'regression': SVR,
            'classification': SVC
        }[objective](C=C, kernel=(lambda x1, x2: kernel(x1, x2, self.sims)))
        self.analysis = analysis
        self.metrics = metrics
        self.objective =objective

    def fit(self, train_data, test_data):
        x = np.stack([_x for _x, _ in train_data])
        y = np.array([_y for _, _y in train_data])
        self.model.fit(x, y)
        x = np.stack([_x for (_x, _), _ in test_data])
        y = np.array([_y for (_, _y), _ in test_data])
        splits = np.array([_split['split'] for _, _split in test_data])
        if self.objective == 'regression':
            preds = self.model.predict(x)
        if self.objective == 'classification':
            preds = self.model.decision_function(x)
        dfs = self.analysis(x, preds, splits, y)
        df_metrics = [
            pd.DataFrame({
                'split': np.unique(splits),
                'metric': 'score',
                'value': [self.model.score(x[splits==split], y[splits==split]) for split in np.unique(splits)]
            })
        ]
        for metric in self.metrics:
            df_metrics.append(
                pd.DataFrame({
                    'split': np.unique(splits),
                    'metric': metric,
                    'value': [metrics[metric](preds[splits==split], y[splits==split]) for split in np.unique(splits)]
                })
            )
        df_metrics = CSVDataFrame(pd.concat(df_metrics))
        df_metrics.file_name = 'metrics'
        dfs.append(df_metrics)
        return dfs

class NonEquivariantKernelMachine(KernelMachine):
    def __init__(self, sims, comps, analysis, objective='regression', C=1., metrics=None):
        self.C = C
        self.model = {
            'regression': SVR,
            'classification': SVC
        }[objective](C=C, kernel=(lambda x1, x2: kernel_index(x1, x2, sims)))
        self.analysis = analysis
        self.metrics = metrics
        self.objective =objective
