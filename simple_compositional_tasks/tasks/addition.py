import warnings

import numpy as np

from . import tasks

def random_binary_matrix(n, n_ones, seed=None):
    rng = np.random.default_rng(seed)
    mat = np.array([
        [
            int((j-i)%n<=n_ones-1) for j in range(n)
        ] for i in range(n)
    ])
    rng.shuffle(mat)
    return mat

class Addition(tasks.Task):
    def __init__(self, values=None, training_comps=None, splits=None, cfg=None):
        values = values or cfg.task_values
        x = tasks.expand_grid(*[len(_v) for _v in values])
        y = np.array([
            sum([values[comp][int(_i)] for comp, _i in enumerate(i)]) for i in x
        ]).astype(float)
        if cfg is not None:
            if cfg.training_selection == 'comps':
                training_comps = training_comps or cfg.training_comps
            if cfg.training_selection == 'dispersed':
                assert len(values)==2
                assert len(values[0])==len(values[1])
                n = len(values[0])
                mat = random_binary_matrix(n, n_ones=cfg.n_training_instances, seed=cfg.instances_seed)
                splits = lambda x: mat[*x]==1
        if splits is not None and training_comps is not None:
            warnings.warn(
                """splits function is specified but is overridden
                by 'training_comps'.
                """
            )
        if splits is None and training_comps is None:
            raise ValueError(
                """You must specify either 'training_comps'
                or 'splits'.
                """
            )
        if training_comps is not None:
            splits = lambda x: any([_x in comp for _x, comp in zip(x, training_comps)])
        super().__init__(x, y, splits, cfg=cfg)
        print(self.df)
        for comp, v in enumerate(values):
            self.df[f'v{comp}'] = [values[comp][int(_x[comp])] for _x in x]
