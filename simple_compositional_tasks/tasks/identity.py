import warnings

import numpy as np

from . import tasks

class Identity(tasks.Task):
    def __init__(self, comps, cfg=None):
        x = tasks.expand_grid(*comps)
        y = np.array([
            sum(_x, start=[]) for _x in x
        ])
        super().__init__(x, y, [True]*len(y), cfg=cfg)
        print(self.df)
