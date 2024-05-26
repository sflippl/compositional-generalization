import numpy as np

from . import tasks

class TransitiveOrdering(tasks.Task):
    def __init__(self, n_items: int, train_sds: list[int]|None=None):
        train_sds = train_sds or [1]
        x = tasks.expand_grid(n_items, n_items)
        x = x[x[:,0]!=x[:,1]]
        y = np.array([
            1. if _x[0] < _x[1] else -1. for _x in x
        ])
        splits = (lambda x: np.abs(x[0]-x[1]) in train_sds)
        super().__init__(x, y, splits=splits)

def is_in(x, e):
    min_e = min(e)
    max_e = max(e)
    return all([(_x>=min_e)&(_x<=max_e) for _x in x])

class TransitiveOrderingWithExceptions(tasks.Task):
    def __init__(self, n_items: int, train_sds: list[int]|None=None, exceptions=None):
        train_sds = train_sds or [1]
        exceptions = exceptions or []
        x = tasks.expand_grid(n_items, n_items)
        x = x[x[:,0]!=x[:,1]]
        is_exception = np.array([
            any([((_x[0]==e[0])&(_x[1]==e[1]))|((_x[1]==e[0])&(_x[0]==e[1])) for e in exceptions])\
            for _x in x
        ])
        train_selector = np.array([(np.abs(_x[0]-_x[1]) in train_sds) for _x in x]) | is_exception
        is_enclosed_by_exception = np.array([
            any([is_in(_x, e) for e in exceptions]) for _x in x
        ])
        subset = (~is_enclosed_by_exception)|train_selector
        x = x[subset]
        is_exception = is_exception[subset]
        y = np.where(is_exception, -1., 1.) * np.where(x[:,0]>x[:,1], 1., -1.)
        splits = train_selector[subset]
        super().__init__(x, y, splits)
