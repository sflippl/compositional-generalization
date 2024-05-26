import numpy as np

from . import tasks

class TransitiveEquivalence(tasks.Task):
    def __init__(self, classes:list[int]|None=None, train_items:list[int]|None=None, cfg=None):
        classes = classes or cfg.classes
        train_items = train_items or cfg.train_items
        x = tasks.expand_grid(sum(classes), sum(classes))
        cumsum_classes = np.cumsum([0]+classes[:(-1)])
        y = np.where(
            np.sum(x[:,0].reshape(-1,1) >= cumsum_classes, axis=1)==np.sum(x[:,1].reshape(-1,1) >= cumsum_classes, axis=1),
            1., -1.
        )
        splits = lambda x: any(
            (x[0]>=cumsum_classes)&(x[0]<cumsum_classes+np.array(train_items))|\
            (x[1]>=cumsum_classes)&(x[1]<cumsum_classes+np.array(train_items))
        )
        super().__init__(x, y, splits)
