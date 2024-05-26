import numpy as np

from . import tasks

class ContextDependence(tasks.Task):
    def __init__(
            self, contexts:int=None, features:list[int]=None, leftout_conjunctions=None,
            context_to_feature=None, feature_to_response=None,
            cfg=None
        ):
        print(feature_to_response)
        contexts = contexts or cfg.contexts
        features = features or cfg.features
        leftout_conjunctions = leftout_conjunctions or cfg.leftout_conjunctions
        context_to_feature = context_to_feature or cfg.context_to_feature
        feature_to_response = feature_to_response or cfg.feature_to_response
        x = tasks.expand_grid(*([contexts]+features))
        y = np.array([
            feature_to_response[context_to_feature[int(_x[0])]][int(_x[context_to_feature[int(_x[0])]+1])] for _x in x
        ]).astype(float)
        splits = tasks.leave_out_conjunctions(leftout_conjunctions)
        super().__init__(x, y, splits, cfg=cfg)
        self.df['context'] = [context_to_feature[int(_x[0])] for _x in x]
        for comp in range(1, x.shape[1]):
            self.df[f'response_feat{comp-1}'] = [feature_to_response[comp-1][int(_x[comp])] for _x in x]
