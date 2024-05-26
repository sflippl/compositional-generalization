import torch
import numpy as np
import pandas as pd

from ..structure.array_to_dataframe import array_to_dataframe
from .analysis_formats import CSVDataFrame

class SimilarityAnalysis:
    def __init__(self, type='features', n_embeddings=10):
        super().__init__()
        self.type = type
        self.feats = []
        self.splits = []
        self.specs = []
        self.n_embeddings = n_embeddings
    
    def validation_step(self, model, x, y, yhat, row):
        selector = (row['input_idx'] <= self.n_embeddings)
        # if 'input_split' in row.keys():
        #     selector = selector & (row['input_split']=='val_2')
        if not selector.any():
            return None
        x, y, yhat = x[selector], y[selector], yhat[selector]
        if self.type == 'features':
            new_feats = model.model.get_features(x)
            if len(self.feats) == 0:
                self.feats = [[feat.cpu().detach()] for feat in new_feats]
            else:
                for feat, new_feat in zip(self.feats, new_feats):
                    feat.append(new_feat.cpu().detach())
        if self.type == 'ntk':
            new_feats = model.model.get_ntk(x).cpu().detach()
            self.feats.append(new_feats)
        self.splits.append(np.array(row['split'])[selector.cpu().numpy()])
        self.specs.append(row['spec'][selector].cpu())
    
    def __call__(self, sim, splits, specs, y=None):
        specs = [str(list(spec.numpy())) for spec in specs]
        df = array_to_dataframe(sim)\
            .merge(pd.DataFrame({'dim0': np.arange(len(splits)), 'split0': list(splits)}))\
            .merge(pd.DataFrame({'dim1': np.arange(len(splits)), 'split1': list(splits)}))\
            .merge(pd.DataFrame({'dim0': np.arange(len(specs)), 'spec0': list(specs)}))\
            .merge(pd.DataFrame({'dim1': np.arange(len(specs)), 'spec1': list(specs)}))
        df = df.groupby(['split0', 'split1',  'spec0', 'spec1'])['array'].mean().reset_index()
        return df
    
    def on_validation_epoch_end(self):
        specs = torch.cat(self.specs)
        splits = np.concatenate(self.splits)
        if self.type == 'features':
            feats = [torch.cat(feat, dim=0) for feat in self.feats]
            sims = [feat@feat.T for feat in feats]
            df = pd.concat([
                self(sim, splits, specs).assign(layer=i) for i, sim in enumerate(sims)
            ])
        if self.type == 'ntk':
            feats = torch.cat(self.feats, dim=0)
            sims = feats@feats.T
            df = self(sims, splits, specs)
        self.feats.clear()
        self.specs.clear()
        self.splits.clear()
        return [CSVDataFrame(df, file_name=f'similarity_{self.type}')]
