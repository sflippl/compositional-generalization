from pathlib import Path
import os

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import hydra
import torch

from ..embeddings.schema import AllConjunctionsGenerator
from .analysis_formats import CSVDataFrame

class AdditivityAnalysis:
    def __init__(self, x, df):
        super().__init__()
        self.generator = AllConjunctionsGenerator(x)
        pred_conj = self.generator(x)
        self.conj_selector = (pred_conj[df.split=='train']!=0).any(axis=0)
        self.labels = self.generator.get_labels(x)[list(self.conj_selector)+[True]]
        self.predictor = Ridge(alpha=1e-10)
        self.validation_step_outputs = []
        self.validation_step_splits = []
        self.validation_step_specs = []

    def __call__(self, specs, yhat, splits, y=None):
        df_yhat = pd.DataFrame({
            'specs': [tuple(spec) for spec in specs],
            'splits': splits,
            'yhat': yhat
        })
        df_yhat = df_yhat.groupby(['specs', 'splits'])['yhat'].mean().reset_index()
        specs = np.array([list(spec) for spec in df_yhat['specs']])
        yhat = df_yhat['yhat']
        splits = df_yhat['splits']
        pred_conj = self.get_features(specs)
        self.predictor.fit(pred_conj, yhat)
        split_values = np.unique(splits)
        additivity = CSVDataFrame({
            'split': split_values,
            'additivity': [
                self.predictor.score(pred_conj[splits==split], yhat[splits==split]) for split in split_values
            ]
        })
        new_label_df = CSVDataFrame(self.labels.copy())
        new_label_df['value'] = np.concatenate([self.predictor.coef_, np.array([self.predictor.intercept_])])
        additivity.file_name = 'additivity'
        new_label_df.file_name = 'labels'
        return [additivity, new_label_df]

    def get_features(self, specs):
        pred_conj = self.generator(specs)
        return pred_conj[:, self.conj_selector]

    def validation_step(self, model, x, y, yhat, row):
        self.validation_step_outputs.append(yhat)
        self.validation_step_splits.append(row['split'])
        self.validation_step_specs.append(row['spec'])

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        all_specs = torch.cat(self.validation_step_specs)
        all_splits = np.concatenate(self.validation_step_splits)
        rtn = self(all_specs.cpu().numpy(), all_preds.cpu().numpy(), all_splits)
        self.validation_step_outputs.clear()
        self.validation_step_splits.clear()
        self.validation_step_specs.clear()
        return rtn
