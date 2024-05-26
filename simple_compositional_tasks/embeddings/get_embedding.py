import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from .schema import generator_dct
from .image_categories import get_image_generator

class DatasetWithDf(Dataset):
    def __init__(self, data, df, row_fun=None, input_transform=None, output_transform=None, split='train'):
        super().__init__()
        self.data = data
        self.df = df
        self.row_fun = row_fun
        self.input_transform = input_transform or (lambda x: x)
        self.output_transform = output_transform or (lambda x: x)
        self.split = split
    
    def __getitem__(self, index):
        #index = self.df.index[index]
        x, y = self.data(split=self.split, *self.df.index[index])
        x = self.input_transform(x)
        y = self.output_transform(y)
        if self.row_fun is None:
            return x,y
        return (x,y), self.row_fun(self.df.reset_index().loc[index])

    def __len__(self):
        return len(self.df)

def get_embeddings(cfg, task):
    input_embedding = get_embedding(cfg.input_generator, task.x, task.df.split=='train')
    output_embedding = get_embedding(cfg.output_generator, task.y, task.df.split=='train')
    return Embeddings(input_embedding, output_embedding, task)

def get_embedding(cfg, spec, split):
    if cfg.generator == 'schema':
        return generator_dct[cfg.type](spec, permutation=cfg.permutation, permutation_seed=cfg.permutation_seed, permutation_index=cfg.permutation_index)
    if cfg.generator == 'image':
        return get_image_generator(cfg, spec, split)

class Embeddings:
    def __init__(self, input_embedding, output_embedding, task):
        super().__init__()
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.df = self.input_embedding.df.reset_index().rename(
                columns={'emb_idx': 'input_idx', 'split': 'input_split'}
            ).merge(
                self.output_embedding.df.reset_index().rename(columns={'emb_idx': 'output_idx', 'split': 'output_split'}),
                on='spec_idx'
            ).merge(
                task.df.rename(columns={'idx': 'spec_idx', 'split': 'task_split'}),
                on='spec_idx'
            ).set_index(['spec_idx', 'input_idx', 'output_idx'])
        self.df['split'] = 'task='+self.df.task_split
        self.df['train'] = self.df.task_split=='train'
        if 'input_split' in self.df.columns:
            self.df['split'] += '--input='+self.df.input_split
            self.df['train'] = np.logical_and(self.df.train, self.df.input_split=='train')
        if 'output_split' in self.df.columns:
            self.df['split'] += '--output='+self.df.output_split
            self.df['train'] = np.logical_and(self.df.train, self.df.output_split=='train')
    
    def __call__(self, spec_idx: int, input_idx: int, output_idx: int, split:str='train'):
        return (self.input_embedding(spec_idx, input_idx, split=split), self.output_embedding(spec_idx, output_idx))
    
    def create_dataset(self, split):
        row_fun = (lambda x: x.to_dict())
        if split == 'train':
            return DatasetWithDf(self, self.df[self.df.train], row_fun=None, split='train')
        else:
            return DatasetWithDf(self, self.df, row_fun=row_fun, split='test')
    
    @property
    def val_splits(self):
        return np.unique(self.df.split)
