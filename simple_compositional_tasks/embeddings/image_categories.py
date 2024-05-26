from typing import Optional, Callable, Sequence
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import lightning as L
import pandas as pd
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Pad
from tqdm import tqdm
from torchvision import transforms

from ..tasks import tasks
from .generator import Generator

def choice(a, size):
    return np.random.choice(a, size=(min(len(a), size)), replace=False)

def get_image_generator(cfg, spec, split):
    if cfg.dataset == 'mnist':
        data = get_mnist_categories(
            root=cfg.root, download=cfg.download, preprocess=cfg.download,
            transform=transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
            ])
        )
        return ImageCategoriesGenerator(
            dataset=data, spec=spec, split=split, n_training_imgs=cfg.n_training_imgs, n_training_samples=cfg.n_training_samples,
            n_training_samples_total=cfg.n_training_samples_total,
            n_validation_samples_1=cfg.n_validation_samples_1,
            n_validation_samples_2=cfg.n_validation_samples_2, seed=cfg.seed, distance=cfg.distance,
            separate_channels=cfg.separate_channels, permutation=cfg.permutation, permutation_seed=cfg.permutation_seed
        )
    if cfg.dataset == 'cifar10':
        transform = []
        if 'crop' in cfg.augmentation:
            transform.append(transforms.RandomCrop(32, padding=4))
        if 'flip' in cfg.augmentation:
            transform.append(transforms.RandomHorizontalFlip())
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        data = get_cifar10_categories(
            root=cfg.root, download=cfg.download, preprocess=cfg.download,
            transform=transforms.Compose(transform)
        )
        if len(cfg.augmentation) > 0:
            data_test = get_cifar10_categories(
                root=cfg.root, download=cfg.download, preprocess=cfg.download,
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            )
        else:
            data_test = None
        return ImageCategoriesGenerator(
            dataset=data, dataset_test=data_test, spec=spec, split=split, n_training_imgs=cfg.n_training_imgs, n_training_samples=cfg.n_training_samples,
            n_training_samples_total=cfg.n_training_samples_total,
            n_validation_samples_1=cfg.n_validation_samples_1,
            n_validation_samples_2=cfg.n_validation_samples_2, seed=cfg.seed, distance=cfg.distance,
            separate_channels=cfg.separate_channels, permutation=cfg.permutation, permutation_seed=cfg.permutation_seed
        )

class ImageCategoriesGenerator(Generator):
    def __init__(
            self, dataset: list[Dataset]|list[list[Dataset]], spec: np.ndarray, split: np.ndarray[bool], dataset_test: None|list[Dataset]|list[list[Dataset]]=None, comps: list[int]|None=None,
            n_training_imgs: int=100, n_training_samples: int=50, n_training_samples_total: int|None=None,
            n_validation_samples_1: int=100, n_validation_samples_2: int=100,
            seed:int|None=None, distance:int=28, separate_channels=False,
            permutation: str='none', permutation_seed: int|None=None, permutation_index: int|None=None
        ):
        self.separate_channels = separate_channels
        super().__init__(spec, comps, permutation=permutation, permutation_seed=permutation_seed, permutation_index=permutation_index)
        L.seed_everything(seed)
        if isinstance(dataset[0], Dataset):
            self.dataset = [dataset]*len(self.comps)
        else:
            self.dataset = dataset
        if dataset_test is None:
            self.dataset_test = self.dataset
        elif isinstance(dataset_test[0], Dataset):
            self.dataset_test = [dataset_test]*len(self.comps)
        else:
            self.dataset_test = dataset_test
        for comp, dataset_comp in zip(self.comps, self.dataset):
            if not isinstance(dataset_comp, (list,)):
                raise ValueError('You must provide a list of datasets.')
            if len(dataset_comp) < comp:
                raise ValueError('Dataset has too few possible categories')
        if n_training_samples_total is not None:
            n_training_samples = int(n_training_samples_total/sum(split))
        self.distance=distance
        gen_df = []
        img_perm = [
            [
                np.random.permutation(np.arange(len(_d)))\
                for _d in d
            ] for d in self.dataset
        ]
        data_lens = [
            [
                len(_d)\
                for _d in d
            ] for d in self.dataset
        ]
        for idx, (_x, train) in tqdm(enumerate(zip(self.spec_perm, split))):
            train_all_len = n_training_imgs**len(_x)
            _data_lens = [data_lens[comp][x_comp]-n_training_imgs for comp, x_comp in enumerate(_x)]
            train_none_len = np.prod(_data_lens)
            train_sample = np.random.choice(train_all_len, size=(n_training_samples+n_validation_samples_1,))
            val_1_sample = train_sample[n_training_samples:]
            train_sample = train_sample[:n_training_samples]
            val_2_sample = np.random.choice(train_none_len, size=(n_validation_samples_2,))
            cum_prod = np.cumprod([1]+[n_training_imgs]*len(_x))
            train_sample = np.array([img_perm[comp][x_comp][:n_training_imgs][np.mod(train_sample//cum_prod[comp], n_training_imgs)] for comp, x_comp in enumerate(_x)]).T
            val_1_sample = np.array([img_perm[comp][x_comp][:n_training_imgs][np.mod(val_1_sample//cum_prod[comp], n_training_imgs)] for comp, x_comp in enumerate(_x)]).T
            cum_prod = np.cumprod([1]+_data_lens)
            val_2_sample = np.array([img_perm[comp][x_comp][n_training_imgs:][np.mod(val_2_sample//cum_prod[comp], _data_lens[comp])] for comp, x_comp in enumerate(_x)]).T
            if train:
                new_df = pd.concat([
                    pd.DataFrame({
                        'arr': list(arr),
                        'split': split
                    }) for split, arr in zip(['train', 'val_1', 'val_2'], [train_sample, val_1_sample, val_2_sample])
                ])
            else:
                new_df = pd.concat([
                    pd.DataFrame({
                        'arr': list(arr),
                        'split': split
                    }) for split, arr in zip(['val_1', 'val_2'], [val_1_sample, val_2_sample])
                ])
            new_df['spec_idx'] = idx
            new_df['emb_idx'] = np.arange(len(new_df))
            gen_df.append(new_df)
        gen_df = pd.concat(gen_df).reset_index(drop=True)
        self.df = gen_df
    
    def __call__(self, spec: int|np.ndarray, emb: int=0, split: str='train'):
        if isinstance(spec, (np.ndarray)):
            spec = np.nonzero((self.spec_perm==spec.reshape(1,-1)).all(axis=-1))[0]
        spec_x = self.spec_perm[spec]
        row = self.df.set_index(keys=['spec_idx', 'emb_idx']).loc[spec,emb]
        dataset = self.dataset if split=='train' else self.dataset_test
        if self.separate_channels:
            img = []
        for comp, (_x_comp, emb_sel) in enumerate(zip(spec_x, row['arr'])):
            new_img = dataset[comp][_x_comp][emb_sel]
            new_img = Pad((comp*self.distance, 0, (len(self.comps)-1-comp)*self.distance,0), padding_mode='edge')(new_img)
            if self.separate_channels:
                img.append(new_img)
            else:
                if comp > 0:
                    img = torch.maximum(img, new_img)
                else:
                    img = new_img
        if self.separate_channels:
            img = torch.cat(img, dim=0)
        return img

class SubsetEntry(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], subset_idx:int=0) -> None:
        super().__init__(dataset, indices)
        self.subset_idx = subset_idx
    
    def __getitem__(self, idx):
        outp = super().__getitem__(idx)
        return outp[self.subset_idx]

def get_mnist_categories(root: str, preprocess_root: str=None, train: bool=True, transform: Optional[Callable]=None, download: bool=False, preprocess: bool=False):
    if preprocess_root is None:
        preprocess_root = root
    preprocess_path = Path(preprocess_root, 'MNIST/preprocessed', 'train_categories.npz' if train else 'test_categories.npz')
    mnist = MNIST(root=root, train=train, transform=transform, download=download)
    if preprocess_path.exists():
        categories = np.load(preprocess_path)
    else:
        if preprocess:
            categories = {f'cat_{i}': [] for i in range(10)}
            for i, (_, y) in enumerate(mnist):
                categories[f'cat_{y}'].append(i)
            Path(os.path.dirname(preprocess_path)).mkdir(exist_ok=True, parents=True)
            np.savez(preprocess_path, **categories)
        else:
            raise RuntimeError('Preprocessing not found. You can use "preprocess=True" to create it.')
    return [SubsetEntry(mnist, cat) for cat in categories.values()]

def get_cifar10_categories(root: str, preprocess_root: str=None, train: bool=True, transform: Optional[Callable]=None, download: bool=False, preprocess: bool=False):
    if preprocess_root is None:
        preprocess_root = root
    preprocess_path = Path(preprocess_root, 'CIFAR10/preprocessed', 'train_categories.npz' if train else 'test_categories.npz')
    cifar = CIFAR10(root=root, train=train, transform=transform, download=download)
    if preprocess_path.exists():
        categories = np.load(preprocess_path)
    else:
        if preprocess:
            categories = {f'cat_{i}': [] for i in range(10)}
            for i, (_, y) in enumerate(cifar):
                categories[f'cat_{y}'].append(i)
            Path(os.path.dirname(preprocess_path)).mkdir(exist_ok=True, parents=True)
            np.savez(preprocess_path, **categories)
        else:
            raise RuntimeError('Preprocessing not found. You can use "preprocess=True" to create it.')
    return [SubsetEntry(cifar, cat) for cat in categories.values()]
