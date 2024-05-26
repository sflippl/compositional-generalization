from itertools import chain, combinations, product
import argparse
import abc

import numpy as np
import pandas as pd

from .generics import choose_default, arg_wrapper

class Generator:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    
    @classmethod
    def add_generator_args(cls, parser, prefix, defaults):
        return parser

def one_hot(i,n):
    rtn = np.zeros(n,)
    rtn[i] = 1.
    return rtn

class OnehotGenerator(Generator):
    def __call__(self, x: np.ndarray, comps=None) -> np.ndarray:
        if comps is None:
            comps = tuple(x.max(axis=0)+1)
        return np.stack([
            np.concatenate([
                one_hot(i, n) for i, n in zip(_x, comps)
            ]) for _x in x
        ])

class AllConjunctionsGenerator(Generator):
    def __init__(self, max_conjunctions: int|None = None):
        super().__init__()
        self.max_conjunctions = max_conjunctions

    def __call__(self, x: np.ndarray, comps=None) -> np.ndarray:
        if comps is None:
            comps = tuple(x.max(axis=0)+1)
        if self.max_conjunctions is None:
            max_r = len(comps)
        else:
            max_r = self.max_conjunctions
        return np.stack([
            np.concatenate([
                self.conjunction_onehot(conj, x_i, lens) for conj, lens in self.get_subsets(comps, min_r=1, max_r=max_r)
            ]) for x_i in x
        ])
    
    def conjunction_onehot(self, conj: tuple, x:np.ndarray, lens: tuple):
        i = (x[list(conj)]*np.concatenate([[1.], np.cumprod(lens)[:(-1)]])).sum().astype(int)
        n = np.prod(lens)
        return one_hot(i, n)
    
    def conjunction_onehot_labels(self, conj: tuple, comps: tuple):
        lens = np.array(comps)[list(conj)]
        outp = np.array([['all']*len(comps)]*np.prod(lens))
        for x_i in product(*[range(_len) for _len in [comps[i] for i in conj]]):
            i = (np.array(x_i)*np.concatenate([[1.], np.cumprod(lens)[:(-1)]])).sum().astype(int)
            outp[i, list(conj)] = x_i
        return pd.DataFrame(outp, columns=[f'comp_{i}' for i in range(len(comps))])
    
    def get_subsets(self, comps, min_r=1, max_r=None):
        max_r = max_r or len(comps)
        subsets = list(chain.from_iterable(combinations(enumerate(comps), r) for r in range(min_r, max_r+1)))
        return [list(zip(*sub)) for sub in subsets]
    
    def get_labels(self, x: np.ndarray, comps=None) -> np.ndarray:
        if comps is None:
            comps = tuple(x.max(axis=0)+1)
        if self.max_conjunctions is None:
            max_r = len(comps)
        else:
            max_r = self.max_conjunctions
        subsets = self.get_subsets(comps, min_r=1, max_r=max_r)
        df = pd.concat([
            self.conjunction_onehot_labels(sub[0], comps) for sub in subsets
        ])
        return df
    
    @classmethod
    def add_generator_args(cls, parser, prefix, defaults):
        parser.add_argument(
            f'{prefix}max_conjunctions', type=int,
            default=choose_default(None, 'max_conjunctions', defaults)
        )
        return parser

generator_dct = {
    'identity': Generator,
    'one_hot': OnehotGenerator,
    'all_conjunctions': AllConjunctionsGenerator
}

def add_generator_args(parser, prefix='', defaults={}):
    parser.add_argument(
        f'{prefix}generator', choices=list(generator_dct.keys()),
        default=choose_default('one_hot', 'generator', defaults)
    )
    for generator in generator_dct.values():
        parser = generator.add_generator_args(parser, prefix, defaults)

def get_generator_from_args(args, prefix):
    if isinstance(args, (argparse.Namespace)):
        args = vars(args)
    args = {
        key[len(prefix):]: value for key, value in args.items() if key.startswith(prefix)
    }
    return arg_wrapper(generator_dct[args['generator']], args)
