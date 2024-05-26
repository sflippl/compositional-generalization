from itertools import chain, combinations, product

import numpy as np
import pandas as pd

from .generator import Generator

def one_hot(i,n):
    rtn = np.zeros(n,)
    rtn[i] = 1.
    return rtn

class AllConjunctionsGenerator(Generator):
    def __init__(self, spec: np.ndarray, comps: list[int]|None=None, min_conjunctions: int=1, max_conjunctions: int|None = None,
                 permutation: str='none', permutation_seed: int|None=None, permutation_index: int|None=None):
        super().__init__(spec, comps=comps, permutation=permutation, permutation_seed=permutation_seed, permutation_index=permutation_index)
        self.max_conjunctions = max_conjunctions or len(self.comps)
        self.min_conjunctions = min_conjunctions

    def __call__(self, spec: int|np.ndarray, emb: int=0, split:str='train'):
        if spec.ndim == 2:
            return np.stack([self(x_i, emb) for x_i in spec])
        if isinstance(spec, (int,np.int64,)):
            spec = self.spec_perm[spec]
        else:
            spec = self.apply_permutation(spec)
        if emb > 0:
            raise ValueError('There is only one possible embedding.')
        return np.concatenate([
            self.conjunction_onehot(conj, spec, lens)\
                for conj, lens in self.get_subsets(self.comps, min_r=self.min_conjunctions, max_r=self.max_conjunctions)
        ])
    
    def conjunction_onehot(self, conj: tuple, x: np.ndarray, lens: tuple):
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
    
    def get_labels(self, x: np.ndarray, comps=None, add_intercept=True) -> np.ndarray:
        if comps is None:
            comps = tuple(x.max(axis=0)+1)
        if self.max_conjunctions is None:
            max_r = len(comps)
        else:
            max_r = self.max_conjunctions
        subsets = self.get_subsets(comps, min_r=1, max_r=max_r)
        df = pd.concat([
            self.conjunction_onehot_labels(sub[0], comps) for sub in subsets
        ] + [pd.DataFrame({f'comp_{i}': ['all'] for i in range(len(comps))})]).reset_index(drop=True)
        return df

class OnehotGenerator(AllConjunctionsGenerator):
    def __init__(self, spec: np.ndarray,
                 permutation: str='none', permutation_seed: int|None=None, permutation_index: int|None=None):
        super().__init__(spec=spec, min_conjunctions=1, max_conjunctions=1, permutation=permutation, permutation_seed=permutation_seed, permutation_index=permutation_index)

generator_dct = {
    'identity': Generator,
    'one_hot': OnehotGenerator,
    'all_conjunctions': AllConjunctionsGenerator
}
