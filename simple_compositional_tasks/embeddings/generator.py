import numpy as np
import pandas as pd

from itertools import permutations, product

class Generator:
    def __init__(self, spec: np.ndarray, comps: list[int]|None=None,
                 permutation: str='none', permutation_seed: int|None=None, permutation_index: int|None=None):
        super().__init__()
        self.spec = spec
        if comps is None:
            if spec.ndim>1:
                self.comps = tuple(spec.max(axis=0)+1)
            else:
                self.compms = (spec.max()+1,)
        else:
            self.comps = comps
        self.permutation = self.get_permutation(permutation, permutation_seed, permutation_index)
        self.spec_perm = self.apply_permutation(self.spec)
        self.df = pd.DataFrame({
            'spec_idx': np.arange(len(self.spec)),
            'emb_idx': 0
        }).set_index(['spec_idx', 'emb_idx'])
    
    def apply_permutation(self, spec):
        if self.permutation is None:
            return spec
        else:
            if isinstance(spec, np.ndarray) and (spec.ndim==2):
                return np.stack([self.apply_permutation(_spec) for _spec in spec])
            return [perm[_spec] for perm, _spec in zip(self.permutation, spec)]
    
    def get_permutation(self, permutation, seed, index):
        if permutation == 'none':
            return None
        if permutation == 'random':
            rng = np.random.default_rng(seed=seed)
            return [
                rng.permutation(n_comp) for n_comp in self.comps
            ]
        if permutation == 'combinatorial':
            perms = [list(permutations(range(n_comp))) for n_comp in self.comps]
            all_perms = list(product(*perms))
            return all_perms[index]

    def __call__(self, spec: int|np.ndarray, emb: int=0, split: str='train'):
        if isinstance(spec, (np.ndarray)):
            spec = np.nonzero((self.spec==spec.reshape(1,-1)).all(axis=-1))[0]
        spec_x = self.spec_perm[spec]
        if emb > 0:
            raise ValueError('There is only one possible embedding.')
        return spec_x
