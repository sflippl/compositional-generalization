from pathlib import Path
import os

import pandas as pd

class CSVDataFrame(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None, file_name=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.file_name = file_name
    
    def save(self, wd):
        assert self.file_name is not None
        path = Path(wd, f'{self.file_name}.csv')
        self.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    
    def set_epoch(self, epoch):
        self['epoch'] = epoch
