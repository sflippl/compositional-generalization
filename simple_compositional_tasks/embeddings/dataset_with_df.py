from torch.utils.data import Dataset

class DatasetWithDf(Dataset):
    def __init__(self, data, df, row_fun=None):
        super().__init__()
        self.data = data
        self.df = df
        self.row_fun = row_fun or (lambda x: x)
    
    def __getitem__(self, index):
        index = self.df.index[index]
        return self.data(*index), self.row_fun(self.df.loc[index])
