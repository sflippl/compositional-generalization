import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import lightning

class Trainer(lightning.Trainer):
    def __init__(self, cfg):
        super().__init__(
            
        )

class GradientDescent:
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = {
            'mse': F.mse_loss,
            'crossentropy': F.cross_entropy
        }[cfg.criterion]
    
    def fit(self, data, df, analysis):
        optimizer = optim.GradientDescent(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)
        train_data = data[df.train]
        if self.cfg.batch_training:
            data_loader = data.DataLoader(train_data, batch_size=self.cfg.batch_size, shuffle=True)
        else:
            data_loader = [train_data]
        if self.cfg.batch_testing:
            data_loader = data.DataLoader(data, batch_size=self.cfg.test_batch_size, shuffle=False)
        else:
            data_loader = [data]
        for i in tqdm(range(self.cfg.epochs)):
            for x, y in data_loader:
                optimizer.zero_grad()
                yhat = self.model(x)
                loss = self.criterion(yhat, y)
                loss.backward()
                optimizer.step()
            
