import torch
import torch.nn as nn
import lightning as L

from .get_ntk import get_ntk

def get_conv_net(inp_dim, outp_dim, cfg, test_input):
    return ConvNet(
        cfg.conv_layers, cfg.dense_layers, in_channels=test_input.shape[0], outp_dim=outp_dim, kernel_size=cfg.kernel_size,
        pool_size=cfg.pool_size, test_input=test_input, seed=cfg.seed
    )

class ConvNet(nn.Sequential):
    def __init__(self, conv_layers, dense_layers, in_channels=1, outp_dim=1, kernel_size=5, pool_size=2, test_input=None, flattened_dim=None, seed=None):
        L.seed_everything(seed)
        super().__init__()
        for filters_in, filters_out in zip([in_channels]+conv_layers[:(-1)], conv_layers):
            conv = nn.Conv2d(
                in_channels=filters_in, out_channels=filters_out, kernel_size=(kernel_size,kernel_size)
            )
            nn.init.kaiming_normal_(conv.weight)
            self.append(conv)
            self.append(
                nn.ReLU()
            )
            self.append(nn.BatchNorm2d(filters_out))
        self.append(nn.MaxPool2d(kernel_size=(pool_size, pool_size), stride=2))
        self.append(nn.Flatten())
        if flattened_dim is None:
            with torch.no_grad():
                flattened_dim = super().forward(test_input.unsqueeze(0)).shape[-1]
        for i, (hdims_in, hdims_out) in enumerate(zip([flattened_dim]+dense_layers, dense_layers+[outp_dim])):
            if i>0:
                self.append(nn.ReLU())
            linear = nn.Linear(hdims_in, hdims_out)
            nn.init.kaiming_normal_(linear.weight)
            self.append(linear)
    
    def forward(self, x):
        outp = super().forward(x)
        return outp.squeeze(-1)

    @torch.no_grad()
    def get_features(self, x):
        n = x.shape[0]
        lst = [x.reshape(n, -1)]
        for feat in self:
            x = feat(x)
            lst.append(x.reshape(n, -1))
        return lst
    
    @torch.enable_grad()
    def get_ntk(self, x):
        return get_ntk(self, x)
