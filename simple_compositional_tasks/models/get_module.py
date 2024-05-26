import os

import numpy as np
import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import hydra
import torchmetrics as tm

from .networks.densenet import get_dense_net
from .networks.convnet import get_conv_net
from .networks.resnet import get_resnet
from .networks.vision_transformer import get_vit
from .kernel_machines import get_kernel_machine

def get_module(cfg, analysis, metrics, inp_dim, outp_dim, flatten, val_splits, test_input=None, comps=None):
    if cfg.model_type=='lightning':
        if cfg.outp_dim != 'automatic':
            outp_dim = int(outp_dim)
            flatten = False
        model = {
            'densenet': get_dense_net,
            'convnet': get_conv_net,
            'resnet': get_resnet,
            'vit': get_vit
        }[cfg.network.network_type](inp_dim, outp_dim, cfg.network, test_input=test_input)
        module = Module(model, cfg.trainer_config, analysis, metrics=metrics, flatten=flatten, val_splits=val_splits)
        return LightningTrainer(module, cfg.trainer_config)
    if cfg.model_type=='kernel_machine':
        return get_kernel_machine(cfg=cfg, analysis=analysis, comps=comps, metrics=metrics)

class BinaryAccuracy(tm.classification.BinaryAccuracy):
    def __init__(self, threshold=0.5, multidim_average='global', ignore_index=None, validate_args=True, **kwargs):
        super().__init__(
            threshold=threshold,
            multidim_average=multidim_average, ignore_index=ignore_index, validate_args=validate_args, **kwargs
        )

    def forward(self, preds, target):
        preds = (preds.sign()+1)/2
        target = (target+1)/2
        outp = super().forward(preds, target)
        return outp

class Margin(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        #preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.total = self.total + torch.sum(preds*target)
        self.n = self.n + target.numel()

    def compute(self):
        return self.total.float() / self.n

def cross_entropy(yhat, y):
    return torch.log(torch.exp(-yhat*y)+1).mean()

metrics_dct = {
    'binary_accuracy': BinaryAccuracy,
    'mse': tm.MeanSquaredError,
    'margin': Margin,
    'accuracy': tm.Accuracy
}

def get_metrics(metrics, device):
    if metrics is None:
        return []
    return {metric: metrics_dct[metric]().to(device) for metric in metrics}

class LightningTrainer:
    def __init__(self, module, cfg):
        super().__init__()
        self.module = module
        self.cfg = cfg
    
    def fit(self, train_data, val_data):
        x, y = train_data[0]
        if isinstance(x, (np.ndarray,)):
            train_data.input_transform = (lambda x: torch.from_numpy(x).float())
            val_data.input_transform = (lambda x: torch.from_numpy(x).float())
        if isinstance(y, (np.float64)):
            train_data.output_transform = (lambda x: x.astype(np.float32))
            val_data.output_transform = (lambda x: x.astype(np.float32))
        wd = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        L.seed_everything(self.cfg.training_seed)
        logger = loggers.CSVLogger(save_dir=wd, name='logs')
        callbacks = []
        if self.cfg.early_stopping:
            callbacks.append(
                L.pytorch.callbacks.EarlyStopping(
                    'train_loss',
                    strict=False,
                    stopping_threshold=self.cfg.early_stopping_stopping_threshold,
                    divergence_threshold=self.cfg.early_stopping_divergence_threshold,
                    verbose=True,
                    patience=1e5
                )
            )
        trainer = L.Trainer(
            accelerator=self.cfg.accelerator,
            devices=self.cfg.devices,
            logger=logger,
            max_epochs=self.cfg.max_epochs,
            log_every_n_steps=self.cfg.log_every_n_steps,
            check_val_every_n_epoch=self.cfg.check_val_every_n_epoch,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            inference_mode=self.cfg.inference_mode
        )
        if self.cfg.batch_training:
            train_loader = data.DataLoader(
                train_data, batch_size=self.cfg.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers
            )
        else:
            train_loader = data.DataLoader(
                train_data, batch_size=len(train_data), num_workers=self.cfg.num_workers
            )
        if self.cfg.batch_testing:
            val_loader = data.DataLoader(
                val_data, batch_size=self.cfg.test_batch_size, shuffle=False, num_workers=self.cfg.num_workers
            )
        else:
            val_loader = data.DataLoader(
                val_data, batch_size=len(val_data), num_workers=self.cfg.num_workers
            )
        if self.cfg.auto_lr_find:
            tuner = Tuner(trainer)
            tuner.lr_find(self.module, train_loader, num_training=self.cfg.auto_lr_find_num_steps, max_lr=self.cfg.lr)
        self.module.overwrite_analyse = True
        trainer.validate(self.module, val_loader)
        self.module.overwrite_analyse = False
        trainer.fit(self.module, train_loader, val_loader)
        self.module.overwrite_analyse = True
        trainer.validate(self.module, val_loader)

class Module(L.LightningModule):
    def __init__(self, model, cfg, analysis, metrics=None, flatten=False, val_splits=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg.lr
        self.criterion = {
            'mse': F.mse_loss,
            'crossentropy': cross_entropy
        }[cfg.criterion]
        self.analysis = analysis
        self.flatten = flatten
        self.overwrite_analyse = True
        self.metrics = {
            split: get_metrics(metrics, self.device) for split in val_splits
        }

    def forward(self, x):
        outp = self.model(x)
        if self.flatten:
            outp = outp.squeeze(-1)
        return outp

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y), row = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log('val_loss', loss)
        if self.analyse_on_this_epoch():
            self.analysis.validation_step(self, x, y, yhat, row)
        for i, metrics in (self.metrics.items()):
            for metric in metrics.values():
                if (np.array(row['split'])==i).any():
                    metric(yhat[np.array(row['split'])==i].cpu(), y[np.array(row['split'])==i].cpu())
        return yhat
    
    def analyse_on_this_epoch(self):
        if self.overwrite_analyse:
            return True
        if self.cfg.analyse_on_epochs is None:
            if self.cfg.analyse_every_n_epoch == -1:
                outp = False
            else:
                outp = self.current_epoch % self.cfg.analyse_every_n_epoch == 0
            if self.cfg.analyse_first_n_epochs is not None:
                outp = outp | (self.current_epoch <= self.cfg.analyse_first_n_epochs)
            return outp
        return self.current_epoch in self.cfg.analyse_on_epochs
    
    def on_validation_epoch_end(self):
        if self.analyse_on_this_epoch():
            wd = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            objs = self.analysis.on_validation_epoch_end()
            for obj in objs:
                obj.set_epoch(self.current_epoch)
                obj.save(wd)
                #df.to_csv(Path(wd, f'{name}.csv'), mode='a', header=not os.path.exists(Path(wd, f'{name}.csv')), index=False)
        for split_name, metrics in self.metrics.items():
            for metric_name, metric in metrics.items():
                self.log(f'{split_name}/{metric_name}', metric.compute())
                metric.reset()
    
    def configure_optimizers(self):
        if self.cfg.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.cfg.momentum)
        if self.cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if self.cfg.scheduler == 'none':
            return optimizer
        if self.cfg.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.max_epochs)
            return [optimizer], [scheduler]
