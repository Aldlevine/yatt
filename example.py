#!/usr/bin/env python
# coding: utf-8

# # PyTorch Yatt
# 
# Yatt is yet another PyTorch trainer.

# ## Imports

# In[1]:


import os

import numpy as np
import torch
from torch import Size, Tensor, nn
from torch.utils import data
from torchvision import datasets as vdata
from torchvision import transforms as vtransforms
from torchvision import utils as vutils

from yatt import DataLoaderConfig, HParams, Trainer


# ## Setup Architecture

# In[2]:


class AutoEncoder(nn.Module):

    def __init__(
        self,
        # in_channels: int,
        in_shape: tuple[int, int, int],
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        ch = [in_shape[0], *hidden_dims]
        hidden_size: np.ndarray = np.fromiter(in_shape[1:], object)
        hidden_size //= 2**(len(hidden_dims) - 1)
        hidden_shape = (hidden_dims[-1], *hidden_size)
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(ch[i],
                              ch[i + 1],
                              kernel_size=3,
                              stride=1 + (i > 0),
                              padding=1),
                    nn.BatchNorm2d(ch[i + 1]),
                    nn.SELU(),
                    nn.Conv2d(ch[i + 1],
                              ch[i + 1],
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(ch[i + 1]),
                    nn.SELU(),
                ) for i in range(len(ch) - 1)
            ],
            nn.Flatten(1),
            nn.Linear(int(np.prod(hidden_shape)), latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, int(np.prod(hidden_shape))),
            nn.Unflatten(1, Size(hidden_shape)),
            *[
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(ch[i],
                              ch[i - 1],
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(ch[i - 1]),
                    nn.SELU(),
                ) for i in range(len(ch) - 1, 0, -1)
            ],
            nn.Tanh(),
        )
        self.downsample = vtransforms.Resize(in_shape[-2:], interpolation=vtransforms.InterpolationMode.BICUBIC)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        xhat_large = self.decoder(z)
        xhat = self.downsample(xhat_large.detach())
        return xhat, xhat_large


# ## Setup Trainer

# In[3]:


class HP(HParams):
    img_shape: tuple[int, int, int]
    hidden_dims: list[int]
    latent_dim: int
    learning_rate: float
    batch_size: int
    num_workers: int = (os.cpu_count() or 0) // 2


class MyTrainer(Trainer[HP]):

    def configure_model(self) -> nn.Module:
        model = AutoEncoder(
            in_shape=self.hparams.img_shape,
            hidden_dims=self.hparams.hidden_dims,
            latent_dim=self.hparams.latent_dim,
        )

        self.downscale = vtransforms.Resize(self.hparams.img_shape[-2:])

        return model

    def configure_optimizer(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer

    def configure_data_loaders(self) -> DataLoaderConfig:
        transform = vtransforms.Compose([
            vtransforms.Resize([s * 2 for s in self.hparams.img_shape[-2:]]),
            vtransforms.CenterCrop([s * 2 for s in self.hparams.img_shape[-2:]]),
            vtransforms.ToTensor(),
            vtransforms.Normalize(0.5, 0.5),
        ])
        # train_ds = vdata.CIFAR10("../data", train=True, transform=transform, download=True)
        # train_ds, val_ds = data.random_split(train_ds, [0.9, 0.1])
        # test_ds = vdata.CIFAR10("../data", train=False, transform=transform, download=True)
        # train_ds = vdata.CelebA("../data", split="train", transform=transform, download=True)
        # val_ds = vdata.CelebA("../data", split="valid", transform=transform, download=True)
        # test_ds = vdata.CelebA("../data", split="test", transform=transform, download=True)
        train_ds = vdata.FGVCAircraft("../data", "train", transform=transform, download=True)
        val_ds = vdata.FGVCAircraft("../data", "val", transform=transform, download=True)
        test_ds = vdata.FGVCAircraft("../data", "test", transform=transform, download=True)

        train_dl = data.DataLoader(train_ds,
                                   shuffle=True,
                                   batch_size=self.hparams.batch_size,
                                   pin_memory=True,
                                   num_workers=self.hparams.num_workers,
                                   persistent_workers=self.hparams.num_workers > 0)
        val_dl = data.DataLoader(val_ds,
                                 batch_size=self.hparams.batch_size,
                                 pin_memory=True,
                                 num_workers=self.hparams.num_workers,
                                 persistent_workers=self.hparams.num_workers > 0)
        test_dl = data.DataLoader(test_ds,
                                 pin_memory=True,
                                 num_workers=self.hparams.num_workers,
                                 persistent_workers=self.hparams.num_workers > 0)
        return DataLoaderConfig(
            train=train_dl,
            val=val_dl,
            test=test_dl,
        )

    def get_loss(self, x: Tensor) -> Tensor:
        x_small = self.downscale(x)
        _, y_large = self.model(x_small)
        loss_large = torch.nn.functional.mse_loss(x, y_large)
        return loss_large
        # loss_small = torch.nn.functional.mse_loss(x_small, y)
        # return (loss_large + loss_small) / 2

    def train_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        return self.get_loss(batch[0])

    def val_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        return self.get_loss(batch[0])

    def test_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        return self.get_loss(batch[0])

    def train_epoch_end(self) -> None:
        pass

    def val_epoch_end(self) -> None:
        if self.data_loaders.val == None:
            return
        x = next(iter(self.data_loaders.val))[0][:8].to(self.device)
        x = self.downscale(x)
        y, y_large = self.model(x)
        grid = vutils.make_grid(torch.cat([x, y]), normalize=True)
        self.log_image("val/sample", grid, self.epoch)
        self.log_graph(x)


# In[4]:



# ## Execution

# In[5]:


hp = HP(
    img_shape=(3,64,64),
    hidden_dims=[16, 32, 64, 128],
    latent_dim=512,
    learning_rate=1e-3,
    batch_size=512,
)

trainer = MyTrainer(
    f"auto_encoder.img_shape={hp.img_shape}.latent_dim={hp.latent_dim}",
    save_best_count=5,
    max_epochs=1000,
    log_interval=200,
    device=torch.device("cuda"),
)

trainer.configure(hp)

# trainer.configure_checkpoint("runs/auto_encoder.img_shape=(3, 32, 32).latent_dim=512/2023-02-16@08:57:48/checkpoints/latest.loss=0.00773597089573741.epoch=334.ckpt")

trainer.train()
# print(trainer.model)

