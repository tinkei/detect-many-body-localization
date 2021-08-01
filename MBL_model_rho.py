"""Model for Many Body Localization"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import gzip
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from MBL_dataset_rho import MBLDatasetRho
from MBL_model_base import MBLModelBase
from file_io import *


class MBLModelRho(MBLModelBase):
    """Model for Many Body Localization"""

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        """
        super().__init__(hparams)

    def prepare_data(self):

        train_dataset = MBLDatasetRho(
            MBL_params=self.hparams['MBL'],
            train=True,
            transform=transforms.ToTensor(),
        )
        valid_dataset = MBLDatasetRho(
            MBL_params=self.hparams['MBL'],
            train=False,
            transform=transforms.ToTensor(),
        )
        print("Number of training samples:", len(train_dataset))
        print("Number of validation samples:", len(valid_dataset))

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"] = train_dataset, valid_dataset
