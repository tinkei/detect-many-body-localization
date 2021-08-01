"""Dataset for Many-Body Localization"""

import torch
import os
import gzip
import json
import pickle
import numpy as np
from collections import OrderedDict
from scipy.sparse import csr_matrix, kron
from MBL_dataset_base import MBLDatasetBase
from file_io import *
from data_gen import *


class MBLDatasetH(MBLDatasetBase):
    """Dataset for Many-Body Localization

    If in extended/ergodic phase, assign class label 0.
    If in localized phase, assign class label 1.
    """

    def __init__(self, MBL_params, train=True, transform=None, **kwargs):
        super().__init__(MBL_params, train, transform)

        obj_name = MBL_params['obj_name']
        L        = MBL_params['L']
        periodic = MBL_params['periodic']

        self.MBL_params = MBL_params
        self.transform = transform
        self.J = 1
        self.L = MBL_params['L']
        self.periodic = MBL_params['periodic']
        if train:
            self.Ws = MBL_params['Ws_train'] # Contains the disorder strength W used to generate each sample.
        else:
            self.Ws = MBL_params['Ws_valid']

    def _get_image(self, idx, Ws):
        W = Ws[idx]
        H = build_H(self.L, W, self.J, self.periodic).toarray()
        H_real = np.real(H)
        H_imag = np.imag(H)
        img = np.stack((H_real, H_imag), axis=2)
        # print(H_real.shape, H_imag.shape)
        # print(img.shape)
        return img.astype(np.float32)
        # return np.expand_dims(img, axis=2).astype(np.float32)

    @staticmethod
    def _get_disorder(idx, Ws):
        W = Ws[idx]
        return W

    @staticmethod
    def _get_label(idx, Ws):
        W = Ws[idx]
        # If in extended/ergodic phase, assign class label 0.
        # If in localized phase, assign class label 1.
        if W <= 3.0:
            label = 0
        else:
            label = 1
        return label

    def __len__(self):
        return len(self.Ws)

    def __getitem__(self, idx):
        image = self._get_image(idx, self.Ws)
        label = self._get_label(idx, self.Ws)
        W     = self._get_disorder(idx, self.Ws)
        if self.transform:
            # print(image.shape)
            image = self.transform(image)
            # print(image.shape) # Why is the axis swapped?
        return {'image': image, 'label': label, 'W': W}

