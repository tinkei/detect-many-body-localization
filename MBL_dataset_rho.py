"""Dataset for Many-Body Localization"""

import numpy as np
import torch
from MBL_dataset_base import MBLDatasetBase
from file_io import *


class MBLDatasetRho(MBLDatasetBase):
    """Dataset for Many-Body Localization

    If in extended/ergodic phase, assign class label 0.
    If in localized phase, assign class label 1.
    """

    def __init__(self, MBL_params, train=True, transform=None, **kwargs):
        super().__init__(MBL_params, train, transform)

        obj_name = MBL_params['obj_name']
        L        = MBL_params['L']
        n        = MBL_params['n']
        periodic = MBL_params['periodic']
        num_EV   = MBL_params['num_EV']

        if train:
            data = load_rho_train( obj_name, L, n, periodic, num_EV)
        else:
            data = load_rho_random( obj_name, L, n, periodic, num_EV)
    
        # if len(data) <= 10000:
        #     raise RuntimeError('Insufficient data. Training data has length {} <= 10000.'.format(len(data)))

        self.MBL_params = MBL_params
        self.data = data
        self.transform = transform

    @staticmethod
    def _get_image(idx, data):
        img = data[idx][0]
        return np.expand_dims(img, axis=2).astype(np.float32)

    @staticmethod
    def _get_disorder(idx, data):
        W = data[idx][1]
        return W

    @staticmethod
    def _get_label(idx, data):
        W = data[idx][1]
        # If in extended/ergodic phase, assign class label 0.
        # If in localized phase, assign class label 1.
        if W <= 3.5:
            label = 0
        else:
            label = 1
        return label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self._get_image(idx, self.data)
        label = self._get_label(idx, self.data)
        W     = self._get_disorder(idx, self.data)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label, 'W': W}
