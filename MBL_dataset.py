"""Dataset for Many-Body Localization"""

import numpy as np
import torch


class MBLDataset():
    """Dataset for Many-Body Localization

    If in extended/ergodic phase, assign class label 0.
    If in localized phase, assign class label 1.
    """

    def __init__(self, data, train=True, transform=None, **kwargs):
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
