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

        self.H_xx, self.H_yy, self.H_zz, self.sx_list, self.sy_list, self.sz_list = MBLDatasetH.build_H_ii(self.L, self.periodic)

    def _get_image(self, idx, Ws):
        W = Ws[idx]
        H = self.build_H(self.L, W, self.J).toarray()
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
        if W <= 3.5:
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

    @staticmethod
    def dict_to_str(_dict):
        
        od = OrderedDict(sorted(_dict.items())) # Sort keys.
        s = json.dumps(od) # Turn dict into str.
        s = s[1:-1].replace('\"', '').replace(' ', '') # Replace some special characeters.
        s = ''.join(x if x.isalnum() else ('=' if x == ':' else '_') for x in s) # Replace all remaining special characters.

        return s

    @staticmethod
    def save_cache(obj, obj_name, obj_params, cache_dir='cache'):
        """Cache an object, together with the parameters used to generate it.
        For `obj_params`, try not to use nested dict or with complicated objects.

        Parameters
        ----------
        obj : object
            An `object` you want to cache.
        obj_name : str
            A unique name you give to this object.
        obj_params : dict
            A `dict` of all parameters necessary to generate this object.
        cache_dir : str, optional
            Directory where the cache is located.
        """

        param_str = MBLDatasetH.dict_to_str(obj_params)
        os.makedirs(os.path.join(cache_dir, obj_name), exist_ok=True)
        with gzip.open(os.path.join(cache_dir, obj_name, param_str + '.pkl.gz'), 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_cache(obj_name, obj_params, cache_dir='cache'):
        """Check if the object is cached. If not, return None.
        For `obj_params`, try not to use nested dict or with complicated objects.

        Parameters
        ----------
        obj_name : str
            A unique name you give to this object.
        obj_params : dict
            A `dict` of all parameters necessary to generate this object.
        cache_dir : str, optional
            Directory where the cache is located.
        """

        param_str = MBLDatasetH.dict_to_str(obj_params)
        os.makedirs(os.path.join(cache_dir, obj_name), exist_ok=True)
        if os.path.isfile(os.path.join(cache_dir, obj_name, param_str + '.pkl.gz')):
            with gzip.open(os.path.join(cache_dir, obj_name, param_str + '.pkl.gz'), 'rb') as handle:
                obj = pickle.load(handle)
                return obj
        else:
            return None

    @staticmethod
    def get_h(L, W):
        h = np.random.uniform(-W, W, L)
        return h

    @staticmethod
    def build_si_list(L):

        # Get single site operaors.
        sx = csr_matrix(np.array([[0.,  1. ], [1. ,  0.]]))
        sy = csr_matrix(np.array([[0., -1.j], [1.j,  0.]]))
        sz = csr_matrix(np.array([[1.,  0. ], [0. , -1.]]))
        id = csr_matrix(np.eye(2))

        # ========================================
        # Start cached area: si_list.
        # ========================================

        obj_params = {'L': L}

        sx_list = MBLDatasetH.load_cache('sx_list', obj_params)
        sy_list = MBLDatasetH.load_cache('sy_list', obj_params)
        sz_list = MBLDatasetH.load_cache('sz_list', obj_params)

        if sx_list is None or sy_list is None or sz_list is None:

            # print('Cache not found for `si_list`. Generate from scratch.')

            sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
            sy_list = []
            sz_list = []

            for i_site in range(L):

                x_ops = [id] * L
                y_ops = [id] * L
                z_ops = [id] * L
                x_ops[i_site] = sx
                y_ops[i_site] = sy
                z_ops[i_site] = sz

                X = x_ops[0]
                Y = y_ops[0]
                Z = z_ops[0]
                for j in range(1, L):
                    X = kron(X, x_ops[j], 'csr')
                    Y = kron(Y, y_ops[j], 'csr')
                    Z = kron(Z, z_ops[j], 'csr')
                sx_list.append(X)
                sy_list.append(Y)
                sz_list.append(Z)

            MBLDatasetH.save_cache(sx_list, 'sx_list', obj_params)
            MBLDatasetH.save_cache(sy_list, 'sy_list', obj_params)
            MBLDatasetH.save_cache(sz_list, 'sz_list', obj_params)

        # else:

        #     print('Cache found for `si_list`. Load from cache.')

        # ========================================
        # End cached area: si_list.
        # ========================================

        return sx_list, sy_list, sz_list

    @staticmethod
    def build_H_ii(L, periodic):

        sx_list, sy_list, sz_list = MBLDatasetH.build_si_list(L)

        # ========================================
        # Start cached area: H_ii.
        # ========================================
        
        obj_params = {'L': L, 'periodic': periodic}

        H_xx = MBLDatasetH.load_cache('H_xx', obj_params)
        H_yy = MBLDatasetH.load_cache('H_yy', obj_params)
        H_zz = MBLDatasetH.load_cache('H_zz', obj_params)

        if H_xx is None or H_yy is None or H_zz is None:

            # print('Cache not found for `H_ii`. Generate from scratch.')

            H_xx = csr_matrix((2**L, 2**L))
            H_yy = csr_matrix((2**L, 2**L))
            H_zz = csr_matrix((2**L, 2**L))

            for i in range(L if periodic else L - 1):
                H_xx = H_xx + sx_list[i] * sx_list[(i + 1) % L]
                H_yy = H_yy + sy_list[i] * sy_list[(i + 1) % L]
                H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]

            MBLDatasetH.save_cache(H_xx, 'H_xx', obj_params)
            MBLDatasetH.save_cache(H_yy, 'H_yy', obj_params)
            MBLDatasetH.save_cache(H_zz, 'H_zz', obj_params)

        # else:

        #     print('Cache found for `H_ii`. Load from cache.')

        # ========================================
        # End cached area: H_ii.
        # ========================================

        return H_xx, H_yy, H_zz, sx_list, sy_list, sz_list

    def build_H(self, L, W, J):

        H_xx, H_yy, H_zz, sx_list, sy_list, sz_list = self.H_xx, self.H_yy, self.H_zz, self.sx_list, self.sy_list, self.sz_list

        # H_z is not cached due to randomness.
        H_z  = csr_matrix((2**L, 2**L))
        h    = MBLDatasetH.get_h(L, W)

        for i in range(L):
            H_z = H_z + h[i] * sz_list[i]

        H = J * (H_xx + H_yy + H_zz) - H_z

        return H

