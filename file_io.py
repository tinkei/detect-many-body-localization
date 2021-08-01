import os
import sys
import copy
import json
import gzip
import lzma
import pytz
import time
import pickle
import numpy as np
from tqdm import tqdm
from numba import jit, njit
from datetime import datetime

tz = pytz.timezone('Europe/Berlin')

import scipy
import scipy.linalg
import scipy.sparse.linalg

from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from scipy.optimize import curve_fit

from collections import OrderedDict



# running_on_colab = False
# os.environ['running_on_colab'] = str(running_on_colab)
running_on_colab = (os.getenv('running_on_colab', 'False') == 'True')

if running_on_colab:
    data_root             = 'drive/MyDrive/Colab Data/MBL/'
    sys.path.append(data_root)
else:
    data_root             = './'

signal_dir             = data_root
ED_data_dir            = data_root + 'ED_data'
rho_train_data_dir     = data_root + 'rho_train_data'
rho_random_data_dir    = data_root + 'rho_random_data'
EVW_train_data_dir     = data_root + 'EVW_train_data'
EVW_random_data_dir    = data_root + 'EVW_random_data'
model_dir              = data_root + 'models'
eval_random_data_dir   = data_root + 'eval_random_data'
H_model_dir            = data_root + 'H_models'
H_eval_random_data_dir = data_root + 'H_eval_random_data'




# ============================================================
# Pytorch imports.
# ============================================================

# pip install torch==1.4.0+cu92 torchsummary==1.5.1 torchvision==0.5.0+cu92 pytorch-lightning==0.7.6  -f https://download.pytorch.org/whl/torch_stable.html
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# from torchsummary import summary
# help(summary)

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('PyTorch device:', device)

from MBL_dataset import MBLDataset
from MBL_model import MBLModel
from MBL_H_dataset import MBLHDataset
from MBL_H_model import MBLHModel
model_version = 1



# ============================================================
# Helper functions.
# ============================================================

def dt():
    return datetime.now(tz=tz).strftime('%Y-%m-%d %H:%M:%S')

def check_shutdown_signal(signal_dir=signal_dir):
    """To gracefully stop generating data by making sure a loop is completed, this function will read a text file in a directory for the value `1`.
    
    Return
    ------
    shutdown : bool
        Shutdown signal detected.
    """

    os.makedirs(os.path.join(signal_dir), exist_ok=True)
    if os.path.isfile(os.path.join(signal_dir, 'shutdown_signal.txt')):
        with open(os.path.join(signal_dir, 'shutdown_signal.txt')) as f:
            lines = f.readlines()
        if lines is not None and len(lines) > 0:
            lines = [x.strip() for x in lines]
            if lines[0] == '1':
                return True

    return False



# ============================================================
# Cache.
# ============================================================

def dict_to_str(_dict):
    
    od = OrderedDict(sorted(_dict.items())) # Sort keys.
    s = json.dumps(od) # Turn dict into str.
    s = s[1:-1].replace('\"', '').replace(' ', '') # Replace some special characeters.
    s = ''.join(x if x.isalnum() else ('=' if x == ':' else '_') for x in s) # Replace all remaining special characters.

    return s

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

    param_str = dict_to_str(obj_params)
    os.makedirs(os.path.join(cache_dir, obj_name), exist_ok=True)
    with gzip.open(os.path.join(cache_dir, obj_name, param_str + '.pkl.gz'), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    param_str = dict_to_str(obj_params)
    os.makedirs(os.path.join(cache_dir, obj_name), exist_ok=True)
    if os.path.isfile(os.path.join(cache_dir, obj_name, param_str + '.pkl.gz')):
        with gzip.open(os.path.join(cache_dir, obj_name, param_str + '.pkl.gz'), 'rb') as handle:
            obj = pickle.load(handle)
            return obj
    else:
        return None



# ============================================================
# Store Exact Diagonalization result directly. Unused.
# ============================================================

def save_ED(obj, obj_name, L, W, periodic, data_dir=ED_data_dir):
    """Save a list of exact diagonalization results, organized by the parameters used to generate them.
    For `obj_params`, try not to use nested dict or with complicated objects.

    Parameters
    ----------
    obj : list
        A list of lists, where each reduced density matrix is paired with its disorder strength W.
        Number of data must be a multiple of 10.
    obj_name : str
        A unique name you give to this object. Call it `rho_A`.
    L : int
        System size.
    W : float
        Disorder strength.
    periodic : bool
        Whether the Hamiltonian is periodic.
    data_dir : str, optional
        Directory where the data is saved.
    """

    directory = os.path.join(data_dir, obj_name, 'L={:02d}'.format(L), 'W={:.2f}'.format(W), 'periodic={}'.format(periodic))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, and increment suffix.
    i = 0
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        i += 1

    with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_ED(obj_name, L, W, periodic, data_dir=ED_data_dir):
    """Check if the object is cached. If not, return None.
    For `obj_params`, try not to use nested dict or with complicated objects.

    Parameters
    ----------
    obj_name : str
        A unique name you give to this object. Call it `rho_A`.
    L : int
        System size.
    W : float
        Disorder strength.
    periodic : bool
        Whether the Hamiltonian is periodic.
    data_dir : str, optional
        Directory where the data is saved.
    """

    raise NotImplementedError('Function not implemented.')

    param_str = dict_to_str(obj_params)
    os.makedirs(os.path.join(data_dir, obj_name, 'L={:2d}'.format(L), 'W={:.2d}'.format(W)), exist_ok=True)
    if os.path.isfile(os.path.join(data_dir, obj_name, 'L={:2d}'.format(L), 'W={:.2d}'.format(W), param_str + '.pkl.gz')):
        with gzip.open(os.path.join(data_dir, obj_name, 'L={:2d}'.format(L), 'W={:.2d}'.format(W), param_str + '.pkl.gz'), 'rb') as handle:
            obj = pickle.load(handle)
            return obj
    else:
        return None



# ============================================================
# Store Reduced Density Matrix.
# ============================================================

def save_rho_train( obj, obj_name, L, n, periodic, num_EV, data_dir=rho_train_data_dir):
    """Save a list of reduced density matrices with W = {0.5, 8}.

    Parameters
    ----------
    obj : list
        A list of lists, where each reduced density matrix is paired with its disorder strength W.
        i.e. obj[i][0] is a 2D numpy.ndarray of the reduced density matrix, and obj[i][1] is the disorder strength used to generate it.
        Number of data must be a multiple of 10.
    obj_name : str
        A name you give to this object. Call it `rho_A`.
    L : int
        System size.
    n : int
        Number of consecutive spins sampled.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, and increment suffix.
    i = 0
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        i += 1

    with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_rho_random(obj, obj_name, L, n, periodic, num_EV, data_dir=rho_random_data_dir):
    """Save a list of reduced density matrices with random W != {0.5, 8}.

    Parameters
    ----------
    obj : list
        A list of lists, where each reduced density matrix is paired with its disorder strength W.
        i.e. obj[i][0] is a 2D numpy.ndarray of the reduced density matrix, and obj[i][1] is the disorder strength used to generate it.
        Number of data must be a multiple of 10.
    obj_name : str
        A name you give to this object. Call it `rho_A`.
    L : int
        System size.
    n : int
        Number of consecutive spins sampled.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, and increment suffix.
    i = 0
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        i += 1

    with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_rho_train( obj_name, L, n, periodic, num_EV, data_dir=rho_train_data_dir):
    """Load a list of reduced density matrices with W = {0.5, 8}.

    Parameters
    ----------
    obj_name : str
        A name you give to this object. Call it `rho_A`.
    L : int
        System size.
    n : int
        Number of consecutive spins sampled.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.

    Return
    ------
    obj : list
        A list of lists, where each reduced density matrix is paired with its disorder strength W.
        i.e. obj[i][0] is a 2D numpy.ndarray of the reduced density matrix, and obj[i][1] is the disorder strength used to generate it.
        Number of data must be a multiple of 10.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, load the file, and increment suffix.
    i = 0
    data = []
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'rb') as handle:
            data = data + pickle.load(handle)
        i += 1

    return data

def load_rho_random(obj_name, L, n, periodic, num_EV, data_dir=rho_random_data_dir):
    """Load a list of reduced density matrices with random W != {0.5, 8}.

    Parameters
    ----------
    obj_name : str
        A name you give to this object. Call it `rho_A`.
    L : int
        System size.
    n : int
        Number of consecutive spins sampled.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.

    Return
    ------
    obj : list
        A list of lists, where each reduced density matrix is paired with its disorder strength W.
        i.e. obj[i][0] is a 2D numpy.ndarray of the reduced density matrix, and obj[i][1] is the disorder strength used to generate it.
        Number of data must be a multiple of 10.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, load the file, and increment suffix.
    i = 0
    data = []
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'rb') as handle:
            data = data + pickle.load(handle)
        i += 1

    return data



# ============================================================
# Store tuple (eigenvalue, eigenvector, disorder strength).
# ============================================================

def save_EVW_train( obj, obj_name, L, periodic, num_EV, data_dir=EVW_train_data_dir):
    """Save a list of (eigenvalue, eigenvector, disorder strength) tuples with W = {0.5, 8}.

    Parameters
    ----------
    obj : list
        A list of (eigenvalue, eigenvector, disorder strength) tuples.
        i.e. obj[i][0] is an eigenvalue, obj[i][1] is a 1D numpy.ndarray of the corresponding eigenvector, and obj[i][2] is the disorder strength used to generate it.
    obj_name : str
        A name you give to this object. Call it `EVW`.
    L : int
        System size.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, and increment suffix.
    i = 0
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        i += 1

    with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_EVW_random(obj, obj_name, L, periodic, num_EV, data_dir=EVW_random_data_dir):
    """Save a list of (eigenvalue, eigenvector, disorder strength) tuples with random W != {0.5, 8}.

    Parameters
    ----------
    obj : list
        A list of (eigenvalue, eigenvector, disorder strength) tuples.
        i.e. obj[i][0] is an eigenvalue, obj[i][1] is a 1D numpy.ndarray of the corresponding eigenvector, and obj[i][2] is the disorder strength used to generate it.
    obj_name : str
        A name you give to this object. Call it `EVW`.
    L : int
        System size.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, and increment suffix.
    i = 0
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        i += 1

    with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_EVW_train( obj_name, L, periodic, num_EV, data_dir=EVW_train_data_dir):
    """Load a list of (eigenvalue, eigenvector, disorder strength) tuples with W = {0.5, 8}.

    Parameters
    ----------
    obj_name : str
        A name you give to this object. Call it `EVW`.
    L : int
        System size.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.

    Return
    ------
    obj : list
        A list of (eigenvalue, eigenvector, disorder strength) tuples.
        i.e. obj[i][0] is an eigenvalue, obj[i][1] is a 1D numpy.ndarray of the corresponding eigenvector, and obj[i][2] is the disorder strength used to generate it.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, load the file, and increment suffix.
    i = 0
    data = []
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'rb') as handle:
            data = data + pickle.load(handle)
        i += 1

    return data

def load_EVW_random(obj_name, L, periodic, num_EV, data_dir=EVW_random_data_dir):
    """Load a list of (eigenvalue, eigenvector, disorder strength) tuples with random W != {0.5, 8}.

    Parameters
    ----------
    obj_name : str
        A name you give to this object. Call it `EVW`.
    L : int
        System size.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.

    Return
    ------
    obj : list
        A list of (eigenvalue, eigenvector, disorder strength) tuples.
        i.e. obj[i][0] is an eigenvalue, obj[i][1] is a 1D numpy.ndarray of the corresponding eigenvector, and obj[i][2] is the disorder strength used to generate it.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    # Check if file exists, load the file, and increment suffix.
    i = 0
    data = []
    while os.path.exists(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i))):
        with gzip.open(os.path.join(directory, obj_name + '-{:09d}.pkl.gz'.format(i)), 'rb') as handle:
            data = data + pickle.load(handle)
        i += 1

    return data



# ============================================================
# Store CNN classifier.
# ============================================================

def save_model(model, file_name, L, n, periodic, num_EV, directory=model_dir):
    """Save model as pickle"""

    model = model.cpu()
    model_dict = {
        "state_dict": model.state_dict(),
        "hparams": model.hparams
    }

    directory = os.path.join(directory, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, file_name)
    with gzip.open(model_path, 'wb', 4) as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model_path

def load_model(file_name, L, n, periodic, num_EV, directory=model_dir):

    directory = os.path.join(directory, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, file_name)
    with gzip.open(model_path, 'rb') as fp:
        model_params = pickle.load(fp)

    hparams = model_params["hparams"]
    model = MBLModel(hparams=hparams)
    model.load_state_dict( model_params["state_dict"] )
    model.prepare_data()

    return model.to(device)

def model_exists(file_name, L, n, periodic, num_EV, directory=model_dir):

    directory = os.path.join(directory, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, file_name)
    return os.path.exists(model_path)



# ============================================================
# Store classifier evaluation output.
# ============================================================

def save_eval_random(obj, model_version, L, n, periodic, num_EV, data_dir=eval_random_data_dir):
    """Save model predictions of random W != {0.5, 8}.

    Parameters
    ----------
    obj : list
        A list of five numpy.ndarray [Ws, Ps, Ws_uniq, Ps_mean, Ps_std]
        Where `W` are disorder strength, and `P` the probability of being in the localized phase.
    model_version : int
        Version of the neural network model.
    L : int
        System size.
    n : int
        Number of consecutive spins sampled.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))
    os.makedirs(directory, exist_ok=True)

    with gzip.open(os.path.join(directory, 'model_v{}_eval.pkl.gz'.format(model_version)), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_eval_random(model_version, L, n, periodic, num_EV, data_dir=eval_random_data_dir):
    """Load model predictions of random W != {0.5, 8}.

    Parameters
    ----------
    model_version : int
        Version of the neural network model.
    L : int
        System size.
    n : int
        Number of consecutive spins sampled.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_EV : int
        Number of eigenvalues around zero being sampled.
    data_dir : str, optional
        Directory where the data is saved.

    Return
    ------
    obj : list
        A list of five numpy.ndarray [Ws, Ps, Ws_uniq, Ps_mean, Ps_std]
        Where `W` are disorder strength, and `P` the probability of being in the localized phase.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'n={:02d}'.format(n), 'periodic={}'.format(periodic), 'num_EV={}'.format(num_EV))

    if os.path.isfile(os.path.join(directory, 'model_v{}_eval.pkl.gz'.format(model_version))):
        os.makedirs(directory, exist_ok=True)
        with gzip.open(os.path.join(directory, 'model_v{}_eval.pkl.gz'.format(model_version)), 'rb') as handle:
            data = pickle.load(handle)
        return data
    else:
        return None



# ============================================================
# Store CNN classifier that learns directly from Hamiltonian.
# ============================================================

def save_H_model(model, file_name, L, periodic, directory=H_model_dir):
    """Save model as pickle"""

    model = model.cpu()
    model_dict = {
        "state_dict": model.state_dict(),
        "hparams": model.hparams
    }

    directory = os.path.join(directory, 'L={:02d}'.format(L), 'periodic={}'.format(periodic))
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, file_name)
    with gzip.open(model_path, 'wb', 4) as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model_path

def load_H_model(file_name, L, periodic, directory=H_model_dir):

    directory = os.path.join(directory, 'L={:02d}'.format(L), 'periodic={}'.format(periodic))
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, file_name)
    with gzip.open(model_path, 'rb') as fp:
        model_params = pickle.load(fp)

    hparams = model_params["hparams"]
    model = MBLHModel(hparams=hparams)
    model.load_state_dict( model_params["state_dict"] )
    model.prepare_data()

    return model.to(device)

def H_model_exists(file_name, L, periodic, directory=H_model_dir):

    directory = os.path.join(directory, 'L={:02d}'.format(L), 'periodic={}'.format(periodic))
    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, file_name)
    return os.path.exists(model_path)



# ============================================================
# Store Hamiltonian classifier evaluation output.
# ============================================================

def save_H_eval_random(obj, model_version, L, periodic, data_dir=H_eval_random_data_dir):
    """Save model predictions of random W != {0.5, 8}.

    Parameters
    ----------
    obj : list
        A list of five numpy.ndarray [Ws, Ps, Ws_uniq, Ps_mean, Ps_std]
        Where `W` are disorder strength, and `P` the probability of being in the localized phase.
    model_version : int
        Version of the neural network model.
    L : int
        System size.
    periodic : bool
        Whether the Hamiltonian is periodic.
    data_dir : str, optional
        Directory where the data is saved.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L),'periodic={}'.format(periodic))
    os.makedirs(directory, exist_ok=True)

    with gzip.open(os.path.join(directory, 'H_model_v{}_eval.pkl.gz'.format(model_version)), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_H_eval_random(model_version, L, periodic, data_dir=H_eval_random_data_dir):
    """Load model predictions of random W != {0.5, 8}.

    Parameters
    ----------
    model_version : int
        Version of the neural network model.
    L : int
        System size.
    periodic : bool
        Whether the Hamiltonian is periodic.
    data_dir : str, optional
        Directory where the data is saved.

    Return
    ------
    obj : list
        A list of five numpy.ndarray [Ws, Ps, Ws_uniq, Ps_mean, Ps_std]
        Where `W` are disorder strength, and `P` the probability of being in the localized phase.
    """

    directory = os.path.join(data_dir, 'L={:02d}'.format(L), 'periodic={}'.format(periodic))
    os.makedirs(directory, exist_ok=True)

    if os.path.isfile(os.path.join(directory, 'H_model_v{}_eval.pkl.gz'.format(model_version))):
        with gzip.open(os.path.join(directory, 'H_model_v{}_eval.pkl.gz'.format(model_version)), 'rb') as handle:
            data = pickle.load(handle)
        return data
    else:
        return None
