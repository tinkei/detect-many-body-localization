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

from file_io import *



# ============================================================
# Helper functions.
# ============================================================

@njit
def get_h(L, W):
    """Generate random field.
    """
    h = np.random.uniform(-W, W, L)
    return h

@njit
def is_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])



# ============================================================
# Construct Hamiltonian.
# ============================================================

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

    sx_list = load_cache('sx_list', obj_params)
    sy_list = load_cache('sy_list', obj_params)
    sz_list = load_cache('sz_list', obj_params)

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

        save_cache(sx_list, 'sx_list', obj_params)
        save_cache(sy_list, 'sy_list', obj_params)
        save_cache(sz_list, 'sz_list', obj_params)

    # else:

    #     print('Cache found for `si_list`. Load from cache.')

    # ========================================
    # End cached area: si_list.
    # ========================================

    return sx_list, sy_list, sz_list

def build_H_ii(L, periodic):

    sx_list, sy_list, sz_list = build_si_list(L)

    # ========================================
    # Start cached area: H_ii.
    # ========================================
    
    obj_params = {'L': L, 'periodic': periodic}

    H_xx = load_cache('H_xx', obj_params)
    H_yy = load_cache('H_yy', obj_params)
    H_zz = load_cache('H_zz', obj_params)

    if H_xx is None or H_yy is None or H_zz is None:

        # print('Cache not found for `H_ii`. Generate from scratch.')

        H_xx = csr_matrix((2**L, 2**L))
        H_yy = csr_matrix((2**L, 2**L))
        H_zz = csr_matrix((2**L, 2**L))

        for i in range(L if periodic else L - 1):
            H_xx = H_xx + sx_list[i] * sx_list[(i + 1) % L]
            H_yy = H_yy + sy_list[i] * sy_list[(i + 1) % L]
            H_zz = H_zz + sz_list[i] * sz_list[(i + 1) % L]

        save_cache(H_xx, 'H_xx', obj_params)
        save_cache(H_yy, 'H_yy', obj_params)
        save_cache(H_zz, 'H_zz', obj_params)

    # else:

    #     print('Cache found for `H_ii`. Load from cache.')

    # ========================================
    # End cached area: H_ii.
    # ========================================

    return H_xx, H_yy, H_zz, sx_list, sy_list, sz_list

def build_H(L, W, J, periodic=False):

    H_xx, H_yy, H_zz, sx_list, sy_list, sz_list = build_H_ii(L, periodic)

    # H_z is not cached due to randomness.
    H_z  = csr_matrix((2**L, 2**L))
    h    = get_h(L, W)

    for i in range(L):
        H_z = H_z + h[i] * sz_list[i]

    H = J * (H_xx + H_yy + H_zz) - H_z

    return H

def build_Hs(L, W, J, periodic=False, num_Hs=1000):

    H_xx, H_yy, H_zz, sx_list, sy_list, sz_list = build_H_ii(L, periodic)

    Hs = []
    for i in tqdm(range(num_Hs), leave=False, desc='build_Hs()'):

        # H_z is not cached due to randomness.
        H_z  = csr_matrix((2**L, 2**L))
        h    = get_h(L, W)

        for i in range(L):
            H_z = H_z + h[i] * sz_list[i]

        H = J * (H_xx + H_yy + H_zz) - H_z
        Hs.append(H)

    return Hs



# ============================================================
# Exact Diagonalization.
# ============================================================

@njit
def ED(H):
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.

    The column V[:, i] is the normalized eigenvector corresponding to the eigenvalue E[i].
    Will return a matrix object if a is a matrix object.

    Parameters
    ----------
    H : numpy.ndarray
        Hamiltonian to diagonalize.

    Return
    ------
    E : 1D numpy.ndarray
        Eigenvalues, sorted in ascending order.
    V : 2D numpy.ndarray
        Eigenvectors.
    """

    # if L >= 20:
    #     warnings.warn("Large L: Exact diagonalization might take a long time!")

    E, V = np.linalg.eigh(H)

    assert is_sorted(E), 'Eigenvalues not sorted!'

    return E, V

def ED_sparse(H, k):
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.

    An array representing the k eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i].

    Parameters
    ----------
    H : numpy.ndarray
        Hamiltonian to diagonalize.
    k : int
        Number of eigenvalues around E = 0 to obtain.

    Return
    ------
    E : 1D numpy.ndarray
        Eigenvalues, sorted in ascending order.
    V : 2D numpy.ndarray
        Eigenvectors.
    """

    # if L >= 20:
    #     warnings.warn("Large L: Exact diagonalization might take a long time!")

    E, V = scipy.sparse.linalg.eigsh(H, k=k, sigma=0, which='LM', return_eigenvectors=True)
    sorted_indices = np.abs(E).argsort()
    E = E[sorted_indices]
    V = V[:, sorted_indices]

    assert is_sorted(np.abs(E)), 'Eigenvalues not sorted!'

    return E, V

def EDs(Hs):
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.

    The column V[:, i] is the normalized eigenvector corresponding to the eigenvalue E[i].
    Will return a matrix object if a is a matrix object.

    Parameters
    ----------
    Hs : list of scipy.sparse.csr_matrix
        A list of Hamiltonians to diagonalize.

    Return
    ------
    E : list of 1D numpy.ndarray
        Eigenvalues of each Hamiltonian, sorted in ascending order.
    V : list of 2D numpy.ndarray
        Eigenvectors of each Hamiltonian.
    """

    # if L >= 20:
    #     warnings.warn("Large L: Exact diagonalization might take a long time!")

    Es = []
    Vs = []
    for H in Hs:

        # Can't use scipy's eigsh, because we need ALL eigenwhatevers.
        # E, V = eigsh(H, k=10, which='SM', return_eigenvectors=True)
        # E, V = np.linalg.eigh(H.A)
        E, V = ED(H.toarray())
        Es.append(E)
        Vs.append(V)

    return Es, Vs

def EDs_sparse(Hs, k):
    """For comparison: obtain ground state energy from exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <~ 20.

    An array representing the k eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i].

    Parameters
    ----------
    Hs : list of scipy.sparse.csr_matrix
        A list of Hamiltonians to diagonalize.
    k : int
        Number of eigenvalues around E = 0 to obtain.

    Return
    ------
    E : 1D numpy.ndarray
        Eigenvalues, sorted in ascending order.
    V : 2D numpy.ndarray
        Eigenvectors.
    """

    # if L >= 20:
    #     warnings.warn("Large L: Exact diagonalization might take a long time!")

    Es = []
    Vs = []
    for H in tqdm(Hs, leave=False, desc='EDs_sparse()'):

        E, V = scipy.sparse.linalg.eigsh(H, k=k, sigma=0, which='LM', return_eigenvectors=True)
        sorted_indices = np.abs(E).argsort()
        E = E[sorted_indices]
        V = V[:, sorted_indices]
        Es.append(E)
        Vs.append(V)

    return Es, Vs

def select_N_eigenvalues(E, V, n, where='zeroest'):
    """
    Select N eigenvalues closest to the lowest, to zero, or to the highest.

    Parameters
    ----------
    E : 1D numpy.ndarray
        Sorted eigenvalues in ascending order.
    V : 2D numpy.ndarray
        Corresponding eigenvectors.
        The column V[:, i] is the normalized eigenvector corresponding to the eigenvalue E[i].
    n : int
        Number of eigenvalues to store.
    where : str
        Where to select the eigenvalues. where = {'lowest', 'zeroest', 'highest'}
    """

    if where == 'lowest':
        E = E[:n]
        V = V[:, :n]
    elif where == 'highest':
        E = E[-n:]
        V = V[:, -n:]
    elif where == 'zeroest':
        # closest_indices = np.abs(E).argsort()[:n]
        # E = E[closest_indices]
        # V = V[:, closest_indices]
        # Faster implementation, but Numba doesn't support this. Still, it is a few microseconds faster.
        # Source: https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
        closest_indices = np.argpartition(np.abs(E), n)[:n]
        E_temp = E[closest_indices]
        V_temp = V[:, closest_indices]
        sorted_indices = np.abs(E_temp).argsort()[:n]
        E = E_temp[sorted_indices]
        V = V_temp[:, sorted_indices]
        # assert np.all(E1 == E), 'Both sorting methods should be identical.'
        # assert np.all(V1 == V), 'Both sorting methods should be identical.'

    return E, V
