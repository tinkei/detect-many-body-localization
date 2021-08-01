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
    """Obtain eigenstates with energy around zero using sparse exact diagonalization.

    Exponentially expensive in L, only works for small enough `L` <= 14.

    An array representing the k eigenvectors. The column v[:, i] is the eigenvector corresponding to the eigenvalue w[i].

    Parameters
    ----------
    Hs : list of scipy.sparse.csr_matrix
        A list of Hamiltonians to diagonalize.
    k : int
        Number of eigenvalues around E = 0 to obtain.

    Return
    ------
    Es : list of 1D numpy.ndarray
        Eigenvalues, sorted in ascending order.
    Vs : list of 2D numpy.ndarray
        Eigenvectors.
    """

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

def batch_generate_ED_data(L, W, J=1, periodic=False, num_Hs=1000, num_EV=20, save_data=False, npsp='sp'):
    """Batch generate random Hamiltonians and apply exact diagonization, then store the output.
    
    Only used for demonstration. Do not use in practice.
    """

    obj_params = {'J': J, 'periodic': periodic} # Drop L and W. We use them for subdirectories.
    Hs = build_Hs(L, W, J, periodic, num_Hs)

    if npsp == 'sp':
        E0s, V0s = EDs_sparse(Hs, num_EV)
    else:
        Es, Vs = EDs(Hs)
        E0s = []
        V0s = []
        for E, V in zip(Es, Vs):
            E0, V0 = select_N_eigenvalues(E, V, num_EV)
            E0s.append(E0)
            V0s.append(V0)

    if save_data:
        save_ED(E0s, 'E0s', L, W, periodic)
        save_ED(V0s, 'V0s', L, W, periodic)

    return E0s, V0s



# ============================================================
# Partial trace.
# ============================================================

@njit
def get_rho(V):
    """Computes a full density matrix from an eigenvector `V`."""
    return np.outer(V, V.conj())

def build_partial_trace_matrix(L, A_sites, B1_sites, B2_sites):
    """INCORRECT IMPLEMENTATION!
    Build a rectangular matrix to take partial trace over subsystem B.
    Can only handle n consecutive sites in the middle of the system, surrounded by B1 and B2.
    i.e. |B1> x |A> x |B2>
    """

    n    = L - len(B1_sites) - len(B2_sites)
    I_A  = identity(2**len(A_sites))
    # Tr_B = csr_matrix((2**L, 2**n))

    if len(B1_sites) != 0:
        B1 = np.ones((2**len(B1_sites), 1))
        B1 = csr_matrix(B1)
    else:
        B1 = None
    if len(B2_sites) != 0:
        B2 = np.ones((2**len(B2_sites), 1))
        B2 = csr_matrix(B2)
    else:
        B2 = None

    if B1 is not None:
        Tr_B = kron(  B1, I_A, 'csr')
    else:
        Tr_B = I_A
    if B2 is not None:
        Tr_B = kron(Tr_B,  B2, 'csr')

    # if B1 is not None:
    #     print(B1.shape)
    # print(I_A.shape)
    # if B2 is not None:
    #     print(B2.shape)
    # print(Tr_B.shape)

    return Tr_B

def partial_trace_matrix(L, A_sites, B1_sites, B2_sites, V):
    """INCORRECT IMPLEMENTATION!
    Take partial trace over subsystem B using matrix product.
    Can only handle n consecutive sites in the middle of the system, surrounded by B1 and B2.
    i.e. |B1> x |A> x |B2>
    """

    rho = get_rho(V)
    Tr_B = build_partial_trace_matrix(L, A_sites, B1_sites, B2_sites)
    rho_A = csr_matrix.dot(Tr_B.T.conj(), csr_matrix.dot(rho, Tr_B))

    return rho_A

def partial_trace_tensor(L, A_sites, B1_sites, B2_sites, V):
    """Take partial trace over subsystem B using tensor contraction.
    Can only handle n consecutive sites in the middle of the system, surrounded by B1 and B2.
    i.e. |B1> x |A> x |B2>
    """

    n = len(A_sites)
    V = V.reshape([2]*L)
    rho_A = np.tensordot(V.conj(), V, axes=(B1_sites+B2_sites, B1_sites+B2_sites))
    rho_A = rho_A.reshape((2**n, 2**n))

    return rho_A

def partial_trace_kevin(L, A_sites, B1_sites, B2_sites, V):
    """Take partial trace over subsystem B using tensor contraction, but using Kevin's order of contraction.
    Can only handle n consecutive sites in the middle of the system, surrounded by B1 and B2.
    i.e. |B1> x |A> x |B2>
    """

    V = V.reshape((2**len(B1_sites), 2**len(A_sites), 2**len(B2_sites)))
    rho_A = np.tensordot(V.conj(), V, axes=([0, 2], [0, 2]))

    return rho_A

def partial_trace_jonas(L, A_sites, B1_sites, B2_sites, V):
    """INCORRECT IMPLEMENTATION!
    Take partial trace over subsystem B using tensor contraction, but using Jonas's implementation.
    Can only handle n consecutive sites in the middle of the system, surrounded by B1 and B2.
    i.e. |B1> x |A> x |B2>
    """

    dm = get_rho(V)

    # Jonas' original implementation:
    # dm_res = dm.reshape(2**sitesA, 2**sitesB, 2**sitesA, 2**sitesB) # rho as 4-tensor
    # rhoA = np.trace(dm_res, axis1=1, axis2=3)

    # My modification:
    dm_res = dm.reshape(2**len(B1_sites), 2**len(A_sites), 2**len(B2_sites), 2**len(B1_sites), 2**len(A_sites), 2**len(B2_sites)) # rho as 6-tensor
    dm_res = np.trace(dm_res, axis1=0, axis2=2)
    rho_A  = np.trace(dm_res, axis1=1, axis2=3)

    return rho_A

# Final version used in production: `partial_trace_kevin()`.
def partial_trace(L, A_sites, B1_sites, B2_sites, V):
    """Take partial trace over subsystem B using tensor contraction.
    Can only handle n consecutive sites in the middle of the system, surrounded by B1 and B2.
    i.e. |B1> x |A> x |B2>
    """

    V = V.reshape((2**len(B1_sites), 2**len(A_sites), 2**len(B2_sites)))
    rho_A = np.tensordot(V.conj(), V, axes=([0, 2], [0, 2]))

    return rho_A



# ============================================================
# Batch generation of reduced density matrix `rho`.
# ============================================================

def batch_gen_rho_data_core(L, Ws, J=1, periodic=False, num_Hs=1000, num_EV=5, max_n=99999, clamp_zero=1e-32):

    rho_As = {}

    for W in Ws:

        Hs = build_Hs(L, W, J, periodic, num_Hs)
        E0s, V0s = EDs_sparse(Hs, num_EV)

        for E0, V0 in tqdm(zip(E0s, V0s), leave=False, desc='partial_trace()'): # Still an array of `num_EV` eigenvectors per E0, V0.
            for i in range(len(E0)): # Flatten list of arrays.

                A_sites  = list(range(1, L-1)) # Keep n consecutive sites, where n = L - 2.
                A_sites0 = A_sites
                parity   = 0

                while len(A_sites) != 0:

                    n = len(A_sites)

                    # Only compute reduced density matrix is size is small.
                    if n <= max_n:

                        B1_sites = list(range(0, A_sites[0]))
                        B2_sites = list(range(A_sites[-1]+1, L))
                        rho_A    = partial_trace(L, A_sites, B1_sites, B2_sites, V0[:,i])

                        # Investigate how many data points are actually zero.
                        # if n > 5:
                        #     print('W:', W)
                        #     print('rho_A <= 2'    , (np.abs(rho_A) <= 2    ).sum())
                        #     print('rho_A <= 1'    , (np.abs(rho_A) <= 1    ).sum())
                        #     print('rho_A <= 1e-8' , (np.abs(rho_A) <= 1e-8 ).sum())
                        #     print('rho_A <= 1e-12', (np.abs(rho_A) <= 1e-12).sum())
                        #     print('rho_A <= 1e-16', (np.abs(rho_A) <= 1e-16).sum())
                        #     print('rho_A <= 1e-20', (np.abs(rho_A) <= 1e-20).sum())
                        #     print('rho_A <= 1e-32', (np.abs(rho_A) <= 1e-32).sum())
                        #     print('rho_A == 0', (np.abs(rho_A) == 0).sum())
                        #     rho_A[np.abs(rho_A) <= 1e-32] = 0
                        #     print('rho_A == 0', (np.abs(rho_A) == 0).sum())

                        # If the magnitude of data is below a certain threshold, set it to zero.
                        if clamp_zero is not None:
                            rho_A[np.abs(rho_A) <= clamp_zero] = 0

                        # Check if imaginary part is negligible.
                        # Conclusion: They are basically 1e-16.
                        # if n > 5:
                        #     print('W:', W)
                        #     print('Max rho_A:', np.max(np.abs(rho_A)))
                        #     print('Max real rho_A:', np.max(np.abs(np.real(rho_A))))
                        #     print('Max imag rho_A:', np.max(np.abs(np.imag(rho_A))))

                        if n not in rho_As:
                            rho_As[n] = []
                        rho_As[n].append([rho_A.astype(np.float32), W]) # .astype(np.complex64)

                    # Shorten n:
                    if parity % 3 == 0:
                        A_sites  = A_sites0[1:]
                        parity  += 1
                    elif parity % 3 == 1:
                        A_sites  = A_sites0[:-1]
                        parity  += 1
                    else:
                        A_sites  = A_sites0[1:-1]
                        A_sites0 = A_sites
                        parity   = 0
                    # print(L, len(A_sites0), len(A_sites), parity, flush=True)

    return rho_As

def batch_gen_rho_data_main(L, Ws, J=1, periodic=False, num_Hs=1000, num_EV=5, max_n=99999, clamp_zero=1e-32, save_data=True):
    """Generate a list of reduced density matrices with W = {0.5, 8}, and save only the real part to file.
    The data is used to train a classifier neutral network.

    Parameters
    ----------
    L : int
        System size.
    Ws : list of float
        A list of disorder strength W to realize.
    J : float
        Coupling strength. Always set to 1.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_Hs : int
        Number of Hamiltonians to generate, per disorder strength W.
    num_EV : int
        Number of eigenvalues around zero to use as samples.
    max_n : int
        Maximum number of consecutive sites n to store.
        Values beyond 8 will consume a GB of storage per 10000 samples.
        Recommended: 8.
    clamp_zero : float
        Threshold of reduced density matrix, magnitudes below which will be clamped to zero.

    Return
    ------
    rho_As : dict of list of list
        A `dict` of keys `n`, which are consecutive spins around the middle of the system, that the reduced density matrix is computed.
        The value of the dict is a list of lists, where each reduced density matrix is paired with its disorder strength W.
        i.e. rho_As[6][i][0] is a 2D numpy.ndarray of the reduced density matrix, and rho_As[6][i][1] is the disorder strength used to generate it.
        Number of data must be a multiple of 10.
    """

    rho_As = batch_gen_rho_data_core(L, Ws, J, periodic, num_Hs, num_EV, max_n, clamp_zero)

    if save_data:
        for n, rho_A in rho_As.items():
            # For odd `n`, #samples = num_Hs * num_EV * 2
            # For even `n`, #samples = num_Hs * num_EV
            tqdm.write('Writing {: 5d} `rho_A` of system size L = {:02d} of n = {:02d}.'.format(len(rho_A), L, n)) #, flush=True)
            save_rho_train(rho_A, 'rho_A', L, n, periodic, num_EV)

    return rho_As

def batch_gen_rho_data_rand(L, Ws, J=1, periodic=False, num_Hs=1000, num_EV=5, max_n=99999, clamp_zero=1e-32, save_data=True):
    """Generate a list of reduced density matrices with W != {0.5, 8}, and save only the real part to file.
    The data is used to predict transition disorder strength W_c.

    Parameters
    ----------
    L : int
        System size.
    Ws : list of float
        A list of disorder strength W to realize.
    J : float
        Coupling strength. Always set to 1.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_Hs : int
        Number of Hamiltonians to generate, per disorder strength W.
    num_EV : int
        Number of eigenvalues around zero to use as samples.
    max_n : int
        Maximum number of consecutive sites n to store.
        Values beyond 8 will consume a GB of storage per 10000 samples.
        Recommended: 8.
    clamp_zero : float
        Threshold of reduced density matrix, magnitudes below which will be clamped to zero.

    Return
    ------
    rho_As : dict of list of list
        A `dict` of keys `n`, which are consecutive spins around the middle of the system, that the reduced density matrix is computed.
        The value of the dict is a list of lists, where each reduced density matrix is paired with its disorder strength W.
        i.e. rho_As[6][i][0] is a 2D numpy.ndarray of the reduced density matrix, and rho_As[6][i][1] is the disorder strength used to generate it.
        Number of data must be a multiple of 10.
    """

    rho_As = batch_gen_rho_data_core(L, Ws, J, periodic, num_Hs, num_EV, max_n, clamp_zero)

    if save_data:
        for n, rho_A in rho_As.items():
            # For odd `n`, #samples = num_Hs * num_EV * len(Ws) * 2
            # For even `n`, #samples = num_Hs * num_EV * len(Ws)
            tqdm.write('Writing {: 5d} `rho_A` of system size L = {:02d} of n = {:02d}.'.format(len(rho_A), L, n)) #, flush=True)
            save_rho_random(rho_A, 'rho_A', L, n, periodic, num_EV)

    return rho_As



# ============================================================
# Batch generation of `EVW` tuples.
# (eigenvalue, eigenvector, disorder strength)
# ============================================================

def batch_gen_EVW_data_core(L, Ws, J=1, periodic=False, num_Hs=1000, num_EV=5):

    EVWs = []

    for W in Ws:

        Hs = build_Hs(L, W, J, periodic, num_Hs)
        E0s, V0s = EDs_sparse(Hs, num_EV)

        for E0, V0 in zip(E0s, V0s): # Still an array of `num_EV` eigenvectors per E0, V0.
            for i, E in enumerate(E0): # Flatten list of arrays.
                EVWs.append((E0[i], V0[:,i].astype(np.complex64), W))

    return EVWs

def batch_gen_EVW_data_main(L, Ws, J=1, periodic=False, num_Hs=1000, num_EV=5, save_data=True):
    """Generate a list of eigenvectors with W = {0.5, 8}, and save them to file.
    The data is used to train a classifier neutral network.

    Parameters
    ----------
    L : int
        System size.
    Ws : list of float
        A list of disorder strength W to realize.
    J : float
        Coupling strength. Always set to 1.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_Hs : int
        Number of Hamiltonians to generate, per disorder strength W.
    num_EV : int
        Number of eigenvalues around zero to use as samples.

    Return
    ------
    EVWs : list of lists
        A list where each element is [E, V, W] = (eigenvalue, eigenvector, disorder strength).
    """

    EVWs = batch_gen_EVW_data_core(L, Ws, J, periodic, num_Hs, num_EV)

    if save_data:
        tqdm.write('Writing {: 5d} `EVW` of system size L = {:02d}.'.format(len(EVWs), L)) #, flush=True)
        save_EVW_train(EVWs, 'EVWs', L, periodic, num_EV)

    return EVWs

def batch_gen_EVW_data_rand(L, Ws, J=1, periodic=False, num_Hs=1000, num_EV=5, save_data=True):
    """Generate a list of eigenvectors with W = {0.5, 8}, and save them to file.
    The data is used to train a classifier neutral network.

    Parameters
    ----------
    L : int
        System size.
    Ws : list of float
        A list of disorder strength W to realize.
    J : float
        Coupling strength. Always set to 1.
    periodic : bool
        Whether the Hamiltonian is periodic.
    num_Hs : int
        Number of Hamiltonians to generate, per disorder strength W.
    num_EV : int
        Number of eigenvalues around zero to use as samples.

    Return
    ------
    EVWs : list of lists
        A list where each element is [E, V, W] = (eigenvalue, eigenvector, disorder strength).
    """

    EVWs = batch_gen_EVW_data_core(L, Ws, J, periodic, num_Hs, num_EV)

    if save_data:
        tqdm.write('Writing {: 5d} `EVW` of system size L = {:02d}.'.format(len(EVWs), L)) #, flush=True)
        save_EVW_random(EVWs, 'EVWs', L, periodic, num_EV)

    return EVWs
