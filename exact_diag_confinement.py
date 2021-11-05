from __future__ import print_function, division
import sys,os

from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian, quantum_operator # Hamiltonian and observables
from quspin.tools.measurements import obs_vs_time # t_dep measurements
from quspin.tools.Floquet import Floquet,Floquet_t_vec # Floquet Hamiltonian
from quspin.basis.photon import coherent_state # HO coherent state
import numpy as np # generic math functions

from numpy.random import ranf,seed # pseudo random numbers
from joblib import delayed,Parallel # parallelisation

from quspin.operators import exp_op # operators
from quspin.basis import spin_basis_general # spin basis constructor
from quspin.tools.measurements import ent_entropy # Entanglement Entropy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from time import clock
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from scipy.sparse import load_npz, save_npz

def generate_initial_state(L, initial_state='Haar', seed=None):

    if initial_state == "Haar":
        init_psi = np.random.normal(size=2**L) + 1.j * np.random.normal(size=2**L)
        init_psi /= np.linalg.norm(init_psi)

    elif initial_state == "pol_z":
        init_psi = np.zeros([2**L]);
        init_psi[0] = 1;

    elif initial_state == "pol_x":
        init_psi = 1/np.sqrt(2**L) * np.ones([2**L]);

    elif initial_state == "pol_y":
        psi_y = [1.0, 1.0j];
        init_psi = psi_y;
        for i in range(L-1):
            init_psi = np.kron(init_psi, psi_y);
        init_psi = 1/np.sqrt(2**L) * init_psi;

    elif initial_state == "InfT":
        init_psi = np.diagonal([2**L]);

    else:
        print("ERROR: invalid initial state defined \n Possible options: 'Haar', 'pol_[u]', 'inft'.")

    assert(init_psi.shape == (2**L,))
    return init_psi;

def kitaev_ladder(L, K_ray, h_ray, BC='periodic'):

    if BC == 'periodic':
        BC = 1;
    else:
        BC = 0;

    Kz, Ky, Kx = K_ray[0], K_ray[1], K_ray[2];
    hz, hy, hx = h_ray[0], h_ray[1], h_ray[2];

    H_zz = [[Kz,i,L + i] for i in range(L)] # PBC
    H_xx_1 = [[Kx,2*i,(2*i+1) % (L-1)] for i in range(L//2 + BC * (L%2))] # PBC
    H_xx_2 = [[Kx,L + 2*i + 1,L + (2*i+2)%L] for i in range(L//2 + BC * (1 - L%2) - 1*(1 - L%2))] # PBC
    H_yy_1 = [[Ky,2*i+1,(2*i+2) % (L-1)] for i in range(L//2 + BC * (1 - L%2) - 1*(1 - L%2))] # PBC
    H_yy_2 = [[Ky,L + 2*i, L + (2*i+1)%L] for i in range(LL//2 + BC * (L%2))] # PBC
    H_z = [[hz,i] for i in range(2*L)] # PBC
    H_y = [[hy,i] for i in range(2*L)] # PBC
    H_x = [[hx,i] for i in range(2*L)] # PBC

    # define static and dynamics lists
    static=[["zz",H_zz], ["xx",H_xx_1], ["xx",H_xx_2], ["yy",H_yy_1], ["yy",H_yy_2], ["z",H_z], ["y",H_y], ["x",H_x]];
    #static=[["zz",H_zz],["z",H_z],["x",H_x]];
    dynamic=[]
    basis=spin_basis_1d(L=2*L)
    H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False);
    return H;

def heisenberg_ladder(L, J_ray, h_ray, rung_c = 1.0, BC='periodic'):

    if BC == 'periodic':
        BC = 1;
    else:
        BC = 0;

    Jz, Jy, Jx = J_ray[0], J_ray[1], J_ray[2];
    hz, hy, hx = h_ray[0], h_ray[1], h_ray[2];

    ### Along Length

    H_zz_r = [[Jz * alpha,i,L + i] for i in range(L)] # PBC
    H_xx_r = [[Jx * alpha,i,L + i] for i in range(L)] # PBC
    H_yy_r = [[Jy * alpha,i,L + i] for i in range(L)] # PBC

    H_zz_l = [[Jz ,i,(i+1)%L)] for i in range(L)] # PBC
    H_xx_l = [[Jx,i,(i+1)%L] for i in range(L)] # PBC
    H_yy_l = [[Jy,i,(i+1)%L] for i in range(L)] # PBC

    H_z = [[hz,i] for i in range(2*L)] # PBC
    H_y = [[hy,i] for i in range(2*L)] # PBC
    H_x = [[hx,i] for i in range(2*L)] # PBC

    # define static and dynamics lists
    static=[["zz",H_zz_r], ["zz",H_zz_l], ["xx",H_xx_r], ["xx",H_xx_l], ["yy",H_yy_r], ["yy",H_yy_l], ["z",H_z], ["y",H_y], ["x",H_x]];
    #static=[["zz",H_zz],["z",H_z],["x",H_x]];
    dynamic=[]
    basis=spin_basis_1d(L=2*L)
    H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False);
    return H;
