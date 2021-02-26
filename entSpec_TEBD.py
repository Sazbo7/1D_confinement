import numpy as np
import matplotlib.pyplot as plt
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain, TFIModel
from tenpy.algorithms import tebd, dmrg
from tenpy.models.spins import SpinModel
import scipy.sparse as sparse
import scipy.sparse.linalg.eigen.arpack as arp
import warnings
import scipy.integrate


def TEBD_mixed_ising_confined(L, h, g, tmax, dt, verbose=True):
    print("finite TEBD, real time evolution")
    print("L={L:d}, h={h:.2f}, g={g:.2f}, tmax={tmax:.2f}, dt={dt:.3f}".format(L=L, h=h, g=g, tmax=tmax, dt=dt))

    model_params = dict(L=L, J=1., h=h, g=g, bc_MPS='finite', conserve=None, verbose=True)
    M = MixedIsingChain(model_params)
    product_state = ["up"] * (M.lat.N_sites//2) + ["down"] * (M.lat.N_sites//2) + ["down"] * (M.lat.N_sites%2)
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    dt_measure = 0.05
    # tebd.Engine makes 'N_steps' steps of `dt` at once; for second order this is more efficient.
    tebd_params = {
        'order': 2,
        'dt': dt,
        'N_steps': int(dt_measure / dt + 0.5),
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'verbose': verbose,
    }
    eng = tebd.Engine(psi, M, tebd_params)
    S = [psi.entanglement_entropy()];
    sz_ray = [psi.expectation_value('Sz')]

    for n in range(int(tmax / dt_measure + 0.5)):
        eng.run()
        S.append(psi.entanglement_entropy())
        sz_ray.append(psi.expectation_value('Sz'))

    plt.figure()
    plt.imshow(sz_ray[::-1],
               vmin=0.,
               aspect='auto',
               interpolation='nearest',
               extent=(0, L - 1., -0.5 * dt_measure, eng.evolved_time + 0.5 * dt_measure))
    plt.xlabel('site $i$')
    plt.ylabel('time $t/J$')
    plt.ylim(0., tmax)
    plt.colorbar().set_label('entropy $S$')
    #filename = 'c_tebd_lightcone_{g:.2f}.pdf'.format(g=g)
    #plt.savefig(filename)
    return S, sz_ray
