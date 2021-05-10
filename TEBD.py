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
from tenpy.models.lattice import Site, Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import asConfig
from tenpy.networks.site import SpinHalfSite


class MixedIsingModel(CouplingMPOModel):
    r"""Spin-1/2 Mixed Ising chain.
    The Hamiltonian reads:
    .. math ::
        H = \sum_i \mathtt{Jz}_{i,i+1} S^z_i S^z_{i+1} \\
            + \sum_i \mathtt{h}_i S^z_i \\
            + \sum_i \mathtt{g}_i S^x_i
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.
    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`MixedIsingChain` below.
    Options
    -------
    .. cfg:config :: MixedIsingChain
        :include: CouplingMPOModel
        L : int
            Length of the chain.
        Jxx, h, g : float | array
            Coupling as defined for the Hamiltonian above.
        bc_MPS : {'finite' | 'infinte'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = SpinHalfSite(conserve=conserve)
        return site


    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        h = np.asarray(model_params.get('h', 1.))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmax')
            self.add_onsite(-h, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        # done


class Kitaev_ladder(CouplingMPOModel):
    """Spin-1/2 Mixed Ising chain.
    The Hamiltonian reads:
    .. math ::
        H = \sum_i \mathtt{J^\mu_i}_{i,i+1} S^\mu_i S^\mu_{i+1} \\
            + \sum_i \mathtt{h}_i S^z_i \\
            + \sum_i \mathtt{g}_i S^x_i
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.
    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`MixedIsingChain` below.
    Options
    -------
    .. cfg:config :: MixedIsingChain
        :include: CouplingMPOModel
        L : int
            Number of rungs of ladder.
        Jxx, h, g : float | array
            Coupling as defined for the Hamiltonian above.
        bc_MPS : {'finite' | 'infinte'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """

    def __init__(self, model_params):

        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', "Ladder")

        # define some default parameters
        Jz = model_params.get('Jz', 2.0)
        Jx = model_params.get('Jx', 1.0)
        Jy = model_params.get('Jy', 1.0)
        hz = model_params.get('hz', 0.0)
        hx = model_params.get('hx', 0.0)
        hy = model_params.get('hy', 0.)

        bc = 'periodic' if model_params['bc_MPS'] == 'infinite' else 'open'
        CouplingMPOModel.__init__(self, model_params)

        self.add_coupling(Jz, 0, 'Sz', 1, 'Sz', 0, plus_hc=True)
        self.add_coupling([Jx, 0], 0, 'Sx', 0, 'Sx', 1, plus_hc=True)
        self.add_coupling([0, Jy], 0, 'Sy', 0, 'Sy', 1, plus_hc=True)
        self.add_coupling([0, Jx], 1, 'Sx', 1, 'Sx', 1, plus_hc=True)
        self.add_coupling([Jy, 0], 1, 'Sy', 1, 'Sy', 1, plus_hc=True)

        self.add_onsite(hz, 0, 'Sz')
        self.add_onsite(hz, 1, 'Sz')
        self.add_onsite(hx, 0, 'Sx')
        self.add_onsite(hx, 1, 'Sx')
        self.add_onsite(hy, 0, 'Sy')
        self.add_onsite(hy, 1, 'Sy')

        # construct the Hamiltonian in the Matrix-Product-Operator (MPO) picture
        MPOModel.__init__(self, lat, self.calc_H_MPO())

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            if self.verbose >= 1.:
                print(self.name + ": set conserve to", conserve)
        site = SpinHalfSite(conserve=conserve)
        return site


    def init_terms(self, model_params):
        Jx = np.asarray(model_params.get('Jx', 1.))
        Jy = np.asarray(model_params.get('Jy', 1.))
        Jz = np.asarray(model_params.get('Jz', 1.))

        g = np.asarray(model_params.get('g', 1.))
        h = np.asarray(model_params.get('h', 1.))

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmax')
            self.add_onsite(-h, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmaz', u2, 'Sigmaz', dx)
        # done


class MixedIsingChain(MixedIsingModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.
    See the :class:`TFIModel` for the documentation of parameters.
    """
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)


def TEBD_mixed_ising_confined(L, h, g, tmax, dt, verbose=True):
    '''Time-evolve mixed Ising Chain using TEBD MPS algorithm. Employs TenPy engine for MPS and MPO representations.
    '''

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

def TEBD_DAOE(L, h, g, tmax, dt, verbose=True):
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
    return S, sz_ray
