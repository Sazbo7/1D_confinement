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
        Jzz, h, g : float | array
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


class MixedIsingChain(MixedIsingModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.
    See the :class:`TFIModel` for the documentation of parameters.
    """
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)


def ApplyTwoSitePurified(psi, i0, op, trunc_params):

    # Get the two site operator either from str or npc.Array;
    if isinstance(op, str):
        if len(op) < 4:
            raise ValueError("len of ops incommensurate two site ")
        for i in range(len(op)%2 - 1):
            index = 2*i
            op1 = psi.sites[i].get_op(op[index:index+2]).replace_labels(['p', 'p*'], ['p0', 'p0*']);
            op2 = psi.sites[i].get_op(op[index+2:index+4]).replace_labels(['p', 'p*'], ['p1', 'p1*']);
        op = npc.outer(op1, op2).itranspose(['p0', 'p1', 'p0*', 'p1*']);
    theta = psi.get_theta(i0, 2);
    theta = npc.tensordot(op, theta, axes=[['p0*', 'p1*'], ['p0', 'p1']]);
    theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1]);
    U, S, V, trunc_err, renormalize = svd_theta(theta, trunc_params, inner_labels=['vR', 'vL']);
    B_R = V.split_legs(1).ireplace_labels(['p1', 'q1'], ['p', 'q']);
    #  In general, we want to do the following:
    #      B_L = U.iscale_axis(S, 'vR')
    #      B_L = B_L.split_legs(0).iscale_axis(self.psi.get_SL(i0)**(-1), 'vL')
    #      B_L = B_L.ireplace_labels(['p0', 'q0'], ['p', 'q'])
    # i.e. with SL = self.psi.get_SL(i0), we have ``B_L = SL**(-1) U S``
    # However, the inverse of SL is problematic, as it might contain very small singular
    # values.  Instead, we calculate ``C == SL**-1 theta == SL**-1 U S V``,
    # such that we obtain ``B_L = SL**-1 U S = SL**-1 U S V V^dagger = C V^dagger``
    C = psi.get_theta(i0, n=n, formL=0.)
    # here, C is the same as theta, but without the `S` on the very left
    # (Note: this requires no inverse if the MPS is initially in 'B' canonical form)
    C = npc.tensordot(op, C, axes=(['p0*', 'p1*'], ['p0', 'p1']))  # apply U as for theta
    B_L = npc.tensordot(C.combine_legs(('vR', 'p1', 'q1'), pipes=theta.legs[1]),
                        V.conj(),
                        axes=['(vR.p1.q1)', '(vR*.p1*.q1*)'])
    B_L.ireplace_labels(['vL*', 'p0', 'q0'], ['vR', 'p', 'q'])
    B_L /= renormalize  # re-normalize to <psi|psi> = 1
    psi.set_SR(i0, S)
    psi.set_B(i0, B_L, form='B')
    psi.set_B(i0+n-1, B_R, form='B')
    return trunc_err


def Purified_TEBD_Heisenberg(L, Jzz, Jxx, h, beta):
    L = 10
    chi = 5
    delta_t = 0.1
    model_params = {
        'L': L,
        'S': 0.5,
        'conserve': 'Sz',
        'Jz': 1.0,
        'Jy': 1.0,
        'Jx': 1.0,
        'hx': 0.0,
        'hy': 0.0,
        'hz': 0.0,
        'muJ': 0.0,
        'bc_MPS': 'finite',
    }

    heisenberg = tenpy.models.spins.SpinChain(model_params)
    product_state = ["up"] * (L // 2) + ["down"] * (L - L // 2)
    # starting from a domain-wall product state which is not an eigenstate of the Heisenberg model
    psi = MPS.from_product_state(heisenberg.lat.mps_sites(),
                                 product_state,
                                 bc=heisenberg.lat.bc_MPS,
                                 form='B')
