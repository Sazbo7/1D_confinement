import numpy as np

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

    #elif initial_state == "Random_product":
    #    init_psi =

    else:
        print("ERROR: invalid initial state defined \n Possible options: 'Haar', 'pol_[u]', 'inft'.")

    assert(init_psi.shape == (2**L, ))
    return init_psi;
