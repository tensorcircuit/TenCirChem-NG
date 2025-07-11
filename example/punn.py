from itertools import product
import time
import pickle

import numpy as np
import jax
from flax.training import train_state
import optax
from matplotlib import pyplot as plt
from tencirchem.molecule import nh3, n2, ch4, nh4, co, h_chain, bh3, h_cube, c4h4
from tencirchem.applications.puccd_dnn.transformations import process_mol
from tencirchem.applications.puccd_dnn.models import MLP


# molecules for the manuscript

# molecules for Fig.2&3
m_list = [
    ("bh3", bh3(), (6, 7)),
    ("nh3", nh3(), (8, 7)),
    ("n2", n2(), (10, 8)),
    ("ch4", ch4(), (8, 8)),
     ("nh4", nh4(), (8, 8)),
    ("co", co(), (10, 8)),
]


# molecules for Fig. 4. H chains
m_list = []
for system_size in range(4, 9):
    system_name = f"H{system_size}"
    print(system_name)
    for d in [10, 25]:
        print(d)
        if system_size % 2 == 0:
            charge = 0
        else:
            charge = 1
        m = h_chain(system_size, bond_distance=d/10, charge=charge)
        active_space = (system_size-charge, system_size)
    m_list.append((system_name, m, active_space))


# molecules for Fig.5, H8 cube
m_list = []
for d in np.linspace(0.5, 2.5, 11):
    print(d)
    m = h_cube(d=d)
    active_space = (8, 8)
    m_list.append((f"H8 cube {d}", m, active_space))


# molecules for Fig. 6.
basis="ccpvdz"
m1 = c4h4(1.35, 1.58, basis=basis)
m2 = c4h4(1.46, 1.46, basis=basis)


# simpler molecules for the example (for raster run)
m_list = [
    ("h5p", h_chain(5, 1.0, charge=1), (4, 5)),
    ("nh3", nh3(), (8, 7)),
]


tx = optax.adamax(optax.linear_schedule(1e-2, 1e-3, 32000, 8000), b1=0.8, b2=0.99)
e_min_lists = []
for name, m, active_space in m_list:
    fermion_circuit_state, hamiltonian, e_nuc, ci_mask, uccsd, puccd = process_mol(m, active_space)
    nn_size = uccsd.n_qubits
    # adjust this for different widths of the NN
    size_factor = 2
    model = MLP([nn_size * size_factor] * (uccsd.n_qubits // 2 - 3) + [1])
    batch = list(product(*[[-1, 1] for _ in range(uccsd.n_qubits)]))

    @jax.jit
    def train_step(ts, fermion_circuit_state, hamiltonian, e_nuc):
        print("jit")

        def loss_fn(params):
            state = ts.apply_fn(params, batch)
            wfn = fermion_circuit_state * state.ravel()
            norm = (wfn.T @ wfn).ravel()
            return ((wfn.T @ hamiltonian @ wfn).ravel() / norm)[0]

        grad_fn = jax.value_and_grad(loss_fn)
        e, grads = grad_fn(ts.params)
        ts = ts.apply_gradients(grads=grads)
        return ts, e + e_nuc

    e_min_list = []
    for seed in range(10, 15):
        # for seed in [12]:
        variables = model.init(jax.random.PRNGKey(seed), batch)
        ts = train_state.TrainState.create(apply_fn=model.apply, params=variables, tx=tx)
        nsteps = 64000
        e_list = []
        time1 = time.time()
        for i in range(nsteps):
            ts, e = train_step(ts, fermion_circuit_state, hamiltonian, e_nuc)
            e_list.append(e)
        # will cost ~10 min for each seed on CPU. Will be faster on GPUs
        print(time.time() - time1)
        plt.plot(np.array(e_list) - uccsd.e_fci)
        e_min_list.append(np.min(e_list))
        print(seed, np.min(e_list))
        if name == "h5p" and seed == 0:
            with open("puccd_dnn_params.pkl", "wb") as fout:
                pickle.dump(ts.params, fout)
    print(e_min_list)
    e_min_lists.append(e_min_list)
    plt.yscale("log")
    plt.show()
