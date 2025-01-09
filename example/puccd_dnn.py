from typing import Sequence
from itertools import product
import time

import numpy as np
import jax
import flax.linen as nn
from flax.training import train_state
import optax
from matplotlib import pyplot as plt
from tensorcircuit import set_backend, Circuit
from tencirchem.molecule import nh3
from tencirchem import UCCSD, PUCCD


K = set_backend("jax")


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def process_mol(m, active_space=None):
    uccsd = UCCSD(m, active_space=active_space, engine="tensornetwork")
    ucc = PUCCD(m, active_space=active_space, engine="tensornetwork")
    ucc.kernel()
    # uccsd.kernel()
    ucc.print_energy()

    ci_addr = uccsd.get_ci_strings()
    ci_mask = np.zeros(1 << (uccsd.n_qubits))
    ci_mask[ci_addr] = 1
    ci_mask = ci_mask > 0

    c = Circuit(ucc.n_qubits)
    c = Circuit(uccsd.n_qubits, inputs=np.kron(ucc.statevector(), c.state()))

    for i in range(ucc.n_qubits):
        theta = 0.2
        c.ry(ucc.n_qubits + i, theta=theta)

    for i in range(ucc.n_qubits):
        c.cnot(i, i + ucc.n_qubits)

    fermion_circuit_state = K.numpy(c.state()).real
    fermion_circuit_state[~ci_mask] = 0
    fermion_circuit_state /= K.norm(fermion_circuit_state)
    print(fermion_circuit_state.T @ (uccsd.hamiltonian @ fermion_circuit_state) + uccsd.e_core)
    return fermion_circuit_state, uccsd.hamiltonian, ucc.e_core, ci_mask, uccsd, ucc


m_list = [
    ("nh3", nh3(), (8, 7)),
]

tx = optax.adamax(optax.linear_schedule(1e-2, 1e-3, 32000, 8000), b1=0.8, b2=0.99)
e_min_lists = []
for name, m, active_space in m_list:
    fermion_circuit_state, hamiltonian, e_nuc, ci_mask, uccsd, ucc = process_mol(m, active_space)
    nn_size = uccsd.n_qubits
    size_factor = 2
    model = MLP([nn_size * size_factor] * (uccsd.n_qubits // 2 - 3) + [1])
    batch = K.convert_to_tensor(list(product(*[[-1, 1] for _ in range(uccsd.n_qubits)])))

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
    print(e_min_list)
    e_min_lists.append(e_min_list)
    plt.yscale("log")
    plt.show()
