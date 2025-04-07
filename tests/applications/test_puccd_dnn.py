import pytest

import numpy as np
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from tensorcircuit import Circuit

from tencirchem import UCCSD
from tencirchem.molecule import h_chain, h4
from tencirchem.applications.puccd_dnn.expectation_sampling import (
    get_batch_norm_and_exp_z,
    get_batch_exp_xy,
    get_prob,
    e_and_norm,
)
from tencirchem.applications.puccd_dnn.transformations import process_mol
from tencirchem.applications.puccd_dnn.models import get_circuit_nosym


def generate_random_circuit_and_nn(n_qubits, seed):
    np.random.seed(seed)

    states = []
    c_list = []
    for _ in range(2):
        circuit_state = np.random.rand(2**n_qubits) - 0.5
        circuit_state /= np.linalg.norm(circuit_state)
        states.append(circuit_state)
        c_list.append(Circuit(n_qubits, inputs=circuit_state))
    state = np.kron(states[0], states[1])
    c1, c2 = c_list
    nn = np.random.rand(2 ** (2 * n_qubits)) - 0.5
    return c1, c2, state, nn


def generate_mol_circuit_and_nn(n_qubits):
    states = []
    c_list = []
    for _ in range(2):
        circuit_state = np.ones(2**n_qubits)
        circuit_state /= np.linalg.norm(circuit_state)
        states.append(circuit_state)
        c_list.append(Circuit(n_qubits, inputs=circuit_state))
    state = np.kron(states[0], states[1])
    c1, c2 = c_list
    mol = h_chain(n_qubits)
    uccsd = UCCSD(mol)
    nn = uccsd.statevector(uccsd.init_guess)
    return c1, c2, state, nn


def query_nn(keys, nn):
    keys_int = [int("".join(map(str, key)), base=2) for key in keys]
    return nn[keys_int]


@pytest.mark.parametrize("system", [(1, "Z0"), (1, "X0 Z1"), (4, "X0 X1 Z6"), (4, "Z0 Z2 Y5 Y6")])
@pytest.mark.parametrize("seed", [3, 4, 5, None])
def test_expectation_simple(system, seed):
    n_qubits, paulistring = system
    if seed is not None:
        c1, c2, state, nn = generate_random_circuit_and_nn(n_qubits, seed)
    else:
        if n_qubits == 1:
            pytest.skip("At least 2 qubits for a mol")
        c1, c2, state, nn = generate_mol_circuit_and_nn(n_qubits)
    state_nn = state * nn

    norm_ref = state_nn @ state_nn
    mat = np.array(get_sparse_operator(QubitOperator(paulistring), n_qubits=2 * n_qubits).todense())
    exp_ref = state_nn @ mat @ state_nn
    atol = 5e-3
    shots = 1 << 13

    qop_list = [QubitOperator(paulistring)]
    if "X" not in paulistring and "Y" not in paulistring:
        norm_keys, norm_values, z_keys, z_values = get_batch_norm_and_exp_z(c1, c2, qop_list, shots=shots)

        b = query_nn(norm_keys, nn)
        norm = (b**2).dot(norm_values)
        np.testing.assert_allclose(norm, norm_ref, atol=atol)

        b = query_nn(z_keys, nn)
        exp = (b**2).dot(z_values)
        np.testing.assert_allclose(exp, exp_ref, atol=atol)

    else:
        key01, key10, key00, key11, values = get_batch_exp_xy(c1, c2, qop_list, n_qubits, shots=shots)
        f01 = query_nn(key01, nn)
        f10 = query_nn(key10, nn)
        f00 = query_nn(key00, nn)
        f11 = query_nn(key11, nn)
        exp_xy = (f01 * f10 / 2 + f00 * f11 / 2) @ np.array(values)
        np.testing.assert_allclose(exp_xy, exp_ref, atol=atol)


def test_expectation_mol():
    n_qubits = 4
    m = h4

    fermion_circuit_state, hamiltonian, e_nuc, ci_mask, uccsd, puccd = process_mol(m)

    params = puccd.params
    c1 = get_circuit_nosym(params, puccd)
    c2 = Circuit(n_qubits)
    for i in range(n_qubits):
        c2.ry(i, theta=0.2)

    c = Circuit(uccsd.n_qubits, inputs=np.kron(c1.state(), c2.state()))

    for i in range(puccd.n_qubits):
        c.cnot(i, i + puccd.n_qubits)

    circuit_state = c.state().real

    # "artificial" neural network
    nn_array = np.zeros_like(circuit_state)
    nn_array[ci_mask] = uccsd.civector_fci / circuit_state[ci_mask]

    def nn_func(batch):
        batch = batch.copy()
        batch[batch == -1] = 0
        key = np.dot(batch, 2 ** np.arange(batch.shape[1])[::-1])
        return nn_array[key]

    e, norm = e_and_norm(c1, c2, n_qubits, uccsd.n_elec // 2, uccsd.h_qubit_op, get_prob, None, nn_func, discard_eps=0)
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)

    # time consuming test
    # e_sum = 0
    # for op in uccsd.h_qubit_op:
    #     e, norm = e_and_norm(c1, c2, n_qubits, uccsd.n_elec // 2, op, get_prob, None, nn_func)
    #     state = np.zeros_like(circuit_state)
    #     state[ci_mask] = uccsd.civector_fci
    #     e_ref = state @ np.array(get_sparse_operator(op, uccsd.n_qubits).todense()) @ state
    #     #print(op, e, e_ref)
    #     np.testing.assert_allclose(e, e_ref, atol=1e-5)
    #     e_sum += e
    # print(e_sum)
