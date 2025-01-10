import pytest

import numpy as np
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator

from tencirchem.applications.puccd_dnn.exp_sampling import get_batch_norm_and_exp_z, get_batch_exp_xy
from tensorcircuit import Circuit


def generate_random_circuit_and_nn(n_qubits, seed):
    np.random.seed(seed)

    states = []
    c_list = []
    for _ in range(2):
        circuit_state = np.random.rand(2 ** n_qubits) - 0.5
        circuit_state /= np.linalg.norm(circuit_state)
        states.append(circuit_state)
        c_list.append(Circuit(n_qubits, inputs=circuit_state))
    state = np.kron(states[0], states[1])
    c1, c2 = c_list
    nn = np.random.rand(2 ** (2 * n_qubits)) - 0.5
    return c1, c2, state, nn


def query_nn(keys, nn):
    keys_int = [int("".join(map(str, key)), base=2) for key in keys]
    return nn[keys_int]


@pytest.mark.parametrize("system", [(1, "Z0"), (1, "X0 Z1"), (4, "X0 X1 Z6"), (4, "Z0 Z2 Y5 Y6")])
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_expectation_simple(system, seed):
    n_qubits, paulistring = system
    c1, c2, state, nn = generate_random_circuit_and_nn(n_qubits, seed)
    state_nn = state * nn

    norm_ref = state_nn @ state_nn
    mat = np.array(get_sparse_operator(QubitOperator(paulistring), n_qubits=2 * n_qubits).todense())
    exp_ref = state_nn @ mat @ state_nn
    atol = 2e-3

    qop_list = [QubitOperator(paulistring)]
    if "X" not in paulistring and "Y" not in paulistring:
        norm_keys, norm_values, z_keys, z_values = get_batch_norm_and_exp_z(c1, c2, qop_list)

        b = query_nn(norm_keys, nn)
        norm = (b ** 2).dot(norm_values)
        np.testing.assert_allclose(norm, norm_ref, atol=atol)

        b = query_nn(z_keys, nn)
        exp = (b ** 2).dot(z_values)
        np.testing.assert_allclose(exp, exp_ref, atol=atol)

    else:
        key01, key10, key00, key11, values = get_batch_exp_xy(c1, c2, qop_list, n_qubits)
        f01 = query_nn(key01, nn)
        f10 = query_nn(key10, nn)
        f00 = query_nn(key00, nn)
        f11 = query_nn(key11, nn)
        exp_xy = (f01 * f10 / 2 + f00 * f11 / 2) @ np.array(values)
        np.testing.assert_allclose(exp_xy, exp_ref, atol=atol)

