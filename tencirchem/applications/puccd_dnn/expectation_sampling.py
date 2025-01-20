from collections import Counter
from typing import List, Callable

import numpy as np
from openfermion import QubitOperator
from tensorcircuit import Circuit

from tencirchem.applications.puccd_dnn.transformations import permute_s_cnot, get_hamiltonian


def add_measurement(c: Circuit, x_idx: List[int], y_idx: List[int]):
    if x_idx:
        star = x_idx[0]
    elif y_idx:
        star = y_idx[0]
    else:
        return c

    for idx in x_idx:
        if idx == star:
            continue
        c.cnot(star, idx)
    for idx in y_idx:
        if idx == star:
            continue
        c.cy(star, idx)

    if x_idx:
        assert star == x_idx[0]
        c.H(star)
    else:
        assert star == y_idx[0]
        c.SD(star)
        c.H(star)
    return c


def merge_prob(prob1, prob2):
    prob = {}
    for k1, v1 in prob1.items():
        for k2, v2 in prob2.items():
            key = list(k1) + list(k2)
            prob[tuple(key)] = v1 * v2
    return prob


prob_cache = {}


def get_prob(c: Circuit, circuit_id: int, x_idx: List[int], y_idx: List[int], shots=1 << 10):
    cache_key = circuit_id, tuple(x_idx), tuple(y_idx)
    # don't actually do cache. The code here is merely a caching example, for experiments on real
    # quantum computers, it might be more economic to cache some of the measurement shots
    prob_cache.clear()
    if cache_key in prob_cache:
        print(f"Cache hit {circuit_id}")
        return prob_cache[cache_key]
    else:
        c = c.copy()
        add_measurement(c, x_idx, y_idx)
        if shots is not None:
            samples = c.sample(shots, allow_state=True)
            counter = Counter([tuple(s) for s, p in samples])
            prob = {}
            for k, v in counter.items():
                prob[k] = v / shots
        else:
            # exact probability
            prob = {}
            for i, p in enumerate(c.probability()):
                k = tuple(map(int, np.binary_repr(i, c._nqubits)))
                prob[k] = p
        prob_cache[cache_key] = prob
        return prob


def to_xyz_index(qop):
    x_idx = [t[0] for t in list(qop.terms.keys())[0] if t[1] == "X"]
    y_idx = [t[0] for t in list(qop.terms.keys())[0] if t[1] == "Y"]
    z_idx = [t[0] for t in list(qop.terms.keys())[0] if t[1] == "Z"]
    return x_idx, y_idx, z_idx


# split the overall index to the two circuits
def split_idx(idx, n_qubits):
    # n_qubits is the number of qubits for one circuit, instead of the overall circuit
    idx1 = []
    idx2 = []
    for idx in idx:
        if idx < n_qubits:
            idx1.append(idx)
        else:
            idx2.append(idx - n_qubits)
    return idx1, idx2


def apply_x_gates(s, x_idx):
    s2 = np.zeros(len(s), dtype=int)
    s2[x_idx] = 1
    return (s + s2) % 2


def get_batch_norm_and_exp_z(c1: Circuit, c2: Circuit, qop_list, shots=1 << 10, fun_get_prob=None):
    if fun_get_prob is None:
        fun_get_prob = get_prob
    prob1 = fun_get_prob(c1, 1, [], [], shots)
    prob2 = fun_get_prob(c2, 2, [], [], shots)
    prob = merge_prob(prob1, prob2)

    # norm keys and values
    norm_keys = []
    norm_values = []
    for k, v in prob.items():
        norm_keys.append(k)
        norm_values.append(v)

    # z keys and values
    z_keys = []
    z_values = []
    for qop in qop_list:
        x_idx, y_idx, z_idx = to_xyz_index(qop)
        coeff = list(qop.terms.values())[0]
        if x_idx or y_idx:
            continue

        for k, v in prob.items():
            phase = 1
            for z in z_idx:
                if k[z] == 1:
                    phase *= -1
            z_keys.append(k)
            z_values.append(phase * v * coeff)

    return norm_keys, norm_values, z_keys, z_values


def get_batch_exp_xy(c1: Circuit, c2: Circuit, qop_list, n_qubits, shots=1 << 10, fun_get_prob=None):
    # n_qubits for the number of qubits in each circuit and not the overall circuit
    if fun_get_prob is None:
        fun_get_prob = get_prob

    # 0 for normal index and 1 for tilde index
    # 00 -> kj, 01 -> k\tilde{j}, etc
    key01 = []
    key10 = []
    key00 = []
    key11 = []
    values = []

    for qop in qop_list:
        x_idx, y_idx, z_idx = to_xyz_index(qop)
        coeff = list(qop.terms.values())[0]

        if not (x_idx or y_idx):
            continue

        x_idx1, x_idx2 = split_idx(x_idx, n_qubits)
        y_idx1, y_idx2 = split_idx(y_idx, n_qubits)
        z_idx1, z_idx2 = split_idx(z_idx, n_qubits)

        if x_idx1:
            star1 = x_idx1[0]
        elif y_idx1:
            star1 = y_idx1[0]
        else:
            star1 = None
        if x_idx2:
            star2 = x_idx2[0]
        elif y_idx2:
            star2 = y_idx2[0]
        else:
            star2 = None

        assert star1 is not None or star2 is not None

        prob1 = get_prob(c1, 1, x_idx1, y_idx1, shots)
        prob2 = get_prob(c2, 2, x_idx2, y_idx2, shots)

        for s1, p1 in prob1.items():
            for s2, p2 in prob2.items():
                phase = 1

                for z in z_idx1:
                    if s1[z] == 1:
                        phase *= -1
                for z in z_idx2:
                    if s2[z] == 1:
                        phase *= -1

                if star1 is not None and star2 is not None:
                    # both c1 and c2 have X/Y
                    if s1[star1] + s2[star2] == 1:
                        phase *= -1
                    else:
                        assert s1[star1] + s2[star2] == 0 or s1[star1] + s2[star2] == 2
                        phase *= 1
                elif star1 is None:
                    phase *= 1 - 2 * s2[star2]
                else:
                    assert star2 is None
                    phase *= 1 - 2 * s1[star1]

                str01 = list(s1) + apply_x_gates(list(s2), x_idx2 + y_idx2).tolist()
                str10 = apply_x_gates(str01, x_idx + y_idx)
                if star1 is not None:
                    str01[star1] = 0
                    str10[star1] = 1
                if star2 is not None:
                    str01[n_qubits + star2] = 1
                    str10[n_qubits + star2] = 0
                # print(str1, str2)
                key01.append(tuple(str01))
                key10.append(tuple(str10))

                str00 = list(s1) + list(s2)
                str11 = apply_x_gates(str00, x_idx + y_idx)
                if star1 is not None:
                    str00[star1] = 0
                    str11[star1] = 1
                if star2 is not None:
                    str00[n_qubits + star2] = 0
                    str11[n_qubits + star2] = 1
                if star1 is None:
                    assert np.allclose(str00, str10) and np.allclose(str01, str11)
                if star2 is None:
                    assert np.allclose(str00, str01) and np.allclose(str10, str11)
                # print(str1, str2)
                key00.append(tuple(str00))
                key11.append(tuple(str11))
                # print(s1[0], s2[0], str1, str2, str3, str4)
                # print(f1, f2, f3, f4)

                values.append(coeff * phase * p1 * p2)

    return key01, key10, key00, key11, values


def get_h_cnot_qop_xyi(h_cnot_qop_xy, n_qubits):
    h_cnot_qop_xyi = QubitOperator()
    for qop in h_cnot_qop_xy:
        x_idx, y_idx, z_idx = to_xyz_index(qop)
        coeff = list(qop.terms.values())[0]

        if not (x_idx or y_idx):
            continue

        x_idx1, x_idx2 = split_idx(x_idx, n_qubits)
        y_idx1, y_idx2 = split_idx(y_idx, n_qubits)
        z_idx1, z_idx2 = split_idx(z_idx, n_qubits)

        # replace an X with -Y or a Y with X
        if x_idx1 and ((not y_idx1) or x_idx1[0] < y_idx1[0]):
            if x_idx2 and ((not y_idx2) or x_idx2[0] < y_idx2[0]):
                # XX -> YY
                phase = 1
                y_idx1.append(x_idx1[0])
                x_idx1 = x_idx1[1:]
                y_idx2.append(x_idx2[0])
                x_idx2 = x_idx2[1:]
            elif y_idx2:
                # XY -> YX
                phase = -1
                y_idx1.append(x_idx1[0])
                x_idx1 = x_idx1[1:]
                x_idx2.append(y_idx2[0])
                y_idx2 = y_idx2[1:]
            else:
                continue
        elif y_idx1:
            if x_idx2 and ((not y_idx2) or x_idx2[0] < y_idx2[0]):
                # YX -> XY
                phase = -1
                x_idx1.append(y_idx1[0])
                y_idx1 = y_idx1[1:]
                y_idx2.append(x_idx2[0])
                x_idx2 = x_idx2[1:]
            elif y_idx2:
                # YY -> XX
                phase = 1
                x_idx1.append(y_idx1[0])
                y_idx1 = y_idx1[1:]
                x_idx2.append(y_idx2[0])
                y_idx2 = y_idx2[1:]
            else:
                continue
        else:
            continue
        # print(qop)
        # print(x_idx1)
        x_idx = x_idx1 + [i + n_qubits for i in x_idx2]
        y_idx = y_idx1 + [i + n_qubits for i in y_idx2]
        z_idx = z_idx1 + [i + n_qubits for i in z_idx2]
        key = []
        for idx_list, symbol in zip([x_idx, y_idx, z_idx], "XYZ"):
            for i in idx_list:
                key.append((i, symbol))
        key.sort()
        h_cnot_qop_xyi.terms[tuple(key)] = phase * coeff
    return h_cnot_qop_xyi


def get_batch(key, n_elec):
    n_qubits = len(key[0]) // 2
    a = np.array([permute_s_cnot(list(s)) for s in key])
    mask = (a[:, :n_qubits].sum(axis=1) == n_elec) & (a[:, n_qubits:].sum(axis=1) == n_elec)
    a[a == 0] = -1
    return a, mask


def e_and_norm(
    c1: Circuit,
    c2: Circuit,
    n_qubits: int,
    n_elec: int,
    h_qubit_op: QubitOperator,
    fun_get_prob: Callable,
    shots: int,
    fun_nn: Callable,
    discard_eps=1e-3,
):
    # n_qubits and n_elec are for one spin sector
    h_cnot_qop_z, h_cnot_qop_xy = get_hamiltonian(h_qubit_op, n_qubits, discard_eps)
    h_cnot_qop_xyi = get_h_cnot_qop_xyi(h_cnot_qop_xy, n_qubits)

    norm_keys, norm_values, z_keys, z_values = get_batch_norm_and_exp_z(
        c1, c2, h_cnot_qop_z, fun_get_prob=fun_get_prob, shots=shots
    )
    norm_values = np.array(norm_values)
    norm_batch, norm_mask = get_batch(norm_keys, n_elec)
    norm = (fun_nn(norm_batch).ravel() * norm_mask) ** 2 @ norm_values

    if z_keys:
        z_values = np.array(z_values)
        z_batch, z_mask = get_batch(z_keys, n_elec)
        exp_z = (fun_nn(z_batch).ravel() * z_mask) ** 2 @ z_values / norm
    else:
        exp_z = 0

    for qop in [h_cnot_qop_xy, h_cnot_qop_xyi]:
        if qop is None or len(qop.terms) == 0:
            # sometimes this term is empty, for example when qop is identity
            if qop is h_cnot_qop_xy:
                exp_xy = 0
            else:
                assert qop is h_cnot_qop_xyi
                exp_xyi = 0
            continue

        key01, key10, key00, key11, values_xy = get_batch_exp_xy(
            c1, c2, qop, n_qubits, fun_get_prob=fun_get_prob, shots=shots
        )
        values_xy = np.array(values_xy)
        f01_batch, f01_mask = get_batch(key01, n_elec)
        f10_batch, f10_mask = get_batch(key10, n_elec)
        f00_batch, f00_mask = get_batch(key00, n_elec)
        f11_batch, f11_mask = get_batch(key11, n_elec)

        f01 = fun_nn(f01_batch).ravel() * f01_mask
        f10 = fun_nn(f10_batch).ravel() * f10_mask
        f00 = fun_nn(f00_batch).ravel() * f00_mask
        f11 = fun_nn(f11_batch).ravel() * f11_mask

        if qop is h_cnot_qop_xy:
            exp_xy = (f01 * f10 / 2 + f00 * f11 / 2) @ values_xy / norm
        else:
            assert qop is h_cnot_qop_xyi
            exp_xyi = (f01 * f10 / 2 - f00 * f11 / 2) @ values_xy / norm

    # print(h_cnot_qop_xy, h_cnot_qop_xyi, exp_z, exp_xy, exp_xyi)
    return exp_z + exp_xy + exp_xyi, norm
