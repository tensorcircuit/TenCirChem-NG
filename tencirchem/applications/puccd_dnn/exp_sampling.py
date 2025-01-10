from collections import Counter
from typing import List

import numpy as np

from tensorcircuit import Circuit


n_samples = 1 << 16


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


# fix the sampling variable and this function in the end, after classical simulation is done
prob_cache = {}
def get_prob(c: Circuit, circuit_id: int, x_idx: List[int], y_idx: List[int]):
    cache_key = circuit_id, tuple(x_idx), tuple(y_idx)
    prob_cache.clear()
    if cache_key in prob_cache:
        print(f"Cache hit {circuit_id}")
        return prob_cache[cache_key]
    else:
        c = c.copy()
        add_measurement(c, x_idx, y_idx)
        samples = c.sample(n_samples, allow_state=True)
        counter = Counter([tuple(s) for s, p in samples])
        prob = {}
        for k, v in counter.items():
            prob[k] = v / n_samples
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
            idx2.append(idx-n_qubits)
    return idx1, idx2


def apply_x_gates(s, x_idx):
    s2 = np.zeros(len(s), dtype=int)
    s2[x_idx] = 1
    return (s + s2) % 2


def get_batch_norm_and_exp_z(c1: Circuit, c2: Circuit, qop_list):
    prob1 = get_prob(c1, 1, [], [])
    prob2 = get_prob(c2, 2, [], [])
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
        x_idx , y_idx, z_idx = to_xyz_index(qop)
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


def get_batch_exp_xy(c1: Circuit, c2: Circuit, qop_list, n_qubits):
    # n_qubits for the number of qubits in each circuit and not the overall circuit

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

        prob1 = get_prob(c1, 1, x_idx1, y_idx1)
        prob2 = get_prob(c2, 2, x_idx2, y_idx2)

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
                    phase *= (1 - 2 * s2[star2])
                else:
                    assert star2 is None
                    phase *= (1 - 2 * s1[star1])

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
