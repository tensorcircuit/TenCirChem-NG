from typing import Sequence

import flax.linen as nn
from tensorcircuit import Circuit

from tencirchem import PUCCD


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def get_circuit_nosym(params, puccd: PUCCD):
    c = Circuit(puccd.n_qubits)
    for i in range(puccd.n_qubits - puccd.n_elec // 2, puccd.n_qubits):
        c.x(i)

    for i, (j, k) in enumerate(puccd.ex_ops):
        c.cnot(k, j)
        c.ry(k, theta=params[i])
        c.cnot(k, j)
    return c
