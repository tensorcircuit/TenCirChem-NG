import numpy as np
from openfermion import QubitOperator, hermitian_conjugated
from tensorcircuit import Circuit, set_backend

from tencirchem import UCCSD, PUCCD


K = set_backend()


def process_mol(m, active_space=None, aslst=None, check_stability=False):
    hf = m.HF()
    hf.kernel()
    if check_stability:
        # run stability check for C4H4
        dm, _, flag, _ = hf.stability(return_status=True)
        while not flag:
            hf.kernel(dm)
            dm, _, flag, _ = hf.stability(return_status=True)

    uccsd = UCCSD(hf, active_space=active_space, aslst=aslst)
    puccd = PUCCD(hf, active_space=active_space, aslst=aslst)

    puccd.kernel()

    puccd.print_energy()
    puccd.print_excitations()

    ci_addr = uccsd.get_ci_strings()
    ci_mask = np.zeros(1 << (uccsd.n_qubits))
    ci_mask[ci_addr] = 1
    ci_mask = ci_mask > 0

    c = Circuit(puccd.n_qubits)
    c = Circuit(uccsd.n_qubits, inputs=np.kron(puccd.statevector(), c.state()))

    for i in range(puccd.n_qubits):
        theta = 0.2
        c.ry(puccd.n_qubits + i, theta=theta)

    for i in range(puccd.n_qubits):
        c.cnot(i, i + puccd.n_qubits)

    fermion_circuit_state = K.numpy(c.state()).real
    fermion_circuit_state[~ci_mask] = 0
    fermion_circuit_state /= np.linalg.norm(fermion_circuit_state)

    # get the matrix Hamiltonian instead of the function Hamiltonian
    uccsd.energy(uccsd.init_guess, engine="tensornetwork")
    hamiltonian = np.array(uccsd.hamiltonian_lib["sparse"].todense())
    print(fermion_circuit_state.T @ (hamiltonian @ fermion_circuit_state) + uccsd.e_core)
    return fermion_circuit_state, hamiltonian, puccd.e_core, ci_mask, uccsd, puccd


def get_hamiltonian(ops, n_qubits, discard_eps=1e-3):
    # get CNOT transformed Hamiltonian
    # could be made more efficient using Clifford algebra
    new_ops = {}
    for k, v in ops.terms.items():
        if np.abs(v) >= discard_eps:
            new_ops[k] = v

    h_qop = QubitOperator()
    h_qop.terms = new_ops

    cnot_idxs = [[i, i + n_qubits] for i in range(n_qubits)]

    cnot_qops = []
    for i, j in cnot_idxs:
        cnot = QubitOperator(f"Z{i}", 0.5) + 0.5 - QubitOperator(f"Z{i} X{j}", 0.5) + QubitOperator(f"X{j}", 0.5)
        cnot_qops.append(cnot)

    cnot_stair_qop = cnot_qops[0]
    for cnot in cnot_qops[1:]:
        cnot_stair_qop = cnot_stair_qop * cnot

    h_cnot_qop = cnot_stair_qop * h_qop * hermitian_conjugated(cnot_stair_qop)
    h_cnot_qop.compress()

    h_cnot_qop_z = QubitOperator()
    h_cnot_qop_xy = QubitOperator()
    for k, v in h_cnot_qop.terms.items():
        is_xy = False
        for term in k:
            if term[1] in ["X", "Y"]:
                is_xy = True
                break
        if is_xy:
            h_cnot_qop_xy.terms[k] = v
        else:
            h_cnot_qop_z.terms[k] = v
    return h_cnot_qop_z, h_cnot_qop_xy


def permute_s_cnot(s):
    # CNOT transformed basis
    n_qubits = len(s) // 2
    cnot_idxs = [[i, i + n_qubits] for i in range(n_qubits)]
    s = s.copy()
    for cnot in cnot_idxs:
        s[cnot[1]] = (s[cnot[1]] + s[cnot[0]]) % 2
    return s
