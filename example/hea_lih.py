import numpy as np
from pyscf import M
from pyscf.mcscf import CASCI

from tencirchem.static.hea import HEA
from tencirchem import UCCSD, set_backend

set_backend("jax")

d = 2.5
mol = M(atom=[["H", 0, 0, 0], ["Li", d, 0, 0]], charge=0, symmetry=True)
print(mol.ao_labels())
# reference energy
ucc = UCCSD(mol)
ucc.print_energy()

# move active orbitals to the middle
hf = mol.HF()
hf.kernel()
# the MO orbitals
# 0      1                      2                      3       4       5
# Li 1s, H 1s + Li 2s + Li 2px, H 1s + Li 2s + Li 2px, Li 2py, Li 2pz, Li 2s
with np.printoptions(precision=2):
    print(hf.mo_coeff)

mycas = CASCI(hf, 3, 2)
# Note sort_mo by default take the 1-based orbital indices.
mo = mycas.sort_mo([2, 3, 6])
# PySCF CASCI reference energy
mycas.kernel(mo)

# reference energy with AS
# Li 1s, 2py, 2pz frozen
ucc = UCCSD(mol, active_space=(2, 3), aslst=[1, 2, 5])
ucc.kernel()
# UCC recovers
ucc.print_energy()

# HEA run

hea = HEA.ry(ucc.int1e, ucc.int2e, ucc.n_elec, ucc.e_core, 6, engine="tensornetwork", mapping="jordan-wigner")
hea.grad = "autodiff"

e_list = []
for i in range(10):
    hea.init_guess = np.random.randn(len(hea.init_guess))
    e = hea.kernel()
    e_list.append(e)

    hea.print_summary()

print(e_list)
