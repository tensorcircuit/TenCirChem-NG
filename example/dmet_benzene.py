from tencirchem import UCCSD
from tencirchem.applications.dmet import DMET, UCCSDSolver
from tencirchem.molecule import benzene

mol = benzene()
for atom in mol.atom:
    print(atom)

# reference energies
hf = mol.HF()
hf.kernel()

ccsd = hf.CCSD()
ccsd.kernel()

solver = UCCSDSolver()

dmet = DMET()
dmet.electronic_structure_solver = solver
dmet.verbose = True

# `fragment_atoms` for the number of atoms in each fragment
# `frag_ids` indicates identical fragments and saves computational cost
# Could be much faster using GPU
energy = dmet.simulate(mol, fragment_atoms=[1] * 12, frag_ids=[1, 2] * 6)
# should be something like -227.917113741135
print(energy)
