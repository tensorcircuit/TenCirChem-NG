from tencirchem import UCC, HEA
from tencirchem.molecule import h2


ucc = UCC(h2)
# OpenFermion fermion operator
print(type(ucc.h_fermion_op))
print(ucc.h_fermion_op)
# OpenFermion qubit operator
print(type(ucc.h_qubit_op))
print(ucc.h_qubit_op)


hea = HEA.from_molecule(h2)
# No fermion operator for the `HEA` class
# OpenFermion qubit operator
# by default performs parity transformation
print(type(hea.h_qubit_op))
print(hea.h_qubit_op)
