from pyscf import M
from tencirchem import ROUCCSD

m = M(atom=[["H", 0, 0, 0], ["He", 0, 0, 1]], spin=1, basis="631g")
uccsd = ROUCCSD(m)
uccsd.kernel()
uccsd.print_summary()
