import numpy as np
from scipy.optimize import minimize

from tencirchem import UCCSD, HEA
from tencirchem.molecule import h4
from tencirchem.utils.optimizer import soap


# see also `test_optimizer.py`
ucc = UCCSD(h4)
ucc.kernel()

opt_res = minimize(ucc.energy, ucc.init_guess, method=soap)
print(opt_res.fun)
assert np.allclose(opt_res.fun, ucc.e_ucc)


opt_res = minimize(ucc.energy, ucc.init_guess, method="cobyla")
print(opt_res.fun)
assert np.allclose(opt_res.fun, ucc.e_ucc)

# not ideal without gradients and starting from random guess
hea = HEA.from_molecule(h4)
opt_res = minimize(hea.energy, hea.init_guess, method="Powell")
print(opt_res.fun)
