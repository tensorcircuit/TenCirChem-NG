import numpy as np
from scipy.optimize import minimize

from tencirchem import UCCSD
from tencirchem.molecule import h4
from tencirchem.utils.optimizer import soap


# see also `test_optimizer.py`
ucc = UCCSD(h4)
ucc.kernel()

opt_res = minimize(ucc.energy, ucc.init_guess, method=soap)
assert np.allclose(opt_res.fun, ucc.e_ucc)


opt_res = minimize(ucc.energy, ucc.init_guess, method="cobyla")
assert np.allclose(opt_res.fun, ucc.e_ucc)
