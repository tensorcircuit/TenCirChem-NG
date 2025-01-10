import numpy as np
import pytest

from tencirchem.molecule import h4
from tencirchem import UCCSD
from tencirchem.applications.dmet import DMET, FCISolver, CCSDSolver, UCCSDSolver, PUCCDSolver


@pytest.mark.parametrize("solver", [FCISolver, CCSDSolver, UCCSDSolver, PUCCDSolver])
def test_classical_solver(solver):
    mol = h4

    # print reference energies
    ucc = UCCSD(mol)
    ucc.print_energy()

    solver = solver()

    dmet = DMET()
    dmet.electronic_structure_solver = solver
    dmet.verbose = True

    energy = dmet.simulate(mol, [1, 1, 1, 1], frag_ids=[1, 2, 2, 1])
    # dmet is not exact, so tolerant 1e-2 error
    assert np.allclose(energy, ucc.e_fci, atol=1e-2)
