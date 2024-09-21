#   Copyright 2019 1QBit
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Perform Full CI calculation.

The electronic structure calculation employing the 
full configuration interaction (CI) method 
is done here.

"""

from functools import reduce
import warnings

import numpy as np
from pyscf import ao2mo, fci, scf

from tencirchem.applications.dmet.solvers.base import ElectronicStructureSolver


class FCISolver(ElectronicStructureSolver):
    """Perform Full CI calculation.

    Uses the Full CI method to solve the electronic structure problem.
    PySCF program will be utilized.
    Users can also provide a function that takes a `pyscf.gto.Mole`
    as its first argument and `pyscf.scf.RHF` as its second.

    Attributes:
        cisolver (pyscf.fci.direct_spin0.FCI): The Full CI object.
        ci (numpy.array): The CI wavefunction (float64).
        norb (int): The number of molecular orbitals.
        nelec (int): The number of electrons.
    """

    def __init__(self):
        self.ci = None
        self.norb = None
        self.nelec = None
        self.cisolver = None

    def simulate(self, molecule, mean_field=None, init_guess=None):
        """Perform the simulation (energy calculation) for the molecule.

        If the mean field is not provided it is automatically calculated.
        `pyscf.ao2mo` is used to transform the AO integrals into
        MO integrals.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.
            init_guess: The initial guess


        Returns:
            float64: The Full CI energy (energy).
        """

        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()

        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn("FCISolver simulating with mean field not converged.", RuntimeWarning)

        h1 = reduce(np.dot, (mean_field.mo_coeff.T, mean_field.get_hcore(), mean_field.mo_coeff))
        twoint = mean_field._eri
        self.norb = len(mean_field.mo_energy)
        eri = ao2mo.restore(8, twoint, self.norb)
        eri = ao2mo.incore.full(eri, mean_field.mo_coeff)
        eri = ao2mo.restore(1, eri, self.norb)
        self.cisolver = fci.direct_spin0.FCI(molecule)
        self.cisolver.verbose = 0
        self.nelec = molecule.nelectron
        energy, self.ci = self.cisolver.kernel(
            h1, eri, h1.shape[1], self.nelec, ecore=molecule.energy_nuc(), ci0=init_guess
        )

        return energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle RDMs.

        Calculate the Full CI reduced density matrices.

        Returns:
            (numpy.array, numpy.array): One & two-particle RDMs (fci_onerdm & fci_twordm, float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """

        # Check if Full CI is performed
        if not self.norb or not self.nelec:
            raise RuntimeError("Cannot retrieve RDM because no simulation has been run.")

        fci_onerdm = self.cisolver.make_rdm1(self.ci, self.norb, self.nelec)
        fci_twordm = self.cisolver.make_rdm2(self.ci, self.norb, self.nelec)

        return fci_onerdm, fci_twordm

    @property
    def params(self):
        return self.ci
