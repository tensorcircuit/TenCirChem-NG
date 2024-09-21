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

"""Perform CCSD calculation.

The electronic structure calculation employing the 
coupled-cluster singles and doubles (CCSD) method 
is done here.

"""

import warnings

from pyscf import cc, scf

from tencirchem.applications.dmet.solvers.base import ElectronicStructureSolver


class CCSDSolver(ElectronicStructureSolver):
    """Perform CCSD calculation.

    Uses the CCSD method to solve the electronic structure problem.
    PySCF program will be utilized.
    Users can also provide a function that takes a `pyscf.gto.Mole`
    as its first argument and `pyscf.scf.RHF` as its second.

    Attributes:
        cc_fragment (pyscf.cc.CCSD): The coupled-cluster object.
    """

    def __init__(self):
        self.cc_fragment = None
        self.params = None

    def simulate(self, molecule, mean_field=None, init_guess=None):
        """Perform the simulation (energy calculation) for the molecule.

        If the mean field is not provided it is automatically calculated.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.
            init_guess: The initial guess

        Returns:
            float64: CCSD energy (total_energy).
        """

        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()

        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn("CCSDSolver simulating with mean field not converged.", RuntimeWarning)

        # Execute CCSD calculation
        self.cc_fragment = cc.ccsd.CCSD(mean_field)
        self.cc_fragment.verbose = 0
        self.cc_fragment.conv_tol = 1e-9
        self.cc_fragment.conv_tol_normt = 1e-7
        if self.params is not None:
            t1, t2 = self.params
        else:
            t1 = t2 = None
        correlation_energy, t1, t2 = self.cc_fragment.ccsd(t1=t1, t2=t2)
        self.params = t1, t2
        scf_energy = mean_field.e_tot
        total_energy = scf_energy + correlation_energy

        return total_energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle RDMs.

        Calculate the CCSD reduced density matrices.
        The CCSD lambda equation will be solved for calculating
        the RDMs.

        Returns:
            (numpy.array, numpy.array): One & two-particle RDMs (cc_onerdm & cc_twordm, float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """

        # Check if CCSD calculation is performed
        if not self.cc_fragment:
            raise RuntimeError("Cannot retrieve RDM because no simulation has been run.")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        self.cc_fragment.solve_lambda()
        cc_onerdm = self.cc_fragment.make_rdm1()
        cc_twordm = self.cc_fragment.make_rdm2()

        return cc_onerdm, cc_twordm
