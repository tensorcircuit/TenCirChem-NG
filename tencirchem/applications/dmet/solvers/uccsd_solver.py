from tencirchem.applications.dmet.solvers.base import ElectronicStructureSolver
from tencirchem import UCCSD


class UCCSDSolver(ElectronicStructureSolver):
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
        super().__init__()
        self.ucc = None
        self.params = None

    def simulate(self, molecule, mean_field=None, init_guess=None):
        """Perform the simulation (energy calculation) for the molecule.

        If the mean field is not provided it is automatically calculated.
        `pyscf.ao2mo` is used to transform the AO integrals into
        MO integrals.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.

        Returns:
            float64: The Full CI energy (energy).
        """

        mo_coeff = None
        if mean_field:
            mo_coeff = mean_field.mo_coeff
        self.ucc = UCCSD(molecule, mo_coeff=mo_coeff, run_fci=False, run_ccsd=False)
        if init_guess is not None:
            self.ucc.init_guess = init_guess
        e = self.ucc.kernel()
        self.params = self.ucc.params
        return e

    def get_rdm(self):
        """Calculate the 1- and 2-particle RDMs.

        Calculate the Full CI reduced density matrices.

        Returns:
            (numpy.array, numpy.array): One & two-particle RDMs (fci_onerdm & fci_twordm, float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """

        if self.ucc.params is None:
            raise RuntimeError("Cannot retrieve RDM because no simulation has been run.")

        rdm1 = self.ucc.make_rdm1(basis="MO")
        rdm2 = self.ucc.make_rdm2(basis="MO")

        return rdm1, rdm2
