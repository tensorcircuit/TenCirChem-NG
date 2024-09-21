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

import abc


class ElectronicStructureSolver(abc.ABC):
    """Sets interface for objects that perform energy estimation of a molecule.

    Specifics vary between concrete implementations, but common to all of them
    is that after `simulate` is called, the right internal state is set for the
    class so that `get_rdm` can return the reduced density matrix for the last
    run simulation.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def simulate(self, molecule, mean_field=None, init_guess=None):
        """Performs the simulation for a molecule.

        The mean field is calculated automatically if not passed in by the
        calling code.

        Args:
            molecule (pyscf.gto.Mole): The molecule on which to perform the
                simluation.
            mean_field (pyscf.scf.RHF): The mean filed of the molecule. Computed
                automatically if `None` is passed in.
            init_guess: The initial guess
        """
        pass

    @abc.abstractmethod
    def get_rdm(self):
        """Returns the RDM for the previous simulation.

        In a concrete implementation, the `simulate` function would set the
        necessary internal state for the class so that this function can return
        the reduced density matrix.

        Returns:
            (numpy.array, numpy.array): The one- and two-particle RDMs (float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """
        pass
