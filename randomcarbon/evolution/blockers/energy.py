import logging
from typing import Optional
from randomcarbon.utils.structure import get_property
from randomcarbon.evolution.core import Blocker
from randomcarbon.utils.factory import Factory
from randomcarbon.run.ase import get_energy
from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


class EnergyAtoms(Blocker):
    """
    Blocks the execution based on the energy per atom and number of atoms
    in the structure. Given a dictionary with with number of atoms as keys
    and energy per atom as values stops if the structure a structure having more
    atoms than the key have an energy higher than the value.
    Example:
        criteria = {100: -1.5, 200: -2}
        A structure with 140 atoms and an energy of -1.3 is blocked.
        A structure with 50 atoms and an energy of -0.1 is not blocked
        A structure with 150 atoms and an energy of -1.8 is not blocked
    """

    def __init__(self, criteria: dict, calculator: Factory,
                 constraints: list = None):
        self.criteria = criteria
        self.calculator = calculator
        self.constraints = constraints

    def block(self, structure: Structure) -> Optional[str]:
        e = get_property(structure, "energy")
        if not e:
            e = get_energy(structure=structure, calculator=self.calculator, constraints=self.constraints,
                           set_in_structure=True)
        nsites = len(structure)

        for n_threshold, e_threshold in self.criteria.items():
            if nsites > n_threshold and e > e_threshold:
                return f"{self.__class__.__name__}. nsites: {nsites}, energy: {e}"

        return None
