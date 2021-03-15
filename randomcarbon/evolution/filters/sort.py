from typing import List
from pymatgen.core.structure import Structure
from randomcarbon.evolution.core import Filter
from randomcarbon.run.ase import get_energy
from randomcarbon.utils.structure import get_property
from randomcarbon.utils.factory import Factory


class EnergySort(Filter):
    """
    A Filter to sort the structures based on their energy in ascending order.
    """

    def __init__(self, calculator: Factory,
                 constraints: list = None):
        self.calculator = calculator
        self.constraints = constraints

    def filter(self, structures: List[Structure]) -> List[Structure]:
        for s in structures:
            if not get_property(s, "energy"):
                get_energy(structure=s, calculator=self.calculator, constraints=self.constraints,
                           set_in_structure=True)

        structures = sorted(structures, key=lambda x: get_property(x, "energy"))

        return structures
