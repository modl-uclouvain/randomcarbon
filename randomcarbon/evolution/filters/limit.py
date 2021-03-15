"""
module for Filters that limit the number of structures.
"""

from typing import List
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from randomcarbon.evolution.core import Filter
from randomcarbon.run.ase import get_energy
from randomcarbon.utils.structure import get_property
from randomcarbon.utils.factory import Factory


class StructuresMaximumNumber(Filter):
    """
    A Filter that just returns the first N structures.
    """

    def __init__(self, num_structures):
        self.num_structures = num_structures

    def filter(self, structures: List[Structure]) -> List[Structure]:
        return structures[:self.num_structures]


class MaxEnergyPerAtom(Filter):
    """
    A filter that excludes structures with energy above a selected threshold.
    """

    def __init__(self, calculator: Factory,
                 constraints: list = None, max_energy: float = 0):
        self.calculator = calculator
        self.constraints = constraints
        self.max_energy = max_energy

    def filter(self, structures: List[Structure]) -> List[Structure]:
        filtered_structures = []
        for s in structures:
            e = get_property(s, "energy")
            if not e:
                e = get_energy(structure=s, calculator=self.calculator, constraints=self.constraints,
                               set_in_structure=True)
            if e <= self.max_energy:
                filtered_structures.append(s)

        return filtered_structures


class MatchingStructures(Filter):
    """
    Removes redundant structures that match according to a StructureMatcher.
    """

    def __init__(self, structure_matcher: StructureMatcher):
        self.structure_matcher = structure_matcher

    def filter(self, structures: List[Structure]) -> List[Structure]:
        if not structures:
            return []
        filtered_structures = [l[0] for l in self.structure_matcher.group_structures(structures)]

        return filtered_structures
