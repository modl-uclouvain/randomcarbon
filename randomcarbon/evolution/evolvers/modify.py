import random
import numpy as np
from typing import List, Union
from randomcarbon.evolution.core import Evolver, Condition
from randomcarbon.utils.symmetry import get_inequivalent_site_representative
from randomcarbon.utils.structure import structure_from_symmetries
from randomcarbon.utils.random import random_point_on_a_sphere
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp


class MoveAtoms(Evolver):
    """
    Evolver that moves one inequivalent atom in a random direction by a random.
    Works on the symmetrized structures and when one atom is
    moved all his symmetrically equivalent atoms are moved as well.
    """

    def __init__(self, symprec: float = 0.01, num_atoms: int = 1,
                 min_displ: float = 0.5, max_displ: float = 3, symm_ops: List[SymmOp] = None,
                 spacegroup: Union[str, int] = None, conditions: List[Condition] = None):
        """
        Args:
            symprec: the symprec value used to determine the equivalent atoms.
            num_atoms: the number of inequivalent atoms to be moved.
            max_displ: the minimum displacement allowed.
            max_displ: the maximum displacement allowed.
            symm_ops: list of SymmOp that will be used to determine the list of inequivalent
                atoms and to reconstruct the full structure. If None the SpacegroupAnalyzer
                will be used.
            spacegroup: number of the spacegroup to reconstruct the structure if symm_ops
                is not defined. If None the value will be determined by the SpacegroupAnalyzer.

        """
        super().__init__(conditions)
        self.symprec = symprec
        self.num_atoms = num_atoms
        self.min_displ = min_displ
        self.max_displ = max_displ
        self.symm_ops = symm_ops
        self.spacegroup = spacegroup

    def _evolve(self, structure: Structure) -> List[Structure]:
        inequivalent_sites = get_inequivalent_site_representative(structure, symm_ops=self.symm_ops,
                                                                  symprec=self.symprec, spgn=self.spacegroup)
        num_atoms = min(len(inequivalent_sites), self.num_atoms)
        random.shuffle(inequivalent_sites)

        moved = inequivalent_sites[:num_atoms]
        fixed = inequivalent_sites[num_atoms:]

        species = [s.specie for s in moved + fixed]
        coords = [s.coords + random_point_on_a_sphere(1) * random.uniform(self.min_displ, self.max_displ) for s in moved]
        coords += [s.coords for s in fixed]

        spacegroup = self.spacegroup
        if self.symm_ops is None and spacegroup is None:
            spga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            spacegroup = spga.get_space_group_number()

        new_structure = structure_from_symmetries(structure.lattice, species, coords, symm_ops=self.symm_ops,
                                                  spacegroup=spacegroup, coords_are_cartesian=True)
        return [new_structure]
