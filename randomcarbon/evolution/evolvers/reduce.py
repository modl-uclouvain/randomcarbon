import random
import itertools
import numpy as np
from typing import List, Union
from randomcarbon.evolution.core import Evolver, Condition
from randomcarbon.utils.symmetry import get_equivalent_indices_grouped
from randomcarbon.utils.structure import structure_from_symmetries
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp


class RemoveAtoms(Evolver):
    """
    Evolver that removes one or more inequivalent atoms.
    Works on the symmetrized structures and when one atom is
    removed all his symmetrically equivalent atoms are removed as well.
    """

    def __init__(self, symprec: float = 0.01, num_atoms: int = 1, symm_ops: List[SymmOp] = None,
                 conditions: List[Condition] = None):
        """
        Args:
            symprec: the symprec value used in spglib to determine the
                spacegroup of the system and the symmetry equivalent atoms.
            num_atoms: the number of inequivalent atoms to be removed.
            symm_ops: list of SymmOp that will be used to determine the list of inequivalent
                atoms. If None the SpacegroupAnalyzer will be used.
        """
        super().__init__(conditions)
        self.symprec = symprec
        self.num_atoms = num_atoms
        self.symm_ops = symm_ops

    def _evolve(self, structure: Structure) -> List[Structure]:
        inequivalent_sites = get_equivalent_indices_grouped(structure, symprec=self.symprec,
                                                            symm_ops=self.symm_ops)
        n_inequivalent_sites = len(inequivalent_sites)
        # check that it is not removing all the atoms
        if n_inequivalent_sites < self.num_atoms + 1:
            return []

        new_indices = random.sample(inequivalent_sites, n_inequivalent_sites - self.num_atoms)
        new_sites = [structure[i] for i in itertools.chain(*new_indices)]
        new_structure = Structure.from_sites(new_sites)

        return [new_structure]


class MergeAtoms(Evolver):
    """
    Evolver that merges equivalent or inequivalent atoms.
    The symmetry should be identified by the SpaceGroupAnalyzer.
    It assumes that atoms are all of the same specie.
    The set of atoms to be removed can be from a random sphere
    in the cell or from a sphere centered at one of the atoms.
    """

    def __init__(self, symprec: float = 0.01, num_atoms: int = 2,
                 max_num_atoms: int = None, r: float = 1.6, atom_centered=True,
                 max_tests: int = 100, symm_ops: List[SymmOp] = None,
                 spacegroup: Union[str, int] = None, conditions: List[Condition] = None):
        """
        Args:
            symprec: the symprec value used to determine the equivalent atoms.
            num_atoms: the number of atoms to be merged. if None all the atoms in
                the sphere. If defined should be larger than 1.
            max_num_atoms: if not None, the number of atoms to be merged would
                be a random number between num_atoms and max_num_atoms.
                Should be strictly larger than num_atoms.
            r: the radius of the sphere that defines the list of atoms to be merged.
            atom_centered: if True the center of the sphere will be a randomly selected
                atom in the structure, otherwise a random sphere in the cell.
            max_tests: the number of maximum test to try to get a sphere with at least
                num_atoms. Only used if atom_centered is False.
            symm_ops: list of SymmOp that will be used to determine the list of inequivalent
                atoms and to reconstruct the full structure. If None the SpacegroupAnalyzer
                will be used.
            spacegroup: number of the spacegroup to reconstruct the structure if symm_ops
                is not defined. If None the value will be determined by the SpacegroupAnalyzer.

        """
        super().__init__(conditions)
        self.symprec = symprec
        if num_atoms is not None:
            if num_atoms < 2:
                raise ValueError("num_atoms should be larger than 2")
            if max_num_atoms is not None and max_num_atoms <= num_atoms:
                raise ValueError("max_num_atoms should be larger than num_atoms")
        self.num_atoms = num_atoms
        self.max_num_atoms = max_num_atoms
        self.r = r
        self.atom_centered = atom_centered
        self.max_tests = max_tests
        self.symm_ops = symm_ops
        self.spacegroup = spacegroup

    def _evolve(self, structure: Structure) -> List[Structure]:
        latt = structure.lattice
        equivalent_indices = get_equivalent_indices_grouped(structure, symm_ops=self.symm_ops, symprec=self.symprec)

        min_num_atoms = 2 if self.num_atoms is None else self.num_atoms
        if self.atom_centered:
            site = random.choice(structure)
            atoms = structure.get_neighbors(site, self.r)
            if len(atoms) < min_num_atoms:
                return []
        else:
            # select one position that contains at least num_atoms
            for _ in range(self.max_tests):
                center_coords = latt.get_cartesian_coords(np.random.uniform(0., 1., size=3))
                atoms = structure.get_sites_in_sphere(center_coords, self.r)
                if len(atoms) >= min_num_atoms:
                    break
            else:
                return []

        if self.num_atoms and len(atoms) != self.num_atoms:
            num_atoms = self.num_atoms
            if self.max_num_atoms is not None:
                max_num_atoms = min(self.max_num_atoms, len(atoms))
                num_atoms = random.randint(num_atoms, max_num_atoms)

            # Find sites to be merged and their indices in the structure.
            # First select the images and then add the initial site, that does not
            # have the index as attribute.
            to_be_merged = random.sample(atoms, num_atoms)
        else:
            to_be_merged = atoms

        new_inequivalent_sites = []
        for indices_list in equivalent_indices:
            if not any(s.index in indices_list for s in to_be_merged):
                new_inequivalent_sites.append(structure[indices_list[0]])

        # NB here it is assumed that the atoms are all of the same specie
        species = [site.specie for site in new_inequivalent_sites] + [to_be_merged[0].specie]
        eq_fcoords = np.concatenate([[s.frac_coords] for s in to_be_merged])
        new_coords = np.mean(eq_fcoords, axis=0)
        frac_coords = [site.frac_coords for site in new_inequivalent_sites] + [new_coords]
        spacegroup = self.spacegroup
        if self.symm_ops is None and spacegroup is None:
            spga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            spacegroup = spga.get_space_group_number()
        new_structure = structure_from_symmetries(latt, species, frac_coords, symm_ops=self.symm_ops,
                                                  spacegroup=spacegroup, coords_are_cartesian=False)

        return [new_structure]
