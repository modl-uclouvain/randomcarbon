import random
import numpy as np
from typing import List
from randomcarbon.evolution.core import Evolver, Condition
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class RemoveAtoms(Evolver):
    """
    Evolver that removes one or more inequivalent atoms.
    Works on the symmetrized structures and when one atom is
    removed all his symmetrically equivalent atoms are removed as well.
    The symmetry should be identified by the SpaceGroupAnalyzed
    """

    def __init__(self, symprec: float = 0.01, num_atoms: int = 1, conditions: List[Condition] = None):
        """
        Args:
            symprec: the symprec value used in spglib to determine the
                spacegroup of the system and the symmetry equivalent atoms.
            num_atoms: the number of inequivalent atoms to be removed.
        """
        super().__init__(conditions)
        self.symprec = symprec
        self.num_atoms = num_atoms

    def _evolve(self, structure: Structure) -> List[Structure]:
        spga = SpacegroupAnalyzer(structure, symprec=self.symprec)
        sym_structure = spga.get_symmetrized_structure()
        inequivalent_sites = [l[0] for l in sym_structure.equivalent_sites]
        n_inequivalent_sites = len(inequivalent_sites)
        # check that it is not removing all the atoms
        if n_inequivalent_sites < self.num_atoms + 1:
            return []

        new_sites = random.sample(inequivalent_sites, n_inequivalent_sites - self.num_atoms)
        spg_num = spga.get_space_group_number()
        species = [site.specie for site in new_sites]
        frac_coords = [site.frac_coords for site in new_sites]
        new_structure = Structure.from_spacegroup(spg_num, structure.lattice, species, frac_coords,
                                                  coords_are_cartesian=False)

        return [new_structure]


class MergeAtoms(Evolver):
    """
    Evolver that merges equivalent or inequivalent atoms.
    The symmetry should be identified by the SpaceGroupAnalyzed.
    It assumes that atoms are all of the same specie.
    The set of atoms to be removed can be from a random sphere
    in the cell or from a sphere centered at one of the atoms.
    """

    def __init__(self, symprec: float = 0.01, num_atoms: int = 2,
                 max_num_atoms: int = None, r: float = 1.6,
                 atom_centered=True, max_tests: int = 100,
                 conditions: List[Condition] = None):
        """
        Args:
            symprec: the symprec value used in spglib to determine the
                spacegroup of the system and the symmetry equivalent atoms.
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

    def _evolve(self, structure: Structure) -> List[Structure]:
        spga = SpacegroupAnalyzer(structure, symprec=self.symprec)
        sym_structure = spga.get_symmetrized_structure()
        latt = sym_structure.lattice
        equivalent_indices = sym_structure.equivalent_indices

        min_num_atoms = 2 if self.num_atoms is None else self.num_atoms
        if self.atom_centered:
            site = random.choice(sym_structure)
            center_coords = site.cart_coords
            atoms = sym_structure.get_neighbors(sym_structure.frac_coords, center_coords, self.r)
            if len(atoms) < min_num_atoms:
                return []
        else:
            # select one position that contains at least num_atoms
            for _ in range(self.max_tests):
                center_coords = latt.get_cartesian_coords(np.random.uniform(0., 1., size=3))
                atoms = sym_structure.get_neighbors(sym_structure.frac_coords, center_coords, self.r)
                if atoms >= min_num_atoms:
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
                new_inequivalent_sites.append(sym_structure[indices_list[0]])

        spg_num = spga.get_space_group_number()
        # NB here it is assumed that the atoms are all of the same specie
        species = [site.specie for site in new_inequivalent_sites] + [to_be_merged[0].specie]
        eq_fcoords = np.concatenate([[s.frac_coords] for s in to_be_merged])
        new_coords = np.mean(eq_fcoords, axis=0)
        frac_coords = [site.frac_coords for site in new_inequivalent_sites] + [new_coords]
        new_structure = Structure.from_spacegroup(spg_num, latt, species, frac_coords,
                                                  coords_are_cartesian=False)

        return [new_structure]
