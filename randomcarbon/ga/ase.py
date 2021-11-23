"""
tools to run genetic algorithm with ASE
"""
from typing import Union, List
import random
import numpy as np
from pymatgen.core import SymmOp, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.ga.offspring_creator import OffspringCreator
from randomcarbon.evolution.core import Evolver
from randomcarbon.utils.symmetry import get_inequivalent_site_representative
from randomcarbon.utils.structure import structure_from_symmetries
from scipy.optimize import linear_sum_assignment


class EvolverWrapper(OffspringCreator):
    """
    An object to mutate a structure based on the given Evolver.
    """

    def __init__(self, evolver: Evolver, max_tests: int = 1, min_dist: float = 0.8,
                 verbose=False, num_muts=1, rng=np.random):
        """

        Args:
            evolver: an Evolver used to mutate the incoming structures.
            max_tests: the maximum number of trials to generate a new structure. It may fail due
                to the evolver or to the produced structure not satisfying the min_dist.
            min_dist: minimum distance among atoms. Below this threshold the generated structure
                will be rejected.
            verbose:
            num_muts:
            rng:
        """
        super().__init__(verbose=verbose, num_muts=num_muts, rng=rng)
        self.descriptor = 'Wrapper - ' + evolver.__class__.__name__
        self.evolver = evolver
        self.max_tests = max_tests
        self.min_dist = min_dist
        self._aaa = AseAtomsAdaptor()

    def get_new_individual(self, parents):
        f = parents[0]
        s = self._aaa.get_structure(f)
        for _ in range(self.max_tests):
            evolved_structures = self.evolver.evolve(s)

            if evolved_structures:
                dm = evolved_structures[0].distance_matrix
                if dm[dm > 1e-10].min() > self.min_dist:
                    break
        else:
            return None, 'mutation: {}'.format(self.descriptor)

        indi = self._aaa.get_atoms(evolved_structures[0])

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return (self.finalize_individual(indi),
                'mutation: {}'.format(self.descriptor))


class MixInequivalent(OffspringCreator):
    """
    An object to mix two structures having the same symmetries.
    Reduces the structures to their inequivalent atoms, mixes these atoms and
    uses the symmetries to generate the full mixed structure.
    The symmetry operations should be preferably passed in order to work with the
    inequivalent atoms. Alternatively the spacegroup number can be given but is
    likely to be less reliable.
    """

    def __init__(self, symprec: float = 0.01,
                 max_reduction: int = 2, max_grow: int = 2, symm_ops: List[SymmOp] = None,
                 spacegroup: Union[str, int] = None, max_tests: int = 100,
                 min_dist: float = 0.8, verbose=False, num_muts=1, rng=np.random):
        """

        Args:
            symprec: tolerance on the identification of symmetrically equivalent atoms.
            max_reduction: the final number of inequivalent atoms cannot be below the minimum
                of the two incoming structures minus this value.
            max_grow: the final number of inequivalent atoms cannot be above the maximum
                of the two incoming structures plus this value.
            symm_ops: the list of symmetry operations.
            spacegroup: the space group.
            max_tests: the maximum number of trials to generate a new structure.
            min_dist: minimum distance among atoms. Below this threshold the generated structure
                will be rejected.
            verbose:
            num_muts:
            rng:
        """
        super().__init__(verbose=verbose, num_muts=num_muts, rng=rng)
        self.descriptor = "MixInequivalent"
        self.symprec = symprec
        self.max_reduction = max_reduction
        self.max_grow = max_grow
        self.symm_ops = symm_ops
        self.spacegroup = spacegroup
        self.max_tests = max_tests
        self.min_dist = min_dist
        self._aaa = AseAtomsAdaptor()

    def get_new_individual(self, parents):
        a1, a2 = parents
        s1 = self._aaa.get_structure(a1)
        s2 = self._aaa.get_structure(a2)
        ineq1 = get_inequivalent_site_representative(s1, symprec=self.symprec, symm_ops=self.symm_ops,
                                                     spgn=self.spacegroup)
        ineq2 = get_inequivalent_site_representative(s2, symprec=self.symprec, symm_ops=self.symm_ops,
                                                     spgn=self.spacegroup)

        n1 = len(ineq1)
        n2 = len(ineq2)
        # determine the range of inequivalent atoms that can be taken into account
        # at least two atoms, otherwise it is not mixing anything
        min_ineq = max(min(n1, n2) - self.max_reduction, 2)
        max_ineq = max(n1, n2) + min(self.max_grow, n1, n2)

        for _ in range(self.max_tests):
            # choose a number of inequivalent atoms for the final mix
            n_final = random.randint(min_ineq, max_ineq)

            # select the number of atoms to be picked from each of the parents
            # take at least one atom from each parent. make sure that the number
            # requires more atoms than those available in one of the two parents
            nn1 = random.randint(max(1, n_final - n2), min(n_final - 1, n1))
            nn2 = n_final - nn1

            sites = random.sample(ineq1, nn1) + random.sample(ineq2, nn2)

            frac_coords = [site.frac_coords for site in sites]
            species = [site.specie for site in sites]

            new_structure = structure_from_symmetries(s1.lattice, species, frac_coords, symm_ops=self.symm_ops,
                                                      spacegroup=self.spacegroup, coords_are_cartesian=False)

            # accept if the new atoms in the structure are not too close
            dm = new_structure.distance_matrix
            if dm[dm>1e-10].min() > self.min_dist:
                indi = self._aaa.get_atoms(new_structure)
                break

        else:
            return None, 'pairing: {}'.format(self.descriptor)

        indi = self.initialize_individual(a1, indi)
        indi.info['data']['parents'] = [a1.info['confid'], a2.info['confid']]

        return (self.finalize_individual(indi),
                'pairing: {}'.format(self.descriptor))


class NumAtomsComparator:
    """
    Compares two structures based on the number of atoms.
    """

    def looks_like(self, a1, a2):
        return len(a1) == len(a2)


class LinearSumComparator:
    """
    Compares two structures based on the scipy linear_sum_assignment on the
    pairwise distances. If the maximum distance is below the selected
    threshold the two structures are considered identical.
    """

    def __init__(self, lattice: Lattice, max_dist: float = 0.1):
        """
        Args:
            lattice: the lattice of the atoms
            max_dist: the threshold on the pairwise distances to consider two structures
                equivalent.
        """
        self.lattice = lattice
        self.max_dist = max_dist

    def looks_like(self, a1, a2):
        f1 = self.lattice.get_fractional_coords(a1.get_positions())
        f2 = self.lattice.get_fractional_coords(a2.get_positions())
        dist = self.lattice.get_all_distances(f1, f2)
        row_ind, col_ind = linear_sum_assignment(dist)
        maxdist = dist[row_ind, col_ind].max()
        return maxdist < self.max_dist


class EnergyPerAtomComparator:
    """Compares the energy per atom of the supplied atoms objects using
       get_potential_energy().

       Parameters:

       dE: the difference in energy below which two energies per atom are
       deemed equal.
    """
    def __init__(self, dE=0.02):
        self.dE = dE

    def looks_like(self, a1, a2):
        dE = abs(a1.get_potential_energy()/len(a1) - a2.get_potential_energy()/len(a2))
        if dE >= self.dE:
            return False
        else:
            return True
