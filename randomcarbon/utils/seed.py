"""
experimental features to randomly extract substructures from previous existing structures,
to be used as starting points. From Jeremie Pirard.
"""
import random
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor


def extract_random_seed(structure: Structure, number: int = 4) -> Structure:
    """
    Function that extracts a given number of atoms SpacegroupAse taken randomly
    in a structure and returns a new one with those atoms
    Args:
        structure: a pymatgen Structure
        number: number of atoms to be extracted from the structure

    Returns:
        the Structure containing the extracted atoms
    """

    seedlist = random.sample(range(0, len(structure) - 1), number)
    return Structure.from_sites(sites=[structure[i] for i in seedlist])


def extract_chain(seed: Structure, lring: int = 6, cut_rad: float = 2) -> Structure:
    """
    Function that extracts a chain of neighbouring atoms (seperated away from each
    other by the value given by cut_rad). The chain is extracted randomly and has
    a lenght of lring atoms.
    Args:
        seed: a pymatgen Structure on which to extract chain
        lring: number of atoms in the chain
        cut_rad: maximum distance between atoms in the chain
    Returns:
        the Structure containing the chain

    """
    nodelist = []
    nodelist.append(random.randint(0, len(seed) - 1))
    site_fcoords = seed.frac_coords
    for i in range(1, lring):

        index = nodelist[i - 1]
        _, _, indices, _ = seed.lattice.get_points_in_sphere_py(site_fcoords,
                                                                center=seed[index].coords, r=cut_rad, zip_results=False)
        if all(elem in nodelist for elem in indices):
            return extract_chain(seed=seed, lring=6, cut_rad=cut_rad + 1)
        for j in indices:

            node = j
            if node not in nodelist:
                nodelist.append(node)
                break

    return Structure.from_sites([seed[i] for i in nodelist])


def extract_sym_seed(seed: Structure, template: Structure, spacegroup: int = None, lring: int = 6,
                     cut_rad: float = 2, temp_dist: float = 2, max_tests: int = 500, merge_rad: float = 1.2):
    """
    Function that will extract a part of a Structure, and returned a symmetrized version of it,
    based on the spacegroup provided. If the spacegroup is not provided, the spacegroup of the template will be used.
    It will check that the structure is far enough from the template
    Args:
        seed: the structure from which the atoms are to be extracted
        template: the zeolite template
        spacegroup: the spacegroup used to symmetrize the seed
        lring: number of atoms to extract
        cut_rad: radius treshold to merge atoms close to each other
        temp_dist: distance to the template
    Returns:
        A structure symmetrized taken from the seed
        None if no structure far enough from the zeolite has been found
    """

    if spacegroup is None:
        spacegroup = SpacegroupAnalyzer(template).get_space_group_number()
    for _ in range(0, max_tests):
        chain = extract_chain(seed=seed, lring=lring, cut_rad=cut_rad)
        test = Structure.from_spacegroup(sg=spacegroup, lattice=chain.lattice, species=chain.species,
                                         coords=chain.frac_coords)
        test.merge_sites(merge_rad, "average")
        dist = get_struc_min_dist(test, template)
        if dist <= temp_dist:
            continue
        return test


def extract_best_sym_seed(seed: Structure, template: Structure, spacegroup: int = None, lring: int = 6,
                          cut_rad: float = 2, temp_dist: float = 2, max_tests: int = 10):
    """
    Function that will extract a part of a Structure a repeated number of times, and returned the best symmetrized
    version of it (based on the energy), based on the spacegroup provided. If the spacegroup is not provided,
    the spacegroup of the template will be used. It will check that the structure is far enough from the template
    Args:
        seed: the structure from which the atoms are to be extracted
        template: the zeolite template
        spacegroup: the spacegroup used to symmetrize the seed
        lring: number of atoms to extract
        cut_rad: radius treshold to merge atoms close to each other
        temp_dist: distance to the template
        max_tests: number of chains to be generated in order to get the best one
    Returns:
        A structure symmetrized taken from the seed
        None if no structure far enough from the zeolite has been found
    """

    if spacegroup is None:
        spacegroup = SpacegroupAnalyzer(template).get_space_group_number()
    tests = []
    energies = []
    from ase.calculators.kim.kim import KIM
    for _ in range(0, max_tests):
        test = extract_sym_seed(seed, template, spacegroup, lring, cut_rad, temp_dist)
        tests.append(test)

        if test is not None:
            atoms = AseAtomsAdaptor().get_atoms(test)
            atoms.calc = KIM("Tersoff_LAMMPS_Tersoff_1989_SiC__MO_171585019474_002")
            energy = atoms.get_potential_energy() / atoms.get_global_number_of_atoms()
            energies.append(energy)
    if all(i == energies[0] for i in energies):
        print("No good structure generated")
        return None
    return tests[np.argmin(np.array(energies))], energies[np.argmin(np.array(energies))]
