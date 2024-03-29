from itertools import chain
import uuid
import logging
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Element, Species, DummySpecies, Composition
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.groups import in_array_list
from pymatgen.util.typing import VectorLike as ArrayLike
from typing import List, Union, Optional, Any, Tuple, Sequence, Callable
from randomcarbon.utils.bonding import get_undercoordinated
from randomcarbon.utils.random import random_point_on_a_sphere

logger = logging.getLogger(__name__)


def get_min_dist(
        frac_coords: Union[ArrayLike, List[ArrayLike]],
        structure: Structure
) -> float:
    """
    Calculates the minimum distance among the fractional coordinates
    and the atoms of a structure.

    Args:
        frac_coords: a list of fractional coordinates.
        structure: a Structure.

    Returns:
        float: minimum distance between the fractional coordinates and
            the atoms of a structure.
    """
    latt = structure.lattice
    distances = latt.get_all_distances(frac_coords, structure.frac_coords)
    return np.min(distances)


def get_struc_min_dist(structure1: Structure, structure2: Structure) -> float:
    """
    Gets the minimum distance between the atoms of two structures.
    Assumes that the two structures have the same lattice. Unexpected
    results if this constraint is not respected.

    Args:
        structure1: a Structure.
        structure2: a Structure.

    Returns:
        float: minimum distance among all the atoms of the two structures
    """
    return get_min_dist(structure1.frac_coords, structure2)

        
def remove_symmetrized_atom(structure: Structure, symprec: float = 0.001, angle_tolerance: float = 5.0) -> Structure:
    """
    Remove a randomly chosen atom and all the symmetry-equivalent atoms, based on the provided spacegroup. If not provided, the 
    spacegroup of the template will be used.

    Args:
        structure: the Structure on which atoms have to be removed
        symprec: the precision in detecting the spacegroup of the structure
        angle_tolerance: tolerance on angle symmetry

    Returns:
        A structure with atoms removed
    """
    spga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    ssym = spga.get_symmetrized_structure()
    indices_to_remove = random.choice(ssym.equivalent_indices)
    new_s = Structure.from_sites([site for i, site in enumerate(ssym) if i not in indices_to_remove])
    return new_s


def add_new_symmetrized_atom_func(template: Structure, postion_generator: Callable, spacegroup: Union[str, int] = None,
                                  current: Structure = None, supergroup_transf: Tuple[List, List] = None,
                                  symm_ops: List[SymmOp] = None, specie: str = "C", max_dist_eq: float = 0.1,
                                  min_dist_current: float = 1.2, max_dist_current: float = 1.6,
                                  min_dist_from_template: float = 3, max_dist_from_template: float = None, max_tests: int = 1000,
                                  symprec: float = 0.001, angle_tolerance: float = 5.0, return_single: bool = False) -> Optional[Structure]:
    """
    Add a new atom to the structure along with all its symmetrically equivalent atoms.
    The symmetric atoms can be determined based on the spacegroup or a list of symmetry operations.
    The latter has the precedence and if none is given the list of operations will be extracted from
    the template with spglib.
    The atoms are added in such a way to be away from a determined template and within a range of
    distanced with other existing atoms. The position of the atom is generated randomly and a maximum
    number of tests will be performed to try to generate a new atoms that satisfies all the required
    constraints. If this is not possible returns None. If current is not defined a new structure is generated.

    Args:
        template: a pymatgen Structure defining the template.
        spacegroup: the space group number of symbol that will be used to symmetrize the structure.
        current: a Structure to which the new atoms will be added. If None a new structure will be
            generated.
        supergroup_transf: a tuple with the (P, p) transformation from the subgroup to the supergroup
            as defined on the Bilbao website. If used the spacegroup argument should indicate the number
            of the supergroup.
        specie: The specie of the atom that should be added.
        symm_ops: a list of symmetry operations to be applied to the single atoms generated. If None the
            list will be extracted from the template.
        max_dist_eq: if the generate atom is closer to one or more of its symmetrical replicas than
            these atoms will be merged in a single one, resulting in a highly symmetrical site.
        min_dist_current: minimal distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        max_dist_current: maximum distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        min_dist_from_template: minimal distance accepted for the newly generated atom with respect
            all the atoms of the template.
        max_dist_from_template: if not None will also constrain the atoms so that the minimal distance
            should will be within this value.
        max_tests: maximum number of loops and inner loops that will be executed trying to generate
            a new structure that satisfy all the contraints.
        symprec (float): Tolerance for symmetry finding.
        angle_tolerance (float): Angle tolerance for symmetry finding.
        return_single (bool): if True only the structure with the new atom will be returned.

    Returns:
        A structure with the added randomly generated atoms. None if the structure could not
        be generated.
    """

    if not spacegroup and not symm_ops:
        spga = SpacegroupAnalyzer(template, symprec=symprec, angle_tolerance=angle_tolerance)
        spacegroup = spga.get_space_group_number()
        symm_ops = spga.get_symmetry_operations()

    lattice = template.lattice
    if supergroup_transf:
        # the convention of the Bilbao should require a transpose to work with
        # the lattice.matrix in its form.
        Pt = np.transpose(supergroup_transf[0])
        lattice = np.matmul(np.linalg.inv(Pt), lattice.matrix)
    # while dist_zeol < min_dist_from_template or dist_curr < min_dist_c or dist_curr > max_dist_c:
    for i in range(max_tests):
        # logger.debug(f"new symm at: external loop {i}")
        test_coords = postion_generator()
        for j in range(max_tests):
            # logger.debug(f"new symm at: internal loop {j}")
            if symm_ops:
                single = structure_from_symmops(symm_ops, lattice, [specie], [test_coords])
            else:
                single = Structure.from_spacegroup(spacegroup, lattice, [specie], [test_coords])

            # if supergroup:
            #     sg_fcoords = sgase.equivalent_sites(single.frac_coords, symprec=symprec, onduplicates="replace")[0]
            #     single = Structure(lattice=lattice, coords=sg_fcoords, species=[specie] * len(sg_fcoords))
            if supergroup_transf:
                single.make_supercell(Pt)
                single.translate_sites(indices=list(range(len(single))), vector=supergroup_transf[1])

            site_fcoords = np.mod(single.frac_coords, 1)
            eq_fcoords, _, _, _ = single.lattice.get_points_in_sphere(
                    site_fcoords, center=single.cart_coords[0], r=max_dist_eq, zip_results=False)
            neq = len(eq_fcoords)
            if neq == 1:
                break
            test_coords = np.mean(eq_fcoords, axis=0)
        else:
            return None

        # dist_template = get_min_dist(single[0].frac_coords, template)
        dist_template = get_min_dist(single.frac_coords, template)
        logger.debug(f"dist min template {dist_template}")
        max_dist_condition = max_dist_from_template is None or dist_template < max_dist_from_template
        if dist_template > min_dist_from_template and max_dist_condition:
            if current is not None:
                dist_curr = get_min_dist(single[0].frac_coords, current)
                if min_dist_current < dist_curr < max_dist_current:
                    break
            else:
                break
    else:
        return None

    if current is None or return_single:
        return single
    else:
        new_coords = np.concatenate((current.frac_coords, single.frac_coords))
        new_species = np.concatenate((current.species, single.species))
        structure = Structure(lattice=single.lattice, species=new_species,
                              coords=new_coords, coords_are_cartesian=False)
        return structure


def add_new_symmetrized_atom(template: Structure, spacegroup: Union[str, int] = None,
                             current: Structure = None, supergroup_transf: Tuple[List, List] = None,
                             symm_ops: List[SymmOp] = None, specie: str = "C", max_dist_eq: float = 0.1,
                             min_dist_current: float = 1.2, max_dist_current: float = 1.6,
                             min_dist_from_template: float = 3, max_dist_from_template: float = None, max_tests: int = 1000,
                             symprec: float = 0.001, angle_tolerance: float = 5.0, return_single: bool = False) -> Optional[Structure]:
    """
    Add a new atom to the structure along with all its symmetrically equivalent atoms.
    The symmetric atoms can be determined based on the spacegroup or a list of symmetry operations.
    The latter has the precedence and if none is given the list of operations will be extracted from
    the template with spglib.
    The atoms are added in such a way to be away from a determined template and within a range of
    distanced with other existing atoms. The position of the atom is generated randomly and a maximum
    number of tests will be performed to try to generate a new atoms that satisfies all the required
    constraints. If this is not possible returns None. If current is not defined a new structure is generated.

    Args:
        template: a pymatgen Structure defining the template.
        spacegroup: the space group number of symbol that will be used to symmetrize the structure.
        current: a Structure to which the new atoms will be added. If None a new structure will be
            generated.
        supergroup_transf: a tuple with the (P, p) transformation from the subgroup to the supergroup
            as defined on the Bilbao website. If used the spacegroup argument should indicate the number
            of the supergroup.
        specie: The specie of the atom that should be added.
        symm_ops: a list of symmetry operations to be applied to the single atoms generated. If None the
            list will be extracted from the template.
        max_dist_eq: if the generate atom is closer to one or more of its symmetrical replicas than
            these atoms will be merged in a single one, resulting in a highly symmetrical site.
        min_dist_current: minimal distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        max_dist_current: maximum distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        min_dist_from_template: minimal distance accepted for the newly generated atom with respect
            all the atoms of the template.
        max_dist_from_template: if not None will also constrain the atoms so that the minimal distance
            should will be within this value.
        max_tests: maximum number of loops and inner loops that will be executed trying to generate
            a new structure that satisfy all the contraints.
        symprec (float): Tolerance for symmetry finding.
        angle_tolerance (float): Angle tolerance for symmetry finding.
        return_single (bool): if True only the structure with the new atom will be returned.

    Returns:
        A structure with the added randomly generated atoms. None if the structure could not
        be generated.
    """
    generator = lambda: np.random.uniform(0., 1., size=3)
    s = add_new_symmetrized_atom_func(template=template, postion_generator=generator, spacegroup=spacegroup, current=current,
                                      supergroup_transf=supergroup_transf, symm_ops=symm_ops, specie=specie,
                                      max_dist_eq=max_dist_eq, min_dist_current=min_dist_current,
                                      max_dist_current=max_dist_current, min_dist_from_template=min_dist_from_template,
                                      max_dist_from_template=max_dist_from_template, max_tests=max_tests,
                                      symprec=symprec, angle_tolerance=angle_tolerance, return_single=return_single)

    return s


def add_new_symmetrized_atom_undercoord(template: Structure, cutoff: float = 1.8, indices: List[int] = None,
                                        min_neighbors: int = 3, spacegroup: Union[str, int] = None, min_r: float = 1.2,
                                        max_r: float = 1.9, current: Structure = None, supergroup_transf: Tuple[List, List] = None,
                                        symm_ops: List[SymmOp] = None, specie: str = "C", max_dist_eq: float = 0.1,
                                        min_dist_current: float = 1.2, max_dist_current: float = 1.6,
                                        min_dist_from_template: float = 3, max_dist_from_template: float = None,
                                        max_tests: int = 1000, symprec: float = 0.001, angle_tolerance: float = 5.0,
                                        return_single: bool = False) -> Optional[Structure]:
    """
    Add a new atom to the structure along with all its symmetrically equivalent atoms.
    The symmetric atoms can be determined based on the spacegroup or a list of symmetry operations.
    The latter has the precedence and if none is given the list of operations will be extracted from
    the template with spglib.
    The atoms are added in such a way to be away from a determined template and within a range of
    distanced with other existing atoms, with the constraint that they can be bound to atoms that have a
    number of neighbors smaller than a specified value. The number of neighbors is determined by a cutoff distance.
    The position of the atom is generated randomly and a maximum number of tests will be performed to try
    to generate a new atoms that satisfies all the required constraints. If this is not possible returns None.
    If current is not defined a new structure is generated.

    Args:
        template: a pymatgen Structure defining the template.
        cutoff: the cutoff used to determine the bonding between atoms.
        indices: if the cutoff is not given, the atoms to which the atoms can be added can be specified
            explicitly with a list of indices in the current structure.
        min_neighbors: the minimum number of neighbors. Sites with less will be considered
            undercoordinated.
        min_r: minimum distance from the atom to which the new atom will be added.
        max_r: maximum distance from the atom to which the new atom will be added.
        spacegroup: the space group number of symbol that will be used to symmetrize the structure.
        current: a Structure to which the new atoms will be added. If None a new structure will be
            generated.
        supergroup_transf: a tuple with the (P, p) transformation from the subgroup to the supergroup
            as defined on the Bilbao website. If used the spacegroup argument should indicate the number
            of the supergroup.
        specie: The specie of the atom that should be added.
        symm_ops: a list of symmetry operations to be applied to the single atoms generated. If None the
            list will be extracted from the template.
        max_dist_eq: if the generate atom is closer to one or more of its symmetrical replicas than
            these atoms will be merged in a single one, resulting in a highly symmetrical site.
        min_dist_current: minimal distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        max_dist_current: maximum distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        min_dist_from_template: minimal distance accepted for the newly generated atom with respect
            all the atoms of the template.
        max_dist_from_template: if not None will also constrain the atoms so that the minimal distance
            should will be within this value.
        max_tests: maximum number of loops and inner loops that will be executed trying to generate
            a new structure that satisfy all the contraints.
        symprec (float): Tolerance for symmetry finding.
        angle_tolerance (float): Angle tolerance for symmetry finding.
        return_single (bool): if True only the structure with the new atom will be returned.

    Returns:
        A structure with the added randomly generated atoms. None if the structure could not
        be generated.
    """
    # for compatibility with other generator functions accepts current=None
    if current is None:
        return None

    if indices is None:
        if cutoff is None:
            raise ValueError("one among indices and cutoff should be defined.")

        indices = get_undercoordinated(structure=current, cutoff=cutoff, min_neighbors=min_neighbors)

    if len(indices) == 0:
        return None

    def generator():
        ind = random.choice(indices)
        r = random.uniform(min_r, max_r)
        p = random_point_on_a_sphere(r) + current[ind].coords
        return current.lattice.get_fractional_coords(p)

    s = add_new_symmetrized_atom_func(template=template, postion_generator=generator, spacegroup=spacegroup, current=current,
                                      supergroup_transf=supergroup_transf, symm_ops=symm_ops, specie=specie,
                                      max_dist_eq=max_dist_eq, min_dist_current=min_dist_current,
                                      max_dist_current=max_dist_current, min_dist_from_template=min_dist_from_template,
                                      max_dist_from_template=max_dist_from_template, max_tests=max_tests,
                                      symprec=symprec, angle_tolerance=angle_tolerance, return_single=return_single)

    return s


def add_new_symmetrized_atom_bridge(template: Structure, cutoff: float = 1.8, indices: List[int] = None,
                                    min_neighbors: int = 3, max_r: float = 0.6, spacegroup: Union[str, int] = None,
                                    min_dist_pair: float = 3, max_dist_pair: float = 4., current: Structure = None,
                                    supergroup_transf: Tuple[List, List] = None, symm_ops: List[SymmOp] = None,
                                    specie: str = "C", max_dist_eq: float = 0.1, min_dist_current: float = 1.2,
                                    max_dist_current: float = 1.6, min_dist_from_template: float = 3,
                                    max_dist_from_template: float = None, max_tests: int = 1000, symprec: float = 0.001,
                                    angle_tolerance: float = 5.0, return_single: bool = False) -> Optional[Structure]:
    """
    Add a new atom to the structure along with all its symmetrically equivalent atoms.
    The symmetric atoms can be determined based on the spacegroup or a list of symmetry operations.
    The latter has the precedence and if none is given the list of operations will be extracted from
    the template with spglib.
    The atoms are added in such a way to be away from a determined template and within a range of
    distanced with other existing atoms, with the constraint that the new atom should make a bridge between
    two undercoordinated atoms of the current structure. More explicitly the atom will be generated randomly
    in a sphere centered midway between the two atoms to bridge. A maximum number of tests will be performed to try
    to generate a new atoms that satisfies all the required constraints. If this is not possible returns None.
    If current is not defined a new structure is generated.

    Args:
        template: a pymatgen Structure defining the template.
        cutoff: the cutoff used to determine the bonding between atoms.
        indices: if the cutoff is not given, the atoms to which the atoms can be added can be specified
            explicitly with a list of indices in the current structure.
        min_neighbors: the minimum number of neighbors. Sites with less will be considered
            undercoordinated.
        max_r: maximum distance from the atom to which the new atom will be added.
        spacegroup: the space group number of symbol that will be used to symmetrize the structure.
        min_dist_pair: minimum distance for considering two undercoordinated atoms a pair that can be bridged.
        max_dist_pair: maximum distance for considering two undercoordinated atoms a pair that can be bridged.
        current: a Structure to which the new atoms will be added. If None a new structure will be
            generated.
        supergroup_transf: a tuple with the (P, p) transformation from the subgroup to the supergroup
            as defined on the Bilbao website. If used the spacegroup argument should indicate the number
            of the supergroup.
        specie: The specie of the atom that should be added.
        symm_ops: a list of symmetry operations to be applied to the single atoms generated. If None the
            list will be extracted from the template.
        max_dist_eq: if the generate atom is closer to one or more of its symmetrical replicas than
            these atoms will be merged in a single one, resulting in a highly symmetrical site.
        min_dist_current: minimal distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        max_dist_current: maximum distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure. Not applied if current is None.
        min_dist_from_template: minimal distance accepted for the newly generated atom with respect
            all the atoms of the template.
        max_dist_from_template: if not None will also constrain the atoms so that the minimal distance
            should will be within this value.
        max_tests: maximum number of loops and inner loops that will be executed trying to generate
            a new structure that satisfy all the contraints.
        symprec (float): Tolerance for symmetry finding.
        angle_tolerance (float): Angle tolerance for symmetry finding.
        return_single (bool): if True only the structure with the new atom will be returned.

    Returns:
        A structure with the added randomly generated atoms. None if the structure could not
        be generated.
    """
    # for compatibility with other generator functions accepts current=None
    if current is None:
        return None

    if indices is None:
        if cutoff is None:
            raise ValueError("one among indices and cutoff should be defined.")

        indices = get_undercoordinated(structure=current, cutoff=cutoff, min_neighbors=min_neighbors)

    if len(indices) == 0:
        return None

    dm = current.distance_matrix[indices][:, indices]
    pairs = list(zip(*np.where((min_dist_pair < dm) & (dm < max_dist_pair))))
    if len(pairs) == 0:
        return None

    def generator():
        sub_ind = random.choice(pairs)
        p1 = current[indices[sub_ind[0]]]
        p2 = current[indices[sub_ind[1]]]
        # take the closest image
        jimage = current.lattice.get_distance_and_image(p1.frac_coords, p2.frac_coords)[1]
        p2_coords = current.lattice.get_cartesian_coords(jimage + p2.frac_coords)
        r = random.uniform(0, max_r)
        p = random_point_on_a_sphere(r) + (p1.coords + p2_coords) / 2
        return current.lattice.get_fractional_coords(p)

    s = add_new_symmetrized_atom_func(template=template, postion_generator=generator, spacegroup=spacegroup, current=current,
                                      supergroup_transf=supergroup_transf, symm_ops=symm_ops, specie=specie,
                                      max_dist_eq=max_dist_eq, min_dist_current=min_dist_current,
                                      max_dist_current=max_dist_current, min_dist_from_template=min_dist_from_template,
                                      max_dist_from_template=max_dist_from_template, max_tests=max_tests,
                                      symprec=symprec, angle_tolerance=angle_tolerance, return_single=return_single)

    return s


def add_c2_symmetrized(template: Structure, spacegroup: Union[str, int] = None,
                       current: Structure = None, supergroup_transf: Tuple[List, List] = None,
                       symm_ops: List[SymmOp] = None, max_dist_eq: float = 0.1,
                       min_dist_current: Union[List[float], float] = 1.2, max_dist_current: Union[List[float], float] = 1.6,
                       min_dist_from_template: float = 3, max_dist_from_template: float = None,
                       min_dist_cc: float = 1.2, max_dist_cc: float = 1.6, max_tests: int = 1000,
                       symprec: float = 0.001, angle_tolerance: float = 5.0) -> Optional[Structure]:
    """
    Add two carbon atoms to the structure along with all its symmetrically equivalent atoms.
    The symmetric atoms can be determined based on the spacegroup or a list of symmetry operations.
    The latter has the precedence and if none is given the list of operations will be extracted from
    the template with spglib.
    The atoms are added in such a way to be away from a determined template and within a range of
    distanced with other existing atoms. The position of the atom is generated randomly and a maximum
    number of tests will be performed to try to generate a new atoms that satisfies all the required
    constraints. If this is not possible returns None. If current is not defined a new structure is generated.

    Args:
        template: a pymatgen Structure defining the template.
        spacegroup: the space group number of symbol that will be used to symmetrize the structure.
        current: a Structure to which the new atoms will be added. If None a new structure will be
            generated.
        supergroup_transf: a tuple with the (P, p) transformation from the subgroup to the supergroup
            as defined on the Bilbao website. If used the spacegroup argument should indicate the number
            of the supergroup.
        specie: The specie of the atom that should be added.
        symm_ops: a list of symmetry operations to be applied to the single atoms generated. If None the
            list will be extracted from the template.
        max_dist_eq: if the generate atom is closer to one or more of its symmetrical replicas than
            these atoms will be merged in a single one, resulting in a highly symmetrical site.
        min_dist_current: minimal distance accepted for the newly generated atoms with respect to at least
            one of the atoms in the "current" structure and from the two C atoms added. If a list the
            two values will be used for the two C atoms.
        max_dist_current: maximum distance accepted for the newly generated atom with respect to at least
            one of the atoms in the "current" structure and from the two C atoms added. If a list the
            two values will be used for the two C atoms.
        min_dist_from_template: minimal distance accepted for the newly generated atom with respect
            all the atoms of the template.
        max_dist_from_template: if not None will also constrain the atoms so that the minimal distance
            should will be within this value.
        min_dist_cc: minimum distance between the added C.
        max_dist_cc: maximum distance between the added C.
        max_tests: maximum number of loops and inner loops that will be executed trying to generate
            a new structure that satisfy all the contraints.
        symprec (float): Tolerance for symmetry finding.
        angle_tolerance (float): Angle tolerance for symmetry finding.

    Returns:
        A structure with the added randomly generated atoms. None if the structure could not
        be generated.
    """

    # run this here since add_symmetrized_atom could be called more than once and this may
    # save some time
    if not spacegroup and not symm_ops:
        spga = SpacegroupAnalyzer(template, symprec=symprec, angle_tolerance=angle_tolerance)
        spacegroup = spga.get_space_group_number()
        symm_ops = spga.get_symmetry_operations()

    if isinstance(min_dist_current, (list, tuple)):
        mindc1, mindc2 = min_dist_current[:2]
    else:
        mindc1 = mindc2 = min_dist_current

    if isinstance(max_dist_current, (list, tuple)):
        maxdc1, maxdc2 = max_dist_current[:2]
    else:
        maxdc1 = maxdc2 = max_dist_current

    for i in range(max_tests):
        # call a first time to get the first C. The constrain is given by the "current" structure
        c1 = add_new_symmetrized_atom(template=template, spacegroup=spacegroup, current=current, supergroup_transf=supergroup_transf,
                                      symm_ops=symm_ops, specie="C", max_dist_eq=max_dist_eq, min_dist_current=mindc1,
                                      max_dist_current=maxdc1, min_dist_from_template=min_dist_from_template,
                                      max_dist_from_template=max_dist_from_template, max_tests=max_tests,
                                      symprec=symprec, angle_tolerance=angle_tolerance, return_single=True)

        if not c1:
            return None

        # the second call is to set the second atom within a range from the first one, so the "current" is c1
        c2 = add_new_symmetrized_atom(template=template, spacegroup=spacegroup, current=c1,
                                      supergroup_transf=supergroup_transf, symm_ops=symm_ops, specie="C",
                                      max_dist_eq=max_dist_eq, min_dist_current=min_dist_cc, max_dist_current=max_dist_cc,
                                      min_dist_from_template=min_dist_from_template, max_dist_from_template=max_dist_from_template,
                                      max_tests=max_tests, symprec=symprec, angle_tolerance=angle_tolerance, return_single=True)

        # If c1 could be found it is unlikely that will not found c2, but check for safety.
        if not c2:
            continue

        # now check that the full structure still respects all the selected criteria

        full = merge_structures(c1, c2)
        # if current is None all the checks have already been done in the second call to add_new_symmetrized_atom
        if current is None:
            return full
        else:
            dist_curr = get_min_dist(c2.frac_coords, current)
            if mindc2 < dist_curr < maxdc2:
                full = merge_structures(full, current)
                return full

    return None


def get_symmetrized_structure(structure: Structure, spacegroup: Union[str, int],
                              symprec: float = 0.1) -> Structure:
    latt = structure.lattice
    symmetrized = False
    while not symmetrized:
        sym_struct = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetrized_structure()

        new_frac_coords = []
        new_species = []
        symmetrized = True
        for eq_sites in sym_struct.equivalent_sites:
            site = eq_sites[0]
            new_species.append(site.specie)
            if len(eq_sites) == 1:
                new_frac_coords.append(site.frac_coords)
            else:
                symmetrized = False
                site_fcoords = np.mod([s.frac_coords for s in eq_sites[1:]], 1)
                eq_fcoords, _, _, _ = latt.get_points_in_sphere(
                    site_fcoords, pt=site.cart_coords[0], r=symprec, zip_results=False)
                new_frac_coords.append(np.mean(eq_fcoords, axis=0))

        structure = Structure.from_spacegroup(
            spacegroup, latt,
            new_species, new_frac_coords
        )

    return structure


def structure_from_symmops(symm_ops: List[SymmOp], lattice: Union[List, np.ndarray, Lattice],
                           species: Sequence[Union[str, Element, Species, DummySpecies, Composition]],
                           coords: Sequence[Sequence[float]], coords_are_cartesian: bool = False,
                           tol: float = 1e-5):
    if not isinstance(lattice, Lattice):
        lattice = Lattice(lattice)

    if len(species) != len(coords):
        raise ValueError(
            "Supplied species and coords lengths (%d vs %d) are "
            "different!" % (len(species), len(coords))
        )

    frac_coords = np.array(coords, dtype=np.float) if not coords_are_cartesian else \
        lattice.get_fractional_coords(coords)

    all_sp = []  # type: List[Union[str, Element, Species, DummySpecies, Composition]]
    all_coords = []  # type: List[List[float]]
    for i, (sp, c) in enumerate(zip(species, frac_coords)):
        cc = []
        for o in symm_ops:
            pp = o.operate(c)
            pp = np.mod(np.round(pp, decimals=10), 1)
            if not in_array_list(cc, pp, tol=tol):
                cc.append(pp)
        all_sp.extend([sp] * len(cc))
        all_coords.extend(cc)

    return Structure(lattice, all_sp, all_coords)


def merge_structures(*structures: Structure,
                     validate_proximity: bool = False) -> Structure:
    """
    Returns a structure that contains the sites of all the initial structures.
    The structures should have the same lattice.

    Args:
        structures: the structures to be merged.
        validate_proximity: if True will validate the proximity.

    Returns:
        a merged Structure.
    """
    if len(structures) < 2:
        raise ValueError("Should provide at least 2 structures")
    for si in structures[1:]:
        if not np.allclose(structures[0].lattice.matrix, si.lattice.matrix):
            raise ValueError("Cannot join structures with different lattices")

    frac_coords = np.concatenate([s.frac_coords for s in structures])
    species = [sp for s in structures for sp in s.species]
    structure = Structure(lattice=structures[0].lattice, species=species, coords=frac_coords,
                          coords_are_cartesian=False, validate_proximity=validate_proximity)

    return structure


def to_primitive(structure: Structure, spacegroup: int = None, symprec: float = 0.01,
                 preserve_properties: bool = True, primitive_method: str = None) -> Tuple[Structure, bool, Optional[List]]:
    spga = SpacegroupAnalyzer(structure, symprec=symprec)

    if spacegroup and spga.get_space_group_number() != spacegroup:
        return structure, False, None

    if not primitive_method:
        methods = ["spga_find", "spga_standard", "structure"]
    else:
        methods = [primitive_method]

    for m in methods:
        if m == "spga_find":
            primitive = spga.find_primitve()
        elif m == "spga_standard":
            primitive = spga.get_primitive_standard_structure()
        elif m == "structure":
            primitive = structure.get_primitive_structure()
        else:
            raise ValueError(f"unknown method {m}")

        latt = primitive.lattice
        # if the primitive does not reduce the structure none of the methods will. It can return here
        if len(structure) == len(primitive):
            return structure, False, None
        new_sites = []
        for s in structure:
            new_s = PeriodicSite(
                s.specie, s.coords, latt,
                to_unit_cell=True, coords_are_cartesian=True,
                properties=s.properties)
            if not any(map(new_s.is_periodic_image, new_sites)):
                new_sites.append(new_s)
        primitive = Structure.from_sites(new_sites)

        # the matrix should be converted to integers otherwise some problems of rounding
        # may happen in the inverse conversion using make_supercell.
        conv_float = np.matmul(structure.lattice.matrix, np.linalg.inv(latt.matrix))
        conversion_matrix = np.rint(conv_float)
        if not np.allclose(conv_float, conversion_matrix):
            continue

        if preserve_properties:
            set_properties(primitive, get_properties(structure))

        return primitive, True, conversion_matrix

    return structure, False, None


def to_supercell(structure: Structure, conversion_matrix: List, preserve_properties=False):
    converted = structure.copy()
    converted.make_supercell(conversion_matrix)

    if preserve_properties:
        set_properties(converted, get_properties(structure))

    return converted


def check_conversion(template: Structure, n_tests: int = 10, min_dist_from_template: float = 1.,
                     symprec: float = 0.01, primitive_method: str = None, spacegroup: Union[str, int] = None,
                     symm_ops: List[SymmOp] = None) -> Tuple[bool, bool, Optional[Structure]]:
    """
    Helper function to check if the subsequent application of to_primitive+to_supercell brings an atom back
    to its original position.
    Generates a number of structures with add_new_symmetrized_atom and all should satisfy the condition.

    Args:
        template: the Structure used as template for add_new_symmetrized_atom.
        n_tests: the number of tests performed.
        min_dist_from_template: the value passed to add_new_symmetrized_atom.
        symprec: the symprec value passed to the SpacegroupAnalyzer, add_new_symmetrized_atom and to_primitive.
        primitive_method: the primitive method passed to to_primitive.
        spacegroup: value passed to add_new_symmetrized_atom.
        symm_ops: value passed to add_new_symmetrized_atom.

    Returns:
        a tuple with:
            - a bool, False if the test failed because the to_primitive conversion failed.
            - a bool, False if when converting back with to_supercell, the resulting cell is not
              equivalent to the original one.
            - the Structure for which the conversion failed. None if the conversion succeded for all the tests.
    """
    spga = SpacegroupAnalyzer(template, symprec=symprec)
    spgn = spga.get_space_group_number()
    for i in range(n_tests):
        s = add_new_symmetrized_atom(template=template, min_dist_from_template=min_dist_from_template,
                                     symprec=symprec, spacegroup=spacegroup, symm_ops=symm_ops)
        if not s:
            raise RuntimeError("Could not generate a test structure")
        prim, converted, matrix = to_primitive(structure=s, spacegroup=spgn, symprec=symprec,
                                               primitive_method=primitive_method)
        if not converted:
            return False, False, s

        conv = to_supercell(prim, matrix)
        if len(s) != len(conv) or not s.lattice == conv.lattice:
            return True, False, s

        dist = s.lattice.get_all_distances(s.frac_coords, conv.frac_coords)
        row_ind, col_ind = linear_sum_assignment(dist)

        if dist[row_ind, col_ind].sum() / len(s) > 1e-5:
            return True, False, s

    return True, True, None


def set_properties(structure: Structure, properties: dict) -> Structure:
    """
    Helper function to set a "properties" attribute in a Structure.
    Updates the dictionary if one is already present.

    Args:
        structure: a pymatgen Structure.
        properties: a dictionary with the properties to be set.

    Returns:
        The Structure with the properties set.
    """
    if not hasattr(structure, "properties"):
        structure.properties = {}

    structure.properties.update(properties)

    return structure


def get_properties(structure: Structure) -> dict:
    """
    Helper function to get all the dictionary of the properties
    set in a structure. If not present it will set a new one and
    returns it, so that it can be edited.

    Args:
        structure: a pymatgen Structure.

    Returns:
        The dictionary of the properties.
    """
    if not hasattr(structure, "properties"):
        structure.properties = {}

    return structure.properties


def get_property(structure: Structure, name: str) -> Any:
    """
    Helper function to get the value of a single property
    inside a Structure.

    Args:
        structure: a pymatgen Structure with the property that should be
            retrieved.
        name: the name of the property.

    Returns:
        The value of the property, if present. None otherwise.
    """
    return get_properties(structure).get(name, None)


def set_structure_id(structure: Structure) -> Structure:
    """
    Helper function to set a "structure_id" property in the
    structure. The id is a string generated with uuid4.
    Will not be changed if already present.

    Args:
        structure: a pymatgen Structure.

    Returns:
        The structure with the added "structure_id" in the properties.
    """
    if get_property(structure, "structure_id") is None:
        structure_id = str(uuid.uuid4())
        set_properties(structure, {"structure_id": structure_id})
    return structure


def has_low_energy(structure: Structure, energy_threshold: float) -> bool:
    """
    Helper function that determines if, based on the properties,
    has an energy per atom lower than the threshold.
    If the energy is not present it will be considered as a structure
    with energy higher than the threshold.

    Args:
        structure: the structure
        energy_threshold: the threshold for the maximum energy.

    Returns:
        True is the structure has energy lower than the threshold
    """

    original_energy = get_property(structure, "energy_per_atom")
    if not original_energy:
        original_energy_tot = get_property(structure, "energy")
        if not original_energy_tot:
            return False
        original_energy = original_energy_tot / len(structure)
    return original_energy <= energy_threshold
