from typing import List
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from randomcarbon.utils.symmetry import unequivalent_ind


def discretize_structure(structure: Structure, grid_density: float = 0.1, symprec: float = 0.01,
                         angle_tolerance: float = 5, to_unit_cell: bool = False) -> List:
    if to_unit_cell:
        structure = Structure.from_sites(structure.sites, to_unit_cell=True)
    spga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    # don't use the symmetrizedStructure as it calculates needless quantities
    # (e.g. the symmetry operations) and slows down the procedure
    # ssym = spga.get_symmetrized_structure()
    grid_coords = np.floor_divide(structure.cart_coords, grid_density).astype(np.int32)
    unique_grid_coords = []
    equivalent_indices = unequivalent_ind(spga.get_symmetry_dataset()["equivalent_atoms"])
    for eq_ind in equivalent_indices:
        inequivalent_representative = min(grid_coords[eq_ind].tolist())
        unique_grid_coords.append(inequivalent_representative)

    unique_grid_coords = sorted(unique_grid_coords)

    return unique_grid_coords


def hash_grid(structure: Structure, grid_density: float = 0.1, symprec: float = 0.01,
              angle_tolerance: float = 5) -> int:
    grid = discretize_structure(structure, grid_density=grid_density, symprec=symprec, angle_tolerance=angle_tolerance,
                                to_unit_cell=True)
    hash_grid = hash(tuple(tuple(g) for g in grid))
    return hash_grid
