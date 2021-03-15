from typing import Union, Any, List
import numpy as np
#import numpy.typing as npt
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.util.coord import pbc_shortest_vectors
from ase.atoms import Atoms


# numpy typing requires python 3.8+ and the latest version of numpy
# for the time being leave Any as hint.
def gaussian(x: Any, sigma: float, height: float) -> Any:
    return height * np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))


class TemplateRepulsiveForce:
    """
    Constraint for ASE optimizer that repels the atoms from a template structure.
    Assumes that the template is large enough and that the repulsive force decays
    fast enough that only the closest replica of the atoms in the template has
    effect on the atoms that are being relaxed. This constraint will not be
    applied correctly if the range of the repulsive force is in the same range
    as the lattice parameters.
    """

    def __init__(self, structure: Union[Structure, Atoms], sigma: float, height: float):
        if isinstance(structure, Atoms):
            self.structure = AseAtomsAdaptor().get_structure(structure)
            self.atoms = structure
        else:
            self.structure = structure
            self.atoms = AseAtomsAdaptor().get_atoms(self.structure)
        self.sigma = sigma
        self.height = height

    def adjust_positions(self, atoms: Atoms, new: Atoms):
        pass

    def adjust_forces(self, atoms: Atoms, forces: List):
        at_frac_coords = atoms.get_scaled_positions(wrap=False)
        v, dist2 = pbc_shortest_vectors(self.structure.lattice, self.structure.frac_coords,
                                        at_frac_coords, return_d2=True)

        dist = np.sqrt(dist2)[:, :, None]
        v = np.divide(v, dist, out=np.zeros_like(v), where=dist != 0)

        forces_module = gaussian(dist, self.sigma, self.height)
        repulsive_forces = np.sum(v * forces_module, axis=0)

        forces += repulsive_forces


class TemplateRangeForce:
    """
    Constraint for ASE optimizer that tries to keep the atoms within a certain range
    of distances from a template structure.
    The repulsive is a gaussian from all the atoms. The attractive force is given
    by a constant attractive force when distances are larger than the specified
    "distance" and by a gaussian centered a "distance" when below this threshold.
    For the attractive forces only the closest atom(s) will be considered.
    Assumes that the template is large enough and that the repulsive force decays
    fast enough that only the closest replica of the atoms in the template has
    effect on the atoms that are being relaxed. This constraint will not be
    applied correctly if the range of the repulsive force is in the same range
    as the lattice parameters.
    """

    def __init__(self, structure: Union[Structure, Atoms], sigma: float, height: float,
                 distance: float, min_dist_range: float = None, normalize_attractive_f: bool = True,
                 sigma2: float = None, height2: float = None):
        if isinstance(structure, Atoms):
            self.structure = AseAtomsAdaptor().get_structure(structure)
            self.atoms = structure
        else:
            self.structure = structure
            self.atoms = AseAtomsAdaptor().get_atoms(self.structure)
        self.sigma = sigma
        self.height = height
        self.distance = distance
        self.min_dist_range = min_dist_range
        self.normalize_attractive_f = normalize_attractive_f

        if sigma2 is None:
            sigma2 = sigma
        self.sigma2 = sigma2
        if height2 is None:
            height2 = height
        self.height2 = height2

    def adjust_positions(self, atoms: Atoms, new: Atoms):
        pass

    def adjust_forces(self, atoms: Atoms, forces: List):
        at_frac_coords = atoms.get_scaled_positions(wrap=False)
        v, dist2 = pbc_shortest_vectors(self.structure.lattice, self.structure.frac_coords,
                                        at_frac_coords, return_d2=True)

        dist = np.sqrt(dist2)
        dist_v = dist[:, :, None]
        v = np.divide(v, dist_v, out=np.zeros_like(v), where=dist_v != 0)

        # repulsive part of the force. all the atoms repel
        repulsive_module = gaussian(dist_v, self.sigma, self.height)
        repulsive_forces = np.sum(v * repulsive_module, axis=0)
        forces += repulsive_forces

        # attractive part. Only the closest, or a subset of the points of the template attract.

        # TODO older version: to remove if not useful
        # # if self.template_neighbours_range is not None:
        # # shape: n_at
        # min_dist_template_ind = np.argmin(dist, axis=0)
        # # shape: n_at, n_temp
        # selected_neighbours = self._distance_matrix[min_dist_template_ind] < self.template_neighbours_range
        # # shape: n_temp, n_at
        # selected_neighbours = selected_neighbours.T[:, :, None]

        # if None use only the closest one
        if self.min_dist_range is None:
            min_dist_template_ind = np.argmin(dist, axis=0)
            selected_neighbours = np.zeros_like(dist, dtype=np.bool)
            selected_neighbours[min_dist_template_ind, np.arange(np.shape(dist)[1])] = True
        else:
            # shape: n_at
            min_dist_template = np.min(dist, axis=0)
            # shape: n_temp, n_at
            selected_neighbours = dist < min_dist_template + self.min_dist_range
        selected_neighbours = selected_neighbours[:, :, None]
        print(np.count_nonzero(selected_neighbours))

        forces_module = np.zeros_like(dist_v)
        cond = dist_v < self.distance
        forces_module[cond & selected_neighbours] = - gaussian(self.distance - dist_v[cond & selected_neighbours], self.sigma2, self.height2)
        forces_module[~cond & selected_neighbours] = -self.height2
        if self.min_dist_range is not None and self.normalize_attractive_f:
            # normalize by the number of neighbors so that the maximum attractive force is always self.height2 and does not cumulate
            forces_module /= np.count_nonzero(selected_neighbours, axis=0)
        attractive_forces = np.sum(v * forces_module, axis=0)
        forces += attractive_forces
