from typing import List
from collections import defaultdict
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import NearNeighbors


def get_pairs_max_dist(structure: Structure, near_neighbors: NearNeighbors, pad: float = 0.01) -> dict:
    """
    Given a structure and a NearNeighbors to define the bonded atoms, creates a dictionary that
    will contain a tuple with the pair of bonded species as key (sorted alphabetically) and the maximum
    distance between the bonded species in the structure. Useful for generating the input for the
    rings code.

    Args:
        structure: the structure from which to extract the values
        near_neighbors: the strategy used to determine if atoms are connected or not.
        pad: an additional value added to each extracted distance.

    Returns:
        a dictionary with tuples of pairs of elemets as keys and distances as values.
    """
    nn_info = near_neighbors.get_all_nn_info(structure)

    max_dist = defaultdict(int)

    for nn_data, site in zip(nn_info, structure):
        for nn in nn_data:
            site_nn = nn["site"]
            k = tuple(sorted((site_nn.specie.name, site.specie.name)))
            d = site_nn.distance(site)
            max_dist[k] = max(d, max_dist[k])

    if pad:
        for k in max_dist.keys():
            max_dist[k] += pad

    return max_dist


def get_undercoordinated(structure: Structure, cutoff: float = 1.8, min_neighbors: int = 3) -> List[int]:
    """
    Determines which atoms are undercoordinated.

    Args:
        structure: the structure to analyze.
        cutoff: the cutoff used to determine the bonding between atoms.
        min_neighbors: the minimum number of neighbors. Sites with less will be considered
            undercoordinated.

    Returns:
        A list of indices of the undercoordinated sites.
    """
    dm = structure.distance_matrix
    bonded = (dm > 0) & (dm < cutoff)
    return np.where(np.count_nonzero(bonded, axis=0) < min_neighbors)[0]


def get_undercoordinated_nn(structure: Structure, near_neighbors: NearNeighbors, min_neighbors: int = 3) -> List[int]:
    """
    Determines which atoms are undercoordinated.

    Args:
        structure: the structure to analyze.
        near_neighbors: the NearNeighbors used to determined the bonding between atoms.
        min_neighbors: the minimum number of neighbors. Sites with less will be considered
            undercoordinated.

    Returns:
        A list of indices of the undercoordinated sites.
    """

    indices = []
    for i in range(len(structure)):
        if near_neighbors.get_cn(structure, i) < min_neighbors:
            indices.append(i)

    return indices
