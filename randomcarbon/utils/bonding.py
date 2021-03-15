from collections import defaultdict
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
