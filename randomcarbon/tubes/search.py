from typing import Union, List, Tuple
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.util.coord import pbc_shortest_vectors


def dist_line_grid_points(lattice: Lattice, vector: Union[List, np.ndarray], grid_points: Union[List, np.ndarray],
                          target_points: Union[List, np.ndarray]) -> np.ndarray:

    # Generate the pairwise distance between the points of the grid and the target points.
    # The distance is with the closest replica of the point
    distance_vect = pbc_shortest_vectors(lattice, target_points, grid_points)
    distances = np.linalg.norm(np.cross(distance_vect, vector), axis=-1) / np.linalg.norm(vector)

    return distances


def find_largest_tube_direction(structure: Structure, vector: Union[List, Tuple], grid_density: float = 0.1) -> Tuple:
    face_axis = np.argmax(vector)

    # generate a 2d grid of points that will cover one face
    n_grid = [int(np.ceil(structure.lattice.abc[i] / grid_density)) for i in range(3) if i != face_axis]

    coords_list = [np.linspace(0, 1, n, endpoint=False) for n in n_grid]
    grid_frac_coords = np.vstack(np.meshgrid(*coords_list)).reshape(2, -1).T
    # extend it with array of 0 in the corresponding direction to get full 3D fraction coordinates
    grid_frac_coords = np.insert(grid_frac_coords, face_axis, np.zeros(grid_frac_coords.shape[0]), axis=1)

    vector = structure.lattice.matrix[face_axis]

    all_dist = dist_line_grid_points(lattice=structure.lattice, vector=vector, grid_points=grid_frac_coords,
                                     target_points=structure.frac_coords)
    min_dist = np.min(all_dist, axis=0)
    i_largest_tube = np.argmax(min_dist)
    center_point = grid_frac_coords[i_largest_tube]
    radius = min_dist[i_largest_tube]

    return center_point, radius


def find_tubes(structure: Structure, grid_density: float = 0.1) -> List:

    results = []
    for i in range(3):
        direction = [0, 0, 0]
        direction[i] = 1
        center_point, radius = find_largest_tube_direction(structure=structure, vector=direction, grid_density=grid_density)
        results.append((direction, center_point, radius))

    return results


def find_largest_tube(structure: Structure, grid_density: float = 0.1) -> Tuple:

    tubes = find_tubes(structure=structure, grid_density=grid_density)

    largest = max(tubes, key=lambda x: x[2])

    return largest
