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


def find_largest_tube_direction(structure: Structure, vector: Union[List, Tuple], grid_density: float = 0.1,diagonal : bool = 0) -> Tuple:
    face_axis = np.argmax(vector)
    
    #building the supercell
    if not diagonal:
        sc_size = [vector[i] for i in range(3)]
        
        for i in range(3):
            if sc_size[i] == 0:
                sc_size[i] = 1
    else:
        sc_size = [4,4,4]
    
    supercell = Structure.from_sites(structure)
    supercell.make_supercell(sc_size)

    # generate a 2d grid of points that will cover one face
    n_grid = [int(np.ceil(structure.lattice.abc[i] / grid_density)) for i in range(3) if i != face_axis]

    coords_list = [np.linspace(0, 1, n, endpoint=False) for n in n_grid]
    grid_frac_coords = np.vstack(np.meshgrid(*coords_list)).reshape(2, -1).T
    
    # extend it with array of 0 in the corresponding direction to get full 3D fraction coordinates, now the grid_frac_coord needs to be divided by sc_size to make the grid only in the original cell
    grid_frac_coords = np.insert(grid_frac_coords, face_axis, np.zeros(grid_frac_coords.shape[0]), axis=1)/sc_size

    vector = np.array(vector)*structure.lattice.abc

    all_dist = dist_line_grid_points(lattice=structure.lattice, vector=vector, grid_points=grid_frac_coords,
                                     target_points=structure.frac_coords)
    min_dist = np.min(all_dist, axis=0)
    i_largest_tube = np.argmax(min_dist)
    center_point = grid_frac_coords[i_largest_tube]
    radius = min_dist[i_largest_tube]

    return center_point, radius


def find_tubes(structure: Structure, grid_density: float = 0.1) -> List:

    results = []
    direction = [[1,0,0],
                 [0,1,0],
                 [0,0,1],
                 [1,1,0],
                 [1,0,1],
                 [0,1,1],
                 [1,1,1],
                 [2,1,0],
                 [2,1,1],
                 [2,0,1],
                 [1,2,0],
                 [1,2,1],
                 [0,2,1],
                 [0,1,2],
                 [1,0,2],
                 [1,1,2]]

    for i in range(len(direction)):
        center_point, radius = find_largest_tube_direction(structure=structure, vector=direction[i], grid_density=grid_density,diagonal=0)
        results.append((direction[i], center_point, radius))

    if np.argmax([results[i][2] for i in range(len(results))]) > 2:
        results = []
        direction[:3] = []
        for i in range(len(direction)):
            center_point, radius = find_largest_tube_direction(structure=structure, vector=direction[i], grid_density=grid_density,diagonal=1)
            results.append((direction[i], center_point, radius))

    return results


def find_largest_tube(structure: Structure, grid_density: float = 0.1) -> Tuple:

    tubes = find_tubes(structure=structure, grid_density=grid_density)

    largest = max(tubes, key=lambda x: x[2])

    return largest
