from typing import Union, List, Tuple
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.util.coord import pbc_shortest_vectors
from randomcarbon.data import get_template
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import itertools
import warnings


def check_vector_list(vector: Union[List, np.ndarray], vector_list: Union[List, np.ndarray]) -> bool:
    """
    Given a vector and a list of vectors, it checks whether the vector is not contained in the list
    
    Args:
        vector: the vector it is needed to be checked
        vector_list: the vectors list to check
    
    Returns:
        a bool : True if vector is not contained in vector_list, False if it is
    
    """ 
    for v in vector_list:
            if np.allclose(v,vector,1e-5):
                return False
    return True



def symmetrically_equivalent_directions(T: Union[List, np.ndarray], vector: Union[List, np.ndarray]) -> List:
    """
    Given a set of symmetry operations (only rotations are considered here) and a vector, it returns a list of symmetrically equivalent vectors
     
    Args:
        T: list of rotation symmetry operations matrices
        vector: vector whose equivalent directions are to be found
    Return:
        a List of symmetrically equivalent directions
    """   
    possible_dir = np.dot(T,vector)
    eq_dir = []
    for x in possible_dir:
        if check_vector_list(x,eq_dir):
            eq_dir.append(x)
    return eq_dir


def dist_line_grid_points(lattice: Lattice, vector: Union[List, np.ndarray], grid_points: Union[List, np.ndarray],
                          target_points: Union[List, np.ndarray]) -> np.ndarray:
    """
    Given a lattice, a direction, a grid and some target points, it calculates the distances between the target points and the 
    vector for each point of the grid. It takes into account the periodic replica of the target points and it returns the
    distance with the closest replica of the target point with respect to the grid point

    Args:
        lattice: the lattice of the structure
        vector: the vector from which to calculate the distance
        grid_points: the points of the grid on a face of the unit cell
        target_points: the points from which to calculate the distance

    Returns:
        an array of the distances between the target points and the vector for each grid point

    """
    distance_vect = pbc_shortest_vectors(lattice, target_points, grid_points)
    distances = np.linalg.norm(np.cross(distance_vect, vector), axis=-1) / np.linalg.norm(vector)

    return distances


def find_largest_tube_direction(structure: Structure, vector: Union[List, Tuple], grid_density: float = 0.1) -> Tuple:
    """
    Given a structure, a directions and a grid density, it finds the largest tube, as in the center point and the radius, for that
    structure in the specified direction. The grid density specifies how many points have to be explored as potential center points
    of the tubes

    Args:
        structure: the structure where the tube is to be found
        vector: the direction in which the tube is to be found
        grid_density: the density of the grid on a face in 1D

    Returns:
        the center point and the radius of the largest tube in the wanted direction

    """
    face_axis = np.argmax(np.abs(vector))
    supercell = Structure.from_sites(structure)
    
    if vector not in [[1,0,0],[0,1,0],[0,0,1]]:
        nn = np.max(np.abs(vector))
        supercell_l = [(nn+1) for i in range(3)]
        supercell.make_supercell(supercell_l)
    else: 
        supercell_l = [1,1,1]
    # generate a 2d grid of points that will cover one face
    n_grid = [int(np.ceil(structure.lattice.abc[i] / grid_density)) for i in range(3) if i != face_axis]
    coords_list = [np.linspace(0, 1, n, endpoint=False) for n in n_grid]
    grid_frac_coords = np.vstack(np.meshgrid(*coords_list)).reshape(2, -1).T
    # extend it with array of 0 in the corresponding direction to get full 3D fraction coordinates
    grid_frac_coords = np.insert(grid_frac_coords, face_axis, np.zeros(grid_frac_coords.shape[0]), axis=1) / supercell_l
    vector_cart = np.dot(vector, structure.lattice.matrix)
    all_dist = dist_line_grid_points(lattice=supercell.lattice, vector=vector_cart, grid_points=grid_frac_coords,
                                     target_points=supercell.frac_coords)
    min_dist = np.min(all_dist, axis=0)
    i_largest_tube = np.argmax(min_dist)
    center_point = grid_frac_coords[i_largest_tube] * supercell_l
    radius = min_dist[i_largest_tube]
   
    
    return center_point,radius


def find_tubes(structure: Structure, grid_density: float = 0.1,n: int = 3, symprec: float = 0.01, angle_tolerance: float = 5.0) -> List:
    """
    Given a structure, a grid density and an integer n, it generates all the directions between (-n,-n,-n) and (n,n,n),
    it removes the equivalent directions taking into account the symmetry of the structure and it returns the largest tube
    in each inequivalent direction. The grid density specifies how many points have to be explored as potential center points
    of the tubes

    Args:
       structure: the structure where the tubes have to be found
       grid_density: the density of the grid on a face in 1D
       n: the maximum dimension the directions' indices
       symprec: tolerance for symmetry finding
       angle_tolerance: angle tolerance for symmetry finding

    Returns:
        a list of tuples of direction, center point, radius for the largest tube in each inequivalent directions


    """

    results = []
    directions = list(itertools.product(range(-n,n+1), repeat=3))[::-1]
    directions.remove((0,0,0))
    dir_list = list(directions)
    norms = np.linalg.norm(directions, axis=1)
    for i, d1 in enumerate(dir_list[:-1]):
        for j, d2 in enumerate(dir_list[i+1:], i+1):
            norm_d1 = norms[i]
            norm_d2 = norms[j]
            # check if two directions are the same
            if abs(abs(np.dot(d1, d2)/(norm_d1 * norm_d2)) - 1) < 1e-10:
                to_remove = d1 if norm_d1 > norm_d2 else d2
                try:
                    directions.remove(to_remove)
                except ValueError:
                    pass
    sym_ds = SpacegroupAnalyzer(structure=structure,symprec=symprec,angle_tolerance=angle_tolerance).get_symmetry_dataset()
    if sym_ds:
        T = sym_ds["rotations"]
        to_compute = []
        computed = []
        for direction in directions:
            equiv = symmetrically_equivalent_directions(T,direction)
            if check_vector_list(direction,computed):
                to_compute.append(direction)
                for x in equiv:
                    if check_vector_list(x,computed):
                        computed.append(x)
    else:
        warnings.warn("No symmetry was found for the given structure",UserWarning)
        to_compute = directions

    for vector in to_compute:
        center_point, radius = find_largest_tube_direction(structure=structure, vector=vector, grid_density=grid_density)
        results.append((vector, center_point, radius))
    return results


def find_largest_tube(structure: Structure, grid_density: float = 0.1,n: int = 3, symprec: float = 0.01, angle_tolerance: float = 5.0) -> Tuple:
    """
    Given a structure, a grid density and an integer n, it returns the direction, the radius and the center point of the 
    largest tube. The grid density specifies how many points have to be explored as potential center points
    of the tubes

    Args:
        structure: the structure where the tube has to be found
        grid_density: the density of the grid on a face in 1D
        n: the maximum dimension the directions' indices
        symprec: tolerance for symmetry finding
        angle_tolerance: angle tolerance for symmetry finding


    Returns:
        the direction, the center point and the radius of the largest tube in the structure

    """
    tubes = find_tubes(structure=structure, grid_density=grid_density, n=n,symprec=symprec,angle_tolerance=angle_tolerance)

    largest = max(tubes, key=lambda x: x[2])

    return largest


