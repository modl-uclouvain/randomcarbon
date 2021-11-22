from typing import Union, List, Tuple
import numpy as np
from monty.dev import requires
from pymatgen.core import Structure
try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def cylinder(vector: Union[List, Tuple], center: List[float], radius: float):
    """
    parametrize the cylinder of radius r, height h, base point a
    """

    mag = np.linalg.norm(vector)
    # unit vector in direction of axis
    vector = vector / mag
    not_v = np.array([1, 0, 0])
    if (vector == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(vector, not_v)
    # normalize n1
    n1 /= np.linalg.norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(vector, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(-mag, mag, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    # generate coordinates for surface
    x, y, z = [center[i] + vector[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    return x, y, z


@requires(go, "plotly should be installed")
def plot_tube_plotly(structure: Structure, vector: Union[List, Tuple], center: List[float], radius: float):

    cart_vector = np.matmul(vector, structure.lattice.matrix)
    cart_center = structure.lattice.get_cartesian_coords(center)
    cyl = cylinder(cart_vector, cart_center, radius=radius)
    cyl_plt = go.Surface(x=cyl[0], y=cyl[1], z=cyl[2], colorbar=None, opacity=1, text="Distance", showscale=False)

    supercell = structure.copy()
    sm = [3, 3, 3]
    sm[np.argmax(vector)] = 1
    supercell.make_supercell(sm)
    trasl_vect = [-1/3, -1/3, -1/3]
    trasl_vect[np.argmax(vector)] = 0
    supercell.translate_sites(list(range(len(supercell))), vector=trasl_vect, frac_coords=True, to_unit_cell=False)
    at_coords = supercell.cart_coords
    center_plt = go.Scatter3d(x=[cart_center[0]], y=[cart_center[1]], z=[cart_center[2]], name="Chosen point")
    atoms_plt = go.Scatter3d(x=at_coords[:, 0], y=at_coords[:, 1], z=at_coords[:,2], mode="markers", name="Atoms")
    fig = go.Figure(data=[cyl_plt, center_plt, atoms_plt])
    fig.show()

