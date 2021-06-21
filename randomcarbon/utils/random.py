import random
import numpy as np


def random_point_on_a_sphere(r: float = 1.0):
    """
    Generates a random point on a sphere of radius r.

    Args:
        r: the radius of the sphere

    Returns:
        A numpy array with size 3 with the coordinates of the generated point.
    """
    z = random.uniform(-1, 1)
    t = random.uniform(-np.pi, np.pi)
    x = np.sin(t) * np.sqrt(1-z**2)
    y = np.cos(t) * np.sqrt(1-z**2)

    return r * np.array([x, y, z])

