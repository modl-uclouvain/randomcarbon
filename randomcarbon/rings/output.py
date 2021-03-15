import os
import glob
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from collections.abc import Sequence
from typing import List, Union, Dict, Generator
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element, Species
from monty.json import MSONable, MontyDecoder
from monty.os import cd
from monty.dev import requires
from randomcarbon.rings.input import RingMethod
try:
    import jupyter_jsmol
    from jupyter_jsmol.viewer import JsmolView
except ImportError:
    jupyter_jsmol = None
    JsmolView = None


rings_colors = defaultdict(lambda: "aqua",
                           {3: "blue", 4: "green", 5: "violet", 6: "red", 7: "yellow",
                            8: "orange", 9: "pink", 10: "lightgrey"})


class Ring(MSONable, Sequence):
    """
    An object describing a ring. Has the list of periodic sites and their indices inside
    the structure to speed up the lookup of a specific site. Does not contain
    the structure to avoid multiple serialization.
    """

    def __init__(self, sites: List[PeriodicSite], indices: List):
        """
        Args:
            sites: a list of sites forming a ring.
            indices: the indices in the structure to which the ring belongs.
        """
        self.sites = sites
        self.indices = np.array(indices)

    def __contains__(self, site: Union[int, PeriodicSite]):
        if isinstance(site, int):
            return site in self.indices
        else:
            return site in self.sites

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, item):
        return self.sites[item]

    @property
    def lattice(self):
        return self.sites[0].lattice

    def iter_from_index(self, start_index: int = 0) -> Generator[PeriodicSite, None, None]:
        """
        Iterates over the sites in the ring starting from a specific site

        Args:
            start_index: the index of the starting atom.
        """
        for idx in range(len(self)):
            yield self.sites[(idx + start_index) % len(self)]

    def iter_from_index_image(self, start_index: int = 0) -> Generator[PeriodicSite, None, None]:
        """
        Iterates over the sites in the ring starting from a specific site. The generated sites
        will be such that they can form the ring locally. In particular, if necessary a
        periodic image of the site will be returned.

        Args:
            start_index: the index of the starting atom.
        """

        lattice = self.lattice
        prev_frac_coords = None
        for idx in range(len(self)):
            site = self.sites[(idx + start_index) % len(self)]
            site_fcoords = site.frac_coords
            if prev_frac_coords is not None:
                _, img = lattice.get_distance_and_image(prev_frac_coords, site_fcoords)
                site_fcoords = site_fcoords + img
                if np.array_equal(img, [0, 0, 0]) and site in self.sites:
                    yield site
                else:
                    img_site = PeriodicSite(species=site.species,
                                            coords=site_fcoords,
                                            lattice=lattice,
                                            to_unit_cell=False,
                                            coords_are_cartesian=False,
                                            properties=site.properties)
                    yield img_site
            else:
                yield site
            prev_frac_coords = site_fcoords

    @property
    def size(self) -> int:
        """
        The number of elements in the ring
        """
        return len(self)

    def get_center(self, ref_index: int = 0, cart_coords: bool = True) -> np.ndarray:
        """
        The coordinates of the center of a ring. Since the ring can be formed by
        atoms that cross the boundary of the cell, the center will be determined
        based on a reference atom. The other atoms forming the ring will be
        determined based on the periodic boundary conditions.

        Args:
            ref_index: the index to which the center will be referred. The index is referred
                to the list of indices in the Ring object (should be in [0, len(self)) ).
            cart_coords: if True the center is given in cartesian coordinates,
                otherwise in fractional coordinates.

        Returns:
            the coordinates of the center.
        """
        coords = []
        lattice = self.lattice
        for site in self.iter_from_index_image(ref_index):
            coords.append(site.coords)

        center = np.mean(coords, axis=0)
        if not cart_coords:
            center = lattice.get_fractional_coords(center)

        return center

    def replace_species(self, structure: Structure, specie: Union[str, Element, Species]) -> Structure:
        """
        Replaces the specie of the sites in the structure for the atoms belonging to the ring.
        It is based on the indices so the structure should match exactly the one used to generate the ring.

        Args:
            structure: the structure to be modified.
            specie: the specie used to replace the sites.

        Returns:
            a copy of the structure with modified sites.
        """
        s = structure.copy()
        for i in self.indices:
            s[i].species = specie

        return s

    def jmol_script(self, structure: Structure, ref_index: int = 0,
                    color: str = None, filepath: str = None) -> str:
        """
        Creates a script for jmol that will draw the structure and color the inside of the ring
        with a selected color. Since rings can cross the cell a reference atom should be specified
        to identify where the ring should be drawn.

        Args:
            structure: the structure containing the ring.
            ref_index: the index of the site to identify which repetition of the ring should be drawn.
                The index is referred to the list of indices in the Ring object (should be in [0, len(self)) ).
            color: a string with the color of the ring.
            filepath: if not None a file will be written with the script.

        Returns:
            the string with the jmol script.
        """

        if color is None:
            color = rings_colors.get(len(self))

        s = structure.copy()

        # use fractional coordinates for the center, since the structure is generated as a cif
        # and uses abc+alpha,beta,gamma to define the lattice.
        c = self.get_center(ref_index=ref_index, cart_coords=False)
        lines = ['set frank off',  # remove jsmol logo
                 'set zoomlarge false',  # use the smaller of height/width when setting zoom level
                 'set waitformoveto off']
        indices = []
        for i, site in enumerate(self.iter_from_index_image(start_index=ref_index)):
            if site not in s:
                s.append(species=site.species,coords=site.frac_coords,
                         validate_proximity=False, coords_are_cartesian=False)
                indices.append(len(s)-1)
            else:
                indices.append(self.indices[i])

        for i in range(len(indices)):
            lines.append(f"draw p{i} POLYGON 3 {{{c[0]}/1 {c[1]}/1 {c[2]}/1}} @{indices[i]+1} @{indices[i-1]+1} translucent {color}")

        lines = ['load inline "' + s.to("cif").replace('"', "'") + '"'] + lines

        script = ";\n".join(lines)

        if filepath:
            with open(filepath, "wt") as f:
                f.write(script)

        return script

    @requires(jupyter_jsmol, "jupyter_jsmol should be install to show the ring")
    def show_jsmol(self, structure: Structure, ref_index: int = 0, color: str = None) -> JsmolView:
        """
        Creates a view with jsmol that will contain the structure and color the inside of the ring
        with a selected color. Since rings can cross the cell a reference atom should be specified
        to identify where the ring should be drawn. Requires jupyter_jsmol.

        Args:
            structure: the structure containing the ring.
            ref_index: the index of the site to identify which repetition of the ring should be drawn.
                The index is referred to the list of indices in the Ring object (should be in [0, len(self)) ).
            color: a string with the color of the ring.
        Returns:
            a view of the structure from jupyter_jsmol.
        """
        script = self.jmol_script(structure=structure, ref_index=ref_index, color=color)
        view = JsmolView(script=script)
        return view


class RingsList(MSONable, Sequence):
    """
    A list of Ring objects for a specific structure as extracted from the rings
    output for a specific method.
    """

    def __init__(self, rings: List[Ring], method: RingMethod, structure: Structure,
                 stats: Union[List, np.ndarray], irreducible: bool):
        """
        Args:
            rings: a list of Ring objects.
            method: the method used to calculate the rings.
            structure: the structure to which the rings belong.
            stats: an array with the statistics of the rings.
            irreducible: if the rings were extracted from the irreducible list.
        """
        self.rings = rings
        self.method = method
        self.structure = structure
        self.stats = np.array(stats)
        self.irreducible = irreducible

    def __len__(self) -> int:
        return len(self.rings)

    def __getitem__(self, item):
        return self.rings[item]

    def get_rings_per_site(self, site: Union[int, PeriodicSite]) -> List[Ring]:
        """
        Get all the rings passing for a site.

        Args:
            site: the selected site or the index of the site in the structure.

        Returns:
            a list of rings to which the site belong.
        """
        return [r for r in self.rings if site in r]

    def get_rings_per_size(self) -> Dict[int, List[Ring]]:
        """
        Gives a dictionary with the rings divided by their size.

        Returns:
            a dictionary with integers representing the size of the rings as keys and
            the list of the rings with that size as values.
        """
        d = defaultdict(list)
        for r in self.rings:
            d[len(r)].append(r)
        return dict(d)

    def get_num_rings_dict(self) -> dict:
        """
        A dictionary with the amount of rings with a specific size.

        Returns:
            a dictionary with integers representing the size of the rings as keys and
            the number of rings with that size as values.
        """
        d = defaultdict(int)
        for r in self.rings:
            d[len(r)] += 1
        return dict(d)

    @classmethod
    def from_dir(cls, path: str, method: Union[RingMethod, int], structure: Structure,
                 irreducible: bool = True):
        """
        Generates the object from the output of the rings calculation. The path should
        be to the "rstat" folder of the output.

        Args:
            path: path to the folder containing the rings output.
            method: the method used to calculate the rings.
            structure: the structure used to calculate the rings.
            irreducible: whether the reducible or irreducible list of rings should be parsed
                (does not apply to the RingMethod.PRIMITIVE and STRONG).

        Returns:
            an instance of RingsList.
        """

        method = RingMethod(method)

        with cd(path):
            if not os.path.isdir(f"liste-{method}") or not os.path.isfile(f"RINGS-res-{method}.dat"):
                raise RuntimeError(f"No outputs for method {method} in {path}")
            # prevent loggings of empty inputs from numpy
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, append=1, message=".*Empty input")
                stats = np.loadtxt(f"RINGS-res-{method}.dat")

            # No rings found
            #TODO should return None instead?
            if len(stats) == 0:
                return cls(rings=[], method=method, structure=structure, irreducible=irreducible, stats=np.array([[]]))

            if len(stats.shape) == 1:
                stats = stats[None, :]

            if irreducible and method < 5:
                filename_pattern = "ri[0-9]*.dat"
            else:
                filename_pattern = "r[0-9]*.dat"
            rings = []
            for p in glob.glob(os.path.join(f"liste-{method.value}", filename_pattern)):
                # switch to 0-based indices
                ring_data = np.loadtxt(p, dtype=np.int) - 1
                for r in ring_data:
                    rings.append(Ring([structure[i] for i in r], r))

        return cls(rings=rings, method=method, structure=structure, irreducible=irreducible, stats=stats)

    @property
    def sizes(self) -> np.ndarray:
        if len(self.stats[0]) == 0:
            return np.array([], dtype=np.int)
        return np.array(self.stats[:, 0], dtype=np.int)

    @property
    def Rc(self) -> np.ndarray:
        if len(self.stats[0]) == 0:
            return np.array([])
        return self.stats[:, 1]

    @property
    def PN(self) -> np.ndarray:
        if len(self.stats[0]) == 0:
            return np.array([])
        return self.stats[:, 2]

    @property
    def P_max(self) -> np.ndarray:
        if len(self.stats[0]) == 0:
            return np.array([])
        return self.stats[:, 3]

    @property
    def P_min(self) -> np.ndarray:
        if len(self.stats[0]) == 0:
            return np.array([])
        return self.stats[:, 4]

    def as_dict(self) -> dict:
        """
        Overridden version of as_dict to only store the indices of the rings and save space when
        the list is serialized.

        Returns:
            dict with the serialized version of the object.
        """

        d = dict(
            structure=self.structure.as_dict(),
            method=self.method.as_dict(),
            stats=self.stats,
            irreducible=self.irreducible,
            rings_indices=[r.indices.tolist() for r in self.rings]
        )

        return d

    @classmethod
    def from_dict(cls, d: dict):
        md = MontyDecoder()
        structure = Structure.from_dict(d["structure"])
        rings = [Ring([structure[i] for i in r], r) for r in d["rings_indices"]]
        stats = md.process_decoded(d["stats"])
        return cls(rings=rings, method=RingMethod.from_dict(d["method"]), structure=structure,
                   irreducible=d["irreducible"], stats=stats)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Generates a pandas DataFrame with the statistics of the rings.
        See http://rings-code.sourceforge.net/index.php?option=com_content&view=article&id=45:ring-statistics&catid=36:phy-chi&Itemid=53
        """
        d = self.get_stats_dict()
        df = pd.DataFrame(d)

        return df

    def get_stats_dict(self) -> dict:
        """
        Generates a dictionary with the rings statistics.
        """
        size = self.sizes
        num_rings = self.get_num_rings_dict()
        d = dict(
            size=size,
            num_rings=[num_rings[i] for i in size],
            Rc=self.Rc,
            PN=self.PN,
            P_max=self.P_max,
            P_min=self.P_min
        )
        return d

    def structure_with_ring_specie(self, ring_ind: int, specie: Union[str, Element, Species]) -> Structure:
        """
        Creates a structure with the atoms belonging to the selected ring replaced by the selected specie.
        Useful for identifying the ring in generic visualization tools.
        Args:
            ring_ind: the index of the ring to be highlighted
            specie: the specie used to replace the original one.

        Returns:
            a new Structure with modified species.
        """
        return self.rings[ring_ind].replace_species(self.structure, specie=specie)

    def jmol_script(self, ring_indices: Union[int, List[int]] = None, rings_sizes: Union[int, List[int]] = None,
                    sites: Union[PeriodicSite, List[PeriodicSite]] = None, colors: dict = None,
                    filepath: str = None) -> str:
        """
        Generates the string for a jmol script that will color the selected rings.
        If more than one option is given only the rings satisfying all of them will be selected.

        Args:
            ring_indices: filtering option based on the indices of the rings in the list.
            rings_sizes: filtering option based on the size of the rings.
            sites: filtering option based on the sites contained in the rings.
            colors: a dictionary with the size of the rings as keys and a string with the color
                as values.
            filepath: if present the script will be written to the specified file.

        Returns:
            a jmol script with the structure and the highlighted rings.
        """
        if ring_indices is not None and not isinstance(ring_indices, (list, tuple, np.ndarray)):
            ring_indices = [ring_indices]

        if rings_sizes is not None and not isinstance(rings_sizes, (list, tuple, np.ndarray)):
            rings_sizes = [rings_sizes]

        if sites is not None and not isinstance(sites, (list, tuple, np.ndarray)):
            sites = [sites]

        if not colors:
            colors = rings_colors

        s = self.structure.copy()

        lines = ['set frank off',  # remove jsmol logo
                 'set zoomlarge false',  # use the smaller of height/width when setting zoom level
                 'set waitformoveto off']

        point_count = 0
        for ir, r in enumerate(self.rings):
            # skip if the conditions do not apply
            if ring_indices is not None and ir not in ring_indices:
                continue
            if rings_sizes is not None and len(r) not in rings_sizes:
                continue
            if sites is not None and not any(site in r for site in sites):
                continue

            # use fractional coordinates for the center, since the structure is generated as a cif
            # and uses abc+alpha,beta,gamma to define the lattice.
            c = r.get_center(ref_index=0, cart_coords=False)
            color = colors.get(len(r))

            indices = []
            for i, site in enumerate(r.iter_from_index_image(start_index=0)):
                try:
                    indices.append(s.index(site))
                except ValueError:
                    s.append(species=site.species, coords=site.frac_coords,
                             validate_proximity=False, coords_are_cartesian=False)
                    indices.append(len(s) - 1)

            for i in range(len(indices)):
                lines.append(
                    f"draw p{point_count} POLYGON 3 {{{c[0]}/1 {c[1]}/1 {c[2]}/1}} @{indices[i] + 1} @{indices[i - 1] + 1} translucent {color}")
                point_count += 1

        lines = ['load inline "' + s.to("cif").replace('"', "'") + '"'] + lines

        script = ";\n".join(lines)

        if filepath:
            with open(filepath, "wt") as f:
                f.write(script)

        return script

    @requires(jupyter_jsmol, "jupyter_jsmol should be install to show the rings")
    def show_jsmol(self, ring_indices: Union[int, List[int]] = None, rings_sizes: Union[int, List[int]] = None,
                    sites: Union[PeriodicSite, List[PeriodicSite]] = None, colors: dict = None) -> JsmolView:
        """
        Visualizes the structures and colors the selected rings with jsmol.
        If more than one option is given only the rings satisfying all of them will be selected.
        Requires jupyter_jsmol.

        Args:
            ring_indices: filtering option based on the indices of the rings in the list.
            rings_sizes: filtering option based on the size of the rings.
            sites: filtering option based on the sites contained in the rings.
            colors: a dictionary with the size of the rings as keys and a string with the color
                as values.

        Returns:
            a view of the structure from jupyter_jsmol.
        """
        script = self.jmol_script(ring_indices=ring_indices, rings_sizes=rings_sizes, sites=sites, colors=colors)
        view = JsmolView(script=script)
        return view

    def genus(self) -> float:
        """
        Calculates the genus based on the number of rings and the expression:
        \sum_k (1 - k/6)N_k = 2 - 2g

        Returns:
            the genus.
        """

        ds = self.get_num_rings_dict()
        s = 0
        for size, num_rings in ds.items():
            s += (6 - size) * num_rings

        return 1 - s/12
