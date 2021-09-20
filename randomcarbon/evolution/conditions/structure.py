import logging
from typing import List, Optional, Tuple, Union
from randomcarbon.utils.structure import get_struc_min_dist
from randomcarbon.evolution.core import Condition
from randomcarbon.utils.factory import Factory
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import NearNeighbors
from randomcarbon.rings.input import RingMethod, RingsInput
from randomcarbon.rings.run import run_rings
import math
from randomcarbon.utils.structure import set_properties, get_property

logger = logging.getLogger(__name__)


class TemplateDistance(Condition):
    """
    Condition to verify if the distance from the template is a specific range
    """

    def __init__(self, template: Structure, min_dist: float = None, max_dist: float = None):
        self.template = template
        self.min_dist = min_dist
        self.max_dist = max_dist

    def satisfied(self, structure: Structure) -> Tuple[bool, Optional[str]]:
        d = get_struc_min_dist(self.template, structure)
        logger.debug(f"TemplateDistance: actual min dist {d}")
        if self.min_dist is not None and d < self.min_dist:
            return False, f"{self.__class__.__name__}. distance {d} lower than: {self.min_dist}"

        if self.max_dist is not None and d > self.max_dist:
            return False, f"{self.__class__.__name__}. distance {d} larger than: {self.max_dist}"

        return True, f"{self.__class__.__name__}. distance {d} in range: {self.min_dist}, {self.max_dist}"


class NumAtoms(Condition):
    """
    Condition checking that the number of sites of a structure is in a specific range.
    """

    def __init__(self, min_sites: int = None, max_sites: int = None):
        self.min_sites = min_sites
        self.max_sites = max_sites

    def satisfied(self, structure: Structure) -> Tuple[bool, Optional[str]]:
        nsites = len(structure)

        if self.min_sites is not None and nsites < self.min_sites:
            return False, f"{self.__class__.__name__}. n sites: {nsites}, lower than: {self.min_sites}"

        if self.max_sites is not None and nsites > self.max_sites:
            return False, f"{self.__class__.__name__}. n sites: {nsites}, larger than: {self.max_sites}"

        return True, f"{self.__class__.__name__}. n sites {nsites} in range: {self.min_sites}, {self.max_sites}"


class NumNeighbors(Condition):
    """
    Condition to determine if for all the atoms in the structure they have a number of
    neighbors in a specific range.
    """

    def __init__(self, nn: Union[Factory, NearNeighbors], min_neighbors: int = None, max_neighbors: int = None):
        self.nn = nn
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors

    def satisfied(self, structure: Structure) -> Tuple[bool, Optional[str]]:
        nn = self.nn
        if isinstance(nn, Factory):
            nn = nn.generate()

        for i in range(len(structure)):
            num_nn = nn.get_cn(structure, i)
            if self.min_neighbors is not None and num_nn < self.min_neighbors:
                return False, f"{self.__class__.__name__}. num nn: {num_nn}, lower than: {self.min_neighbors}"

            if self.max_neighbors is not None and num_nn > self.max_neighbors:
                return False, f"{self.__class__.__name__}. num nn: {num_nn}, larger than: {self.max_neighbors}"

        return True, f"{self.__class__.__name__}. num nn {nn} in range: {self.min_neighbors}, {self.max_neighbors}"


class AnyPolygonSize(Condition):
    """
    Condition that determines if there is any polygon in the structure that matches one of the
    values given list of nsides. It uses  the R.I.N.G.S utility to detect the rings and their size.
    It adds the rings statistics to the structure as well as the 'rings' property.
    """

    def __init__(self, nsides: List[int] = None,
                 method: Union[RingMethod, int] = 5, lattice_matrix: bool = True,
                 cutoff_rad: Union[dict, NearNeighbors] = None, grmax: float = None,
                 executable: str = "rings", irreducible: bool = True,
                 value_for_undetermined: bool = True):

        self.nsides = [3, 4] if nsides is None else nsides

        self.method = method
        self.lattice_matrix = lattice_matrix
        self.maximum_search_depth = int(math.ceil(max(self.nsides) / 2))
        self.cutoff_rad = {("C", "C"): 1.9} if cutoff_rad is None else cutoff_rad
        self.grmax = grmax
        self.executable = executable
        self.irreducible = irreducible
        self.value_for_undetermined = value_for_undetermined

    def satisfied(self, structure: Structure) -> Tuple[bool, Optional[str]]:

        rings_prop = get_property(structure, "rings")
        inp = RingsInput(structure=structure, methods=[self.method], lattice_matrix=self.lattice_matrix,
                         maximum_search_depth=self.maximum_search_depth, cutoff_rad=self.cutoff_rad,
                         grmax=self.grmax)

        if rings_prop and rings_prop["rings_input"] == inp:
            d = rings_prop["stats"]
        else:
            out = run_rings(inp, executable=self.executable, irreducible=self.irreducible)

            if not out:
                sid = get_property(structure, "structure_id")
                logger.warning(f"no output produced by rings for structure {sid}, polygon condition")
                return self.value_for_undetermined, "No output from rings"

            d = out[self.method].get_stats_dict()
            set_properties(structure, {'rings': {'stats': d, 'rings_input': inp}})
        if any(ns in d['size'] for ns in self.nsides):
            return True, f"{self.__class__.__name__}. {[ns for ns in self.nsides if ns in d['size']]}-gon detected in the structure"

        return False, f"{self.__class__.__name__}. {[ns for ns in self.nsides if ns in d['size']]}-gon not detected in the structure"
