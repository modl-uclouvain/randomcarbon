import logging
from typing import List, Optional, Union
from randomcarbon.utils.structure import get_struc_min_dist
from randomcarbon.evolution.core import Blocker
from randomcarbon.utils.factory import Factory
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import NearNeighbors
from randomcarbon.output.taggers import RingsStatsTag
from randomcarbon.rings.input import RingMethod, RingsInput
from randomcarbon.rings.run import run_rings
import math
from randomcarbon.utils.structure import get_properties, set_properties, get_property
logger = logging.getLogger(__name__)


class MinTemplateDistance(Blocker):

    def __init__(self, template: Structure, min_dist: float = 3):
        self.template = template
        self.min_dist = min_dist

    def block(self, structure: Structure) -> Optional[str]:
        d = get_struc_min_dist(self.template, structure)
        logger.debug(f"MinTemplateDistance: actual min dist {d}, selected min dist {self.min_dist}")
        if d < self.min_dist:
            return f"{self.__class__.__name__}. dist: {d}, max allowed: {self.min_dist}"
        return None


class MaxNumAtoms(Blocker):
    def __init__(self, num_atoms):
        self.num_atoms = num_atoms

    def block(self, structure: Structure) -> Optional[str]:
        if len(structure) > self.num_atoms:
            return f"{self.__class__.__name__}. num at: {len(structure)}, max allowed: {self.num_atoms}"
        return None


class MaxNumNeighbors(Blocker):

    def __init__(self, nn: Union[Factory, NearNeighbors], max_neighbors):
        self.nn = nn
        self.max_neighbors = max_neighbors

    def block(self, structure: Structure) -> Optional[str]:
        nn = self.nn
        if isinstance(nn, Factory):
            nn = nn.generate()

        for i in range(len(structure)):
            num_nn = nn.get_cn(structure, i)
            if num_nn > self.max_neighbors:
                return f"{self.__class__.__name__}. num nn: {num_nn}, max allowed: {self.max_neighbors}"

        return None
class PolygonBlocker(Blocker):
    """
    Blocker function that will stop the evolution of the structures if polygons with a given number of sides equals to the parameter nsides than the  is detected
    inside the structure. It uses the R.I.N.G.S utility to detect the rings and their size. It adds the rings statistics to the structure 
    as well as the 'rings' property.  
    """
    def __init__(self,method: Union[RingMethod, int]=5, lattice_matrix: bool = True,
                  cutoff_rad: Union[dict, NearNeighbors] = {("C", "C"): 1.9},
                 grmax: float = None, executable: str = "rings", irreducible: bool = True, nsides: List[int] = [3,4]):
        
        self.nsides = nsides
        self.method = method
        self.lattice_matrix = lattice_matrix
        self.maximum_search_depth = int(math.ceil(max(nsides)/2))
        self.cutoff_rad = cutoff_rad
        self.grmax = grmax
        self.executable = executable
        self.irreducible = irreducible
    def block(self, structure: Structure) -> Optional[str]:
        
        inp = RingsInput(structure=structure, methods=[self.method], lattice_matrix=self.lattice_matrix,
                         maximum_search_depth=self.maximum_search_depth, cutoff_rad=self.cutoff_rad,
                         grmax=self.grmax) 
        
        out = run_rings(inp, executable=self.executable, irreducible=self.irreducible)
        
        if not out:
            sid = get_property(structure, "structure_id")
            logger.warning(f"no output produced by rings for structure {sid}, polygon blocker")
            return None
        
        rings_list = out[self.method]
        stats = rings_list.get_stats_dict()
        size = stats["size"]
        size = [i for i in size if i not in [0, 1, 2]]
        unwanted= [i for i in size if i in self.nsides]
        if unwanted !=None:
            return f"{self.__class__.__name__}. {unwanted}-gon detected in the structure"

        properties = get_properties(structure)
        properties['rings']= stats
        structure = set_properties(structure, {'rings': stats})

        return None
        