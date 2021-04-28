import logging
from typing import Optional, Union
from randomcarbon.utils.structure import get_struc_min_dist
from randomcarbon.evolution.core import Blocker
from randomcarbon.utils.factory import Factory
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import NearNeighbors

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
