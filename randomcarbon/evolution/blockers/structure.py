import logging
from typing import Optional
from randomcarbon.utils.structure import get_struc_min_dist
from randomcarbon.evolution.core import Blocker
from pymatgen.core.structure import Structure

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

