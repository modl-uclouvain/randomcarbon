from typing import List
from monty.json import MSONable
from pymatgen.core.structure import Structure
from randomcarbon.evolution.core import Evolver


class Generator(MSONable):

    def __init__(self, evolver: Evolver, structure: Structure = None):
        self.evolver = evolver
        self.structure = structure

    def generate(self) -> List[Structure]:
        return self.evolver.evolve(self.structure)
