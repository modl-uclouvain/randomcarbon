from typing import Union, List, Tuple, Optional
import random
import numpy as np
from pymatgen.core import Structure, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.calculator import Calculator
from ase.ga.offspring_creator import OffspringCreator
from randomcarbon.evolution.core import Evolver
from randomcarbon.evolution.evolvers.grow import AddSymmAtom, AddSymmAtomUndercoord
from randomcarbon.utils.symmetry import unequivalent_ind
from randomcarbon.utils.structure import structure_from_symmops, get_struc_min_dist
from randomcarbon.utils.factory import generate_optimizer
from randomcarbon.run.ase import relax
from randomcarbon.utils.symmetry import get_inequivalent_site_representative
from randomcarbon.utils.structure import structure_from_symmetries, get_struc_min_dist
import spglib
from ase import Atoms
from ase.ga.population import Population
from scipy.optimize import linear_sum_assignment


class RandomGenerator:

    def __init__(self, n_atoms: Union[int, Tuple[int]], evolver: Evolver, calculator: Calculator = None,
                 constraints: List = None, min_dist_template: float = 1.8, template: Structure = None,
                 max_tests: int = 50):
        self.n_atoms = n_atoms
        self.evolver = evolver
        self.calculator = calculator
        self.constraints = constraints
        self.min_dist_template = min_dist_template
        self.template = template
        self.max_tests = max_tests

    def generate_n_atoms(self):
        if isinstance(self.n_atoms, int):
            return self.n_atoms
        else:
            return np.random.randint(self.n_atoms[0], self.n_atoms[1] + 1)

    def get_new_individual(self):
        for _ in range(self.max_tests):
            n_atoms = self.generate_n_atoms()
            current = None
            for i in range(n_atoms):
                generated = self.evolver.evolve(current)
                if not generated:
                    current = None
                    break
                current = generated[0]
                if self.calculator:
                    current = relax(current, calculator=self.calculator, fmax=0.05, constraints=self.constraints,
                                    opt_kwargs={"logfile": None}, allow_not_converged=True, set_energy_in_structure=True)
                                    # allow_not_converged=True)

                if self.template:
                    d = get_struc_min_dist(self.template, current)
                    if d < self.min_dist_template:
                        current = None
                        break

            if current:
                return current

        return None
