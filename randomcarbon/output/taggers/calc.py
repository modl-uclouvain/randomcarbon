import yaml
import numpy as np
from typing import List, Union
from monty.dev import requires
from pymatgen.core.structure import Structure
from randomcarbon.output.taggers.core import Tagger
from randomcarbon.run.ase import get_energy, relax
from randomcarbon.run.phonon import get_phonons, extract_instabilities
from randomcarbon.utils.factory import Factory
from randomcarbon.utils.structure import get_property, has_low_energy
from randomcarbon.evolution.core import Evolver, Filter, Blocker
try:
    import phonopy
    from phonopy.interface.phonopy_yaml import PhonopyYaml
except ImportError:
    phonopy = None


class CalculationInfoTag(Tagger):

    def __init__(self, calculator: Factory, optimizer: str = None,
                 constraints: list = None, fmax: float = None):
        self.calculator = calculator
        self.optimizer = optimizer
        self.constraints = constraints
        self.fmax = fmax

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        d = {}
        dc = self.calculator.as_dict()
        dc.pop("@class", None)
        dc.pop("@module", None)

        d["calculator"] = dc
        d["constraints"] = self.constraints
        d["optimizer"] = self.optimizer
        d["fmax"] = self.fmax

        doc["calculation"] = d

        return doc


class EvolutionInfoTag(Tagger):

    def __init__(self, evolvers: List[Union[Evolver, List]] = None,
                 blockers: List[Blocker] = None,
                 filters: List[Filter] = None):
        self.evolvers = evolvers
        self.blockers = blockers
        self.filters = filters

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        doc["evolution"] = dict(
            evolvers=self.evolvers,
            blockers=self.blockers,
            filters=self.filters
        )

        return doc


class EnergyTag(Tagger):

    def __init__(self, calculator: Factory,
                 constraints: list = None):
        self.calculator = calculator
        self.constraints = constraints

    def tag(self, doc: dict, structure: Structure = None) -> dict:

        energy = get_property(structure, "energy")

        if energy is None:
            if not self.calculator:
                return doc
            energy = get_energy(structure=structure, calculator=self.calculator,
                                constraints=self.constraints, set_in_structure=True)

        doc["energy"] = energy
        doc["energy_per_atom"] = energy / len(structure)

        return doc


class RelaxTag(Tagger):
    def __init__(self, calculator: Factory, energy_threshold: float = None, constraints: list = None,
                 fmax: float = 0.05, steps: int = 1000, optimizer: str = "BFGS",
                 opt_kwargs: dict = None, prefix: str = "", store_structure: bool = True):
        self.calculator = calculator
        self.energy_threshold = energy_threshold
        self.constraints = constraints
        self.fmax = fmax
        self.steps = steps
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.prefix = prefix
        self.store_structure = store_structure

    def tag(self, doc: dict, structure: Structure = None) -> dict:

        if self.energy_threshold is not None:
            if has_low_energy(structure, self.energy_threshold):
                return doc

        relaxed = relax(structure=structure, calculator=self.calculator,
                        constraints=self.constraints, fmax=self.fmax, steps=self.steps,
                        optimizer=self.optimizer, opt_kwargs=self.opt_kwargs,
                        allow_not_converged=False, set_energy_in_structure= True)

        if not relaxed:
            return doc

        energy = get_property(relaxed, "energy")
        energy_per_atom = energy / len(structure)

        doc[f"{self.prefix}energy_per_atom"] = energy_per_atom
        if self.store_structure:
            doc[f"{self.prefix}structure"] = relaxed

        return doc


@requires(phonopy, "phonopy should be installed to calculate phonons")
class PhononTag(Tagger):
    def __init__(self, calculator: Factory, energy_threshold: float = None, constraints: list = None,
                 phfreqs_threshold: float = -0.01, store_phonopy_data: bool = False):
        self.calculator = calculator
        self.energy_threshold = energy_threshold
        self.constraints = constraints
        self.phfreqs_threshold = phfreqs_threshold
        self.store_phonopy_data = store_phonopy_data

    def tag(self, doc: dict, structure: Structure = None) -> dict:

        if self.energy_threshold is not None:
            if has_low_energy(structure, self.energy_threshold):
                return doc

        phonon = get_phonons(structure=structure, calculator=self.calculator,
                             constraints=self.constraints, supercell_matrix=np.eys(3))

        info = extract_instabilities(phonon=phonon, threshold=self.phfreqs_threshold)

        if self.store_phonopy_data:
            phpy_yaml = PhonopyYaml(settings={})
            phpy_yaml.set_phonon_info(phonon)
            info["phonopy_data"] = yaml.safe_load(str(phpy_yaml))

        return doc
