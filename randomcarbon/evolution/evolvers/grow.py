from typing import Union, List, Tuple
from randomcarbon.evolution.core import Evolver, Condition
from randomcarbon.utils.structure import add_new_symmetrized_atom, add_c2_symmetrized
from randomcarbon.utils.structure import add_new_symmetrized_atom_bridge, add_new_symmetrized_atom_undercoord
from pymatgen.core.structure import Structure
from pymatgen.core.operations import SymmOp


class AddSymmAtom(Evolver):
    """
    Evolver that adds one symmetrized atom to the existing structure
    based on the template. The template should be the same as the one
    used to generate the structure initially.
    """

    def __init__(self, template: Structure, num_structures: int = 5,
                 spacegroup: Union[str, int] = None, symm_ops: List[SymmOp] = None,
                 specie: str = "C", max_dist_eq: float = 0.1, min_dist_current: float = 1.2,
                 max_dist_current: float = 1.6, min_dist_from_template: float = 3,
                 max_dist_from_template: float = None, max_tests: int = 1000,
                 supergroup_transf: Tuple[List, List] = None, symprec: float = 0.001,
                 angle_tolerance: float = 5.0, num_atoms: int = 1, conditions: List[Condition] = None):
        super().__init__(conditions)
        self.template = template
        self.num_structures = num_structures
        self.spacegroup = spacegroup
        self.symm_ops = symm_ops
        self.specie = specie
        self.max_dist_eq = max_dist_eq
        self.min_dist_current = min_dist_current
        self.max_dist_current = max_dist_current
        self.min_dist_from_template = min_dist_from_template
        self.max_dist_from_template = max_dist_from_template
        self.max_tests = max_tests
        self.supergroup_transf = supergroup_transf
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.num_atoms = num_atoms

    def _evolve(self, structure: Structure) -> List[Structure]:
        new_structures = []
        for i in range(self.num_structures):
            ns = structure
            for j in range(self.num_atoms):
                ns = add_new_symmetrized_atom(template=self.template, spacegroup=self.spacegroup, symm_ops=self.symm_ops,
                                              current=ns, specie=self.specie, max_dist_eq=self.max_dist_eq,
                                              min_dist_current=self.min_dist_current, max_dist_current=self.max_dist_current,
                                              min_dist_from_template=self.min_dist_from_template,
                                              max_dist_from_template=self.max_dist_from_template, max_tests=self.max_tests,
                                              supergroup_transf=self.supergroup_transf, symprec=self.symprec,
                                              angle_tolerance=self.angle_tolerance)

            if ns:
                new_structures.append(ns)

        return new_structures


class AddSymmC2(Evolver):
    """
    Evolver that adds two C atoms symmetrized to the existing structure
    based on the template. The template should be the same as the one
    used to generate the structure initially.
    """

    def __init__(self, template: Structure, num_structures: int = 5, spacegroup: Union[str, int] = None,
                 supergroup_transf: Tuple[List, List] = None,
                 symm_ops: List[SymmOp] = None, max_dist_eq: float = 0.1,
                 min_dist_current: Union[List[float], float] = 1.2, max_dist_current: Union[List[float], float] = 1.6,
                 min_dist_from_template: float = 3, max_dist_from_template: float = None,
                 min_dist_cc: float = 1.2, max_dist_cc: float = 1.6, max_tests: int = 1000,
                 symprec: float = 0.001, angle_tolerance: float = 5.0, conditions: List[Condition] = None):
        super().__init__(conditions)
        self.template = template
        self.num_structures = num_structures
        self.spacegroup = spacegroup
        self.symm_ops = symm_ops
        self.max_dist_eq = max_dist_eq
        self.min_dist_current = min_dist_current
        self.max_dist_current = max_dist_current
        self.min_dist_from_template = min_dist_from_template
        self.max_dist_from_template = max_dist_from_template
        self.min_dist_cc = min_dist_cc
        self.max_dist_cc = max_dist_cc
        self.max_tests = max_tests
        self.supergroup_transf = supergroup_transf
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def _evolve(self, structure: Structure) -> List[Structure]:
        new_structures = []
        for i in range(self.num_structures):
            ns = add_c2_symmetrized(template=self.template, spacegroup=self.spacegroup, symm_ops=self.symm_ops,
                                    current=structure, max_dist_eq=self.max_dist_eq,
                                    min_dist_current=self.min_dist_current, max_dist_current=self.max_dist_current,
                                    min_dist_from_template=self.min_dist_from_template,
                                    max_dist_from_template=self.max_dist_from_template, min_dist_cc=self.min_dist_cc,
                                    max_dist_cc=self.max_dist_cc, max_tests=self.max_tests,
                                    supergroup_transf=self.supergroup_transf, symprec=self.symprec,
                                    angle_tolerance=self.angle_tolerance)

            if ns:
                new_structures.append(ns)

        return new_structures


class AddSymmAtomUndercoord(Evolver):
    """
    Evolver that adds one symmetrized atom close to an undercoordinated atoms of an
    existing structure based on the template. The template should be the same as the one
    used to generate the structure initially.
    """

    def __init__(self, template: Structure, num_structures: int = 5,
                 cutoff: float = 1.8, min_neighbors: int = 3, spacegroup: Union[str, int] = None,
                 min_r: float = 1.2, max_r: float = 1.9, symm_ops: List[SymmOp] = None,
                 specie: str = "C", max_dist_eq: float = 0.1, min_dist_current: float = 1.2,
                 max_dist_current: float = 1.6, min_dist_from_template: float = 3,
                 max_dist_from_template: float = None, max_tests: int = 1000,
                 supergroup_transf: Tuple[List, List] = None, symprec: float = 0.001,
                 angle_tolerance: float = 5.0, num_atoms: int = 1, conditions: List[Condition] = None):
        super().__init__(conditions)
        self.template = template
        self.num_structures = num_structures
        self.cutoff = cutoff
        self.min_neighbors = min_neighbors
        self.min_r = min_r
        self.max_r = max_r
        self.spacegroup = spacegroup
        self.symm_ops = symm_ops
        self.specie = specie
        self.max_dist_eq = max_dist_eq
        self.min_dist_current = min_dist_current
        self.max_dist_current = max_dist_current
        self.min_dist_from_template = min_dist_from_template
        self.max_dist_from_template = max_dist_from_template
        self.max_tests = max_tests
        self.supergroup_transf = supergroup_transf
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.num_atoms = num_atoms

    def _evolve(self, structure: Structure) -> List[Structure]:
        new_structures = []
        for i in range(self.num_structures):
            ns = structure
            for j in range(self.num_atoms):
                ns = add_new_symmetrized_atom_undercoord(
                    template=self.template, cutoff=self.cutoff, min_neighbors=self.min_neighbors,
                    min_r=self.min_r, max_r=self.max_r, spacegroup=self.spacegroup, symm_ops=self.symm_ops,
                    current=ns, specie=self.specie, max_dist_eq=self.max_dist_eq,
                    min_dist_current=self.min_dist_current, max_dist_current=self.max_dist_current,
                    min_dist_from_template=self.min_dist_from_template,
                    max_dist_from_template=self.max_dist_from_template, max_tests=self.max_tests,
                    supergroup_transf=self.supergroup_transf, symprec=self.symprec,
                    angle_tolerance=self.angle_tolerance)

            if ns:
                new_structures.append(ns)

        return new_structures


class AddSymmAtomBridge(Evolver):
    """
    Evolver that adds one symmetrized atom bridging two undercoordinated atoms of an
    existing structure if they exist. The template, that should be the same as the one
    used to generate the structure initially, will also be considered.
    """

    def __init__(self, template: Structure, num_structures: int = 5,
                 cutoff: float = 1.8, min_neighbors: int = 3, spacegroup: Union[str, int] = None,
                 max_r: float = 1.9, symm_ops: List[SymmOp] = None,
                 min_dist_pair: float = 3, max_dist_pair: float = 4.,
                 specie: str = "C", max_dist_eq: float = 0.1, min_dist_current: float = 1.2,
                 max_dist_current: float = 1.6, min_dist_from_template: float = 3,
                 max_dist_from_template: float = None, max_tests: int = 1000,
                 supergroup_transf: Tuple[List, List] = None, symprec: float = 0.001,
                 angle_tolerance: float = 5.0, num_atoms: int = 1, conditions: List[Condition] = None):
        super().__init__(conditions)
        self.template = template
        self.num_structures = num_structures
        self.cutoff = cutoff
        self.min_neighbors = min_neighbors
        self.max_r = max_r
        self.min_dist_pair = min_dist_pair
        self.max_dist_pair = max_dist_pair
        self.spacegroup = spacegroup
        self.symm_ops = symm_ops
        self.specie = specie
        self.max_dist_eq = max_dist_eq
        self.min_dist_current = min_dist_current
        self.max_dist_current = max_dist_current
        self.min_dist_from_template = min_dist_from_template
        self.max_dist_from_template = max_dist_from_template
        self.max_tests = max_tests
        self.supergroup_transf = supergroup_transf
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.num_atoms = num_atoms

    def _evolve(self, structure: Structure) -> List[Structure]:
        new_structures = []
        for i in range(self.num_structures):
            ns = structure
            for j in range(self.num_atoms):
                ns = add_new_symmetrized_atom_bridge(
                    template=self.template, cutoff=self.cutoff, min_neighbors=self.min_neighbors,
                    max_r=self.max_r, spacegroup=self.spacegroup, symm_ops=self.symm_ops,
                    min_dist_pair=self.min_dist_pair, max_dist_pair=self.max_dist_pair,
                    current=ns, specie=self.specie, max_dist_eq=self.max_dist_eq,
                    min_dist_current=self.min_dist_current, max_dist_current=self.max_dist_current,
                    min_dist_from_template=self.min_dist_from_template,
                    max_dist_from_template=self.max_dist_from_template, max_tests=self.max_tests,
                    supergroup_transf=self.supergroup_transf, symprec=self.symprec,
                    angle_tolerance=self.angle_tolerance)

            if ns:
                new_structures.append(ns)

        return new_structures
