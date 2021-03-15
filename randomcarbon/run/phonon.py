import numpy as np
import logging
from typing import Union, List
from monty.dev import requires
from pymatgen.core.structure import Structure
from pymatgen.io.phonopy import get_phonopy_structure
from ase.calculators.calculator import Calculator
from ase.atoms import Atoms
from randomcarbon.utils.factory import Factory
try:
    import phonopy
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError:
    phonopy = None
    Phonopy = None


logger = logging.getLogger(__name__)


@requires(phonopy, "phonopy should be installed to calculate phonons")
def get_phonons(structure: Structure, calculator: Union[Calculator, Factory], constraints: list = None,
                supercell_matrix: List[List[int]] = None, primitive_matrix: List[List[float]] = None) -> Phonopy:
    unitcell = get_phonopy_structure(structure)
    if supercell_matrix is None:
        supercell_matrix = np.eye(3)

    if isinstance(calculator, Factory):
        calculator = calculator.generate()

    phonon = Phonopy(unitcell,
                     supercell_matrix=supercell_matrix,
                     primitive_matrix=primitive_matrix)
    phonon.generate_displacements(distance=0.03)
    supercells = phonon.supercells_with_displacements
    supercells_atoms = []
    for sc in supercells:
        a = Atoms(symbols=sc.symbols,
                  positions=sc.positions,
                  masses=sc.masses,
                  cell=sc.cell, pbc=True,
                  constraint=None,
                  calculator=calculator)
        if constraints:
            tmp_constraints = []
            for i, c in enumerate(constraints):
                if isinstance(c, Factory):
                    tmp_constraints.append(c.generate(atoms=a))
                else:
                    tmp_constraints.append(c)
            a.set_constraint(tmp_constraints)
        supercells_atoms.append(a)

    forces = []
    for i, sca in enumerate(supercells_atoms):
        logger.debug(f"calculating forces for supercell {i+1} of {len(supercells_atoms)}")
        forces.append(sca.get_forces())

    phonon.set_forces(forces)
    phonon.produce_force_constants()
    return phonon


@requires(phonopy, "phonopy should be installed to calculate phonons")
def extract_instabilities(phonon: Phonopy, threshold: float = -0.01) -> dict:
    phonon.symmetrize_force_constants()
    freqs, eigvec = phonon.get_frequencies_with_eigenvectors(q=[0, 0, 0])
    neg_ind = np.where(freqs <= threshold)[0]
    if len(neg_ind) == 0:
        info = {
            "has_neg_freqs": False,
            "freqs": None,
            "displ": None
        }
    else:
        displ = eigvec.T.reshape((-1, len(phonon.unitcell), 3))[neg_ind] / phonon.unitcell.masses[:, None]

        info = {
            "has_neg_freqs": False,
            "freqs": freqs[neg_ind],
            "displ": displ[neg_ind]
        }

    return info


@requires(phonopy, "phonopy should be installed to calculate phonons")
def get_instability_info(structure: Structure, calculator: Union[Calculator, Factory],
                         constraints: list = None, threshold: float = -0.01) -> dict:
    phonon = get_phonons(structure=structure, calculator=calculator, constraints=constraints,
                         supercell_matrix=np.eye(3))

    return extract_instabilities(phonon, threshold)


def displace_structure(structure: Structure, displ: List[List[float]], max_displ: float = 0.1) -> Structure:
    norm = np.linalg.norm(displ, axis=-1)
    displ = np.array(displ) * max_displ / norm.max()
    s_trasl = Structure(lattice=structure.lattice, species=structure.species,
                        coords=structure.cart_coords + displ, coords_are_cartesian=True)

    return s_trasl
