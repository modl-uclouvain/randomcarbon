import logging
import numpy as np
from typing import Optional, Union
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen import Structure
from ase.calculators.calculator import Calculator
from randomcarbon.utils.factory import generate_optimizer, Factory
from randomcarbon.utils.structure import set_properties, get_properties


logger = logging.getLogger(__name__)


def relax(structure: Structure, calculator: Union[Calculator, Factory],
          fmax: float, steps: int = 1000, constraints: list = None,
          optimizer: str = "BFGS", opt_kwargs: dict = None,
          allow_not_converged: bool = False, set_energy_in_structure: bool = False,
          preserve_properties: bool = True) -> Optional[Structure]:
    """
    Helper function to run a relaxation with ASE, based on the typical inputs
    in the package.

    Args:
        structure:
        calculator:
        fmax:
        steps:
        constraints:
        optimizer:
        opt_kwargs:
        allow_not_converged:
        set_energy_in_structure:
        preserve_properties:

    Returns:

    """

    aaa = AseAtomsAdaptor()
    atoms = aaa.get_atoms(structure)

    if isinstance(calculator, Factory):
        calculator = calculator.generate(atoms=atoms)

    if constraints:
        tmp_constraints = []
        for i, c in enumerate(constraints):
            if isinstance(c, Factory):
                tmp_constraints.append(c.generate(atoms=atoms))
            else:
                tmp_constraints.append(c)
        constraints = tmp_constraints

    atoms.calc = calculator
    atoms.set_constraint(constraints)

    dyn = generate_optimizer(atoms=atoms, optimizer=optimizer, opt_kwargs=opt_kwargs)
    converged = dyn.run(fmax=fmax, steps=steps)

    if not converged and not allow_not_converged:
        try:
            final_max_force = np.linalg.norm(atoms.calc.forces, axis=-1).max()
            logger.info(f"relaxation did not converge. Max force : {final_max_force}")
        except:
            logger.info(f"relaxation did not converge.")
        return None

    relaxed_structure = aaa.get_structure(atoms)

    set_properties(relaxed_structure, {"converged": converged})

    if preserve_properties:
        set_properties(relaxed_structure, get_properties(structure))

    if set_energy_in_structure:
        set_properties(relaxed_structure, {"energy": atoms.calc.results["energy"]})

    return relaxed_structure


def get_energy(structure: Structure, calculator: Union[Calculator, Factory],
               constraints: list = None, set_in_structure: bool = False) -> Optional[Structure]:
    """
        Helper function to get the energy with ASE, based on the typical inputs
        in the package.

    Args:
        structure:
        calculator:
        constraints:
        set_in_structure:

    Returns:

    """
    aaa = AseAtomsAdaptor()
    atoms = aaa.get_atoms(structure)

    if isinstance(calculator, Factory):
        calculator = calculator.generate(atoms=atoms)

    if constraints:
        tmp_constraints = []
        for i, c in enumerate(constraints):
            if isinstance(c, Factory):
                tmp_constraints.append(c.generate(atoms=atoms))
            else:
                tmp_constraints.append(c)
        constraints = tmp_constraints

    atoms.calc = calculator
    atoms.set_constraint(constraints)

    energy = atoms.get_potential_energy()
    if set_in_structure:
        set_properties(structure, {"energy": energy})

    return energy
