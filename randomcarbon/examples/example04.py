import os
import logging
from randomcarbon.utils.structure import add_new_symmetrized_atom, get_struc_min_dist
from pymatgen.core.structure import Structure
from randomcarbon.run.runners import BranchingParallelRunner
from randomcarbon.evolution.evolvers.grow import AddSymmAtom
from randomcarbon.evolution.blockers.structure import MinTemplateDistance
from randomcarbon.utils.factory import Factory
from randomcarbon.evolution.filters.limit import MaxEnergyPerAtom
from randomcarbon.evolution.filters.sort import EnergySort
from randomcarbon.data import get_template
from randomcarbon.output.store import MultiJsonStore
from randomcarbon.output.taggers import get_basic_taggers
from ase.calculators.kim.kim import KIM
from ase.spacegroup.symmetrize import FixSymmetry


logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
logger = logging.getLogger("randomcarbon")
logger.setLevel(logging.INFO)

template = Structure.from_file(get_template("FAU_zeolite_conv_std.cif"))

initial_structure = add_new_symmetrized_atom(template=template, spacegroup=216)
print("initial min dist: ", get_struc_min_dist(template, initial_structure))

constraints = [Factory(callable=FixSymmetry, set_atoms=True)]
# calculator = Factory(callable=KIM, model_name="Sim_LAMMPS_AIREBO_Morse_OConnorAndzelmRobbins_2015_CH__SM_460187474631_000")
calculator = Factory(callable=KIM, model_name="Tersoff_LAMMPS_Tersoff_1989_SiC__MO_171585019474_002")

evolvers = [AddSymmAtom(template=template, num_structures=10, max_tests=100, spacegroup=216)]
filters = [MaxEnergyPerAtom(calculator=calculator, max_energy=0), EnergySort(calculator=calculator, constraints=constraints)]
blockers = [MinTemplateDistance(template, min_dist=3)]
taggers = get_basic_taggers(template=template, info={"run_name": "example4", "zeolite": "FAU", "subgroup": 216},
                            calculator=calculator, constraints=constraints)

store = MultiJsonStore(os.path.expanduser("~/test_example"))

runner = BranchingParallelRunner(calculator_factory=calculator, evolvers=evolvers,
                                 initial_structures=initial_structure, blockers=blockers, filters=filters,
                                 fmax=0.05, steps=1000, constraints=constraints, optimizer="BFGS",
                                 opt_kwargs={"logfile": None}, store=store, taggers=taggers,
                                 spacegroup_primitive=216)

runner.run(nprocs=2, max_structures=30)
