import os
import logging
from randomcarbon.utils.structure import add_new_symmetrized_atom, get_struc_min_dist
from pymatgen.core.structure import Structure
from randomcarbon.run.runners import SerialRunner
from randomcarbon.evolution.evolvers.grow import AddSymmAtom
from randomcarbon.evolution.conditions.structure import TemplateDistance
from randomcarbon.utils.factory import Factory
from randomcarbon.evolution.filters.sort import EnergySort
from randomcarbon.data import get_template
from randomcarbon.output.store import MultiJsonStore
from randomcarbon.output.taggers import get_basic_taggers, get_calc_taggers
from ase.calculators.kim.kim import KIM
from ase.spacegroup.symmetrize import FixSymmetry


logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
logger = logging.getLogger("randomcarbon")
logger.setLevel(logging.INFO)

# NB it is important to use a template that is generated according to a standard representation
# of the conventional cell.
template = Structure.from_file(get_template("FAU.cif"))

initial_structure = add_new_symmetrized_atom(template=template)
print("initial min dist: ", get_struc_min_dist(template, initial_structure))

constraints = [Factory(callable=FixSymmetry, set_atoms=True)]
# calculator = Factory(callable=KIM, model_name="Sim_LAMMPS_AIREBO_Morse_OConnorAndzelmRobbins_2015_CH__SM_460187474631_000")
calculator = Factory(callable=KIM, model_name="Tersoff_LAMMPS_Tersoff_1989_SiC__MO_171585019474_002")

evolvers = [AddSymmAtom(template=template, num_structures=10, max_tests=100)]
filters = [EnergySort(calculator=calculator, constraints=constraints)]
blockers = [TemplateDistance(template, max_dist=3)]
# blockers = []

store = MultiJsonStore(os.path.expanduser("~/test_example"))
#NB  here the evolver is added in the tag as an example. Currently all the inputs of the evolvers
# are added to the results DB. This means that the template is also saved in the database
# (and if the AddSymmAtom evolver is present multiple times the results will contain multiple copies
# of the template). While this could be desirable it will also lead to an increase of the size of the DB
# and it is prefereable to avoid adding it in the results.
taggers = get_basic_taggers(template=template, info={"run_name": "example1", "zeolite": "FAU"},
                            calculator=calculator, constraints=constraints) + \
          get_calc_taggers(calculator=calculator, constraints=constraints, optimizer="BFGS",
                           fmax=0.05, evolvers=evolvers, blockers=blockers, filters=filters)

runner = SerialRunner(calculator_factory=calculator, evolvers=evolvers,
                      initial_structure=initial_structure, blockers=blockers, filters=filters,
                      fmax=0.05, steps=1000, constraints=constraints,
                      optimizer="BFGS", store=store, taggers=taggers)

runner.run(max_structures=10)
