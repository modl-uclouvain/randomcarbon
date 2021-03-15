import numpy as np
import os
import logging
from randomcarbon.utils.structure import add_new_symmetrized_atom, get_struc_min_dist
from pymatgen.core.structure import Structure
from randomcarbon.run.runners import BranchingParallelRunner
from randomcarbon.evolution.evolvers.grow import AddSymmAtom
from randomcarbon.evolution.blockers.structure import MinTemplateDistance
from randomcarbon.evolution.blockers.energy import EnergyAtoms
from randomcarbon.utils.factory import Factory
from randomcarbon.evolution.filters.limit import MaxEnergyPerAtom
from randomcarbon.evolution.filters.sort import EnergySort
from randomcarbon.data import get_template
from randomcarbon.output.store import  MongoStore
from randomcarbon.output.taggers import get_basic_taggers
from ase.calculators.kim.kim import KIM
from ase.spacegroup.symmetrize import FixSymmetry


logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
logger = logging.getLogger("randomcarbon")
logger.setLevel(logging.INFO)

template = Structure.from_file(get_template("FAU_zeolite_conv_std.cif"))

# transformation from 227 to 224
supergroup_transf = ((np.eye(3) * 2), [0] * 3)

initial_structure = add_new_symmetrized_atom(template=template, spacegroup=224, supergroup_transf=supergroup_transf,
                                             min_dist_from_template=2.)

print("initial min dist: ", get_struc_min_dist(template, initial_structure))

constraints = [Factory(callable=FixSymmetry, set_atoms=True)]
# calculator = Factory(callable=KIM, model_name="Sim_LAMMPS_AIREBO_Morse_OConnorAndzelmRobbins_2015_CH__SM_460187474631_000")
calculator = Factory(callable=KIM, model_name="Tersoff_LAMMPS_Tersoff_1989_SiC__MO_171585019474_002")

evolvers = [AddSymmAtom(template=template, num_structures=3, max_tests=100, spacegroup=224, supergroup_transf=supergroup_transf)]
filters = [MaxEnergyPerAtom(calculator=calculator, max_energy=0), EnergySort(calculator=calculator, constraints=constraints)]
blockers = [MinTemplateDistance(template, min_dist=3),
            EnergyAtoms(criteria={400: -3, 600: -4, 700: -5}, calculator=calculator)]
taggers = get_basic_taggers(template=template, info={"run_name": "example5", "zeolite": "FAU", "group": 227, "supergroup": 224},
                            calculator=calculator, constraints=constraints)

store = MongoStore(database="db_name", collection_name="collection_name", host="host", port=27017,
                   username="username", password="password")

runner = BranchingParallelRunner(calculator_factory=calculator, evolvers=evolvers,
                                 initial_structures=initial_structure, blockers=blockers, filters=filters,
                                 fmax=0.05, steps=1000, constraints=constraints, optimizer="BFGS",
                                 opt_kwargs={"logfile": None}, store=store, taggers=taggers,
                                 spacegroup_primitive=224)

runner.run(nprocs=2, max_structures=30)
